"""고정 가상 입력으로 기존 CPU 분할투자 구현을 재현한다. DB/GPU 접속 없음."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.backtest.cpu.backtester import BacktestEngine
from src.backtest.cpu.execution import BasicExecutionHandler
from src.backtest.cpu.portfolio import Portfolio
from src.backtest.cpu.strategy import MagicSplitStrategy


TICKER = "SYNTH001"
TRADING_DATES = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                 "2024-01-08", "2024-01-09", "2024-01-10")
OHLC = ((10000, 10100, 9900, 10000), (10000, 10100, 9400, 9500),
        (9400, 9600, 9300, 9400), (10000, 10500, 9900, 10400),
        (10400, 11200, 10300, 11100), (11100, 11200, 11000, 11100),
        (11100, 11200, 11000, 11100))
PRICE_FIELDS = ("open_price", "high_price", "low_price", "close_price")
INDICATORS = {"atr_14_ratio": 0.1, "market_cap": 1_000_000_000,
              "cheap_score": 0.8, "cheap_score_confidence": 1.0, "flow5_mcap": 0.01}
INITIAL_CASH = 1_000_000
# 제품 설정과 구분되는 이 합성 시나리오만의 고정 조건.
STRATEGY_SETTINGS = dict(
    max_stocks=1, order_investment_ratio=0.1, additional_buy_drop_rate=0.05,
    sell_profit_rate=0.1, backtest_start_date=TRADING_DATES[0],
    backtest_end_date=TRADING_DATES[-1], additional_buy_priority="lowest_order",
    cooldown_period_days=5, stop_loss_rate=-0.15, max_splits_limit=2,
    max_inactivity_period=90, candidate_source_mode="tier", use_weekly_alpha_gate=False,
    min_liquidity_20d_avg_value=0, min_tier12_coverage_ratio=0.0,
    tier_hysteresis_mode="strict_hysteresis_v1", candidate_lookup_error_policy="raise",
    buy_commission_rate=0.00015,
)
EXECUTION_SETTINGS = dict(buy_commission_rate=0.00015, sell_commission_rate=0.00015,
                          sell_tax_rate=0.0018)
TRADE_FIELDS = ("trade_date", "signal_date", "ticker", "side", "split", "quantity",
                "price", "commission", "tax", "cash_after")
DAILY_FIELDS = ("date", "cash", "quantity", "market_value", "total_value")
EXPECTED_TRADES = (
    ("2024-01-03", "2024-01-02", TICKER, "BUY", 1, 10, 10000, 15, 0, 899985),
    ("2024-01-04", "2024-01-03", TICKER, "BUY", 2, 10, 9400, 14, 0, 805971),
    ("2024-01-08", "2024-01-05", TICKER, "SELL", 2, 10, 10400, 15, 188, 909768),
    ("2024-01-09", "2024-01-08", TICKER, "SELL", 1, 10, 11100, 16, 201, 1020551),
)
EXPECTED_DAILY = tuple(zip(
    TRADING_DATES, (1000000, 899985, 805971, 805971, 909768, 1020551, 1020551),
    (0, 10, 20, 20, 10, 0, 0), (0, 95000, 188000, 208000, 111000, 0, 0),
    (1000000, 994985, 993971, 1013971, 1020768, 1020551, 1020551),
))
EXPECTED_SUMMARY = dict(initial_cash=1000000, final_cash=1020551, final_value=1020551,
                        total_commission=60, total_tax=389, net_pnl=20551)


def make_price_rows():
    return [dict(date=date, **dict(zip(PRICE_FIELDS, values)), **INDICATORS)
            for date, values in zip(TRADING_DATES, OHLC)]


def make_tier_rows():
    return [dict(date=date, tier=1) for date in TRADING_DATES]


def _date_frame(rows, label):
    frame = pd.DataFrame(rows).copy()
    if "date" not in frame:
        raise ValueError(f"{TICKER} {label}: date 필드가 없습니다.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError(f"{TICKER} {label}: 유효하지 않거나 중복된 날짜입니다.")
    frame = frame.set_index("date").sort_index()
    expected = pd.DatetimeIndex(pd.to_datetime(TRADING_DATES), name="date")
    if not frame.index.equals(expected):
        missing = [str(d.date()) for d in expected.difference(frame.index)]
        raise ValueError(f"{TICKER} {label}: 7거래일 입력 불일치, 누락={missing}")
    return frame


def _number(value, date, field):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{TICKER} {date} {field}: 숫자가 아닙니다.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{TICKER} {date} {field}: 유한한 값이 필요합니다.")
    return number


def _validate_prices(frame):
    fields = (*PRICE_FIELDS, *INDICATORS)
    if missing := set(fields).difference(frame.columns):
        raise ValueError(f"{TICKER} 가격: 필수 필드 누락={sorted(missing)}")
    for date, row in frame.iterrows():
        values = {field: _number(row[field], date.date(), field) for field in fields}
        if any(values[field] <= 0 for field in (*PRICE_FIELDS, "atr_14_ratio", "market_cap")):
            raise ValueError(f"{TICKER} {date.date()}: 가격·ATR·시가총액은 양수여야 합니다.")
        if not (values["low_price"] <= min(values["open_price"], values["close_price"])
                <= max(values["open_price"], values["close_price"]) <= values["high_price"]):
            raise ValueError(f"{TICKER} {date.date()}: 고가·저가 관계가 잘못됐습니다.")
        if any(not 0 <= values[field] <= 1 for field in ("cheap_score", "cheap_score_confidence")):
            raise ValueError(f"{TICKER} {date.date()}: 가격 매력도·신뢰도 범위는 0~1입니다.")
        for field, value in values.items():
            frame.at[date, field] = value


class InMemoryDataHandler:
    """예제의 사전 구성 가격·등급만 조회한다. SQL/지표/등급 산정은 하지 않는다."""

    def __init__(self, prices, tiers):
        self.prices = _date_frame(prices, "가격")
        self.tiers = _date_frame(tiers, "종목 등급")
        _validate_prices(self.prices)
        if "tier" not in self.tiers:
            raise ValueError(f"{TICKER} 종목 등급: tier 필드 누락")
        for date, row in self.tiers.iterrows():
            if _number(row["tier"], date.date(), "tier") not in (1, 2, 3):
                raise ValueError(f"{TICKER} {date.date()}: tier는 1/2/3이어야 합니다.")
        self.lookups = []

    def _check_ticker(self, ticker):
        if ticker != TICKER:
            raise ValueError(f"가상 종목 {TICKER}만 지원합니다: {ticker}")

    def _as_of(self, frame, date):
        date = pd.Timestamp(date)
        eligible = frame.loc[frame.index <= date]
        if eligible.empty:
            return None
        row = eligible.iloc[-1].copy()
        self.lookups.append((date, row.name))
        return row

    def get_trading_dates(self, start, end):
        return self.prices.loc[pd.Timestamp(start):pd.Timestamp(end)].index

    @staticmethod
    def get_previous_trading_date(trading_dates, current_day_idx):
        return None if current_day_idx <= 0 else pd.Timestamp(trading_dates[current_day_idx - 1])

    def get_candidates_with_tier_fallback_pit_gated(
        self, signal_date, *, min_liquidity_20d_avg_value=0, min_tier12_coverage_ratio=0.0,
    ):
        if min_liquidity_20d_avg_value != 0 or min_tier12_coverage_ratio != 0:
            raise ValueError("이 예제는 유동성·종목 등급 충족 비율을 계산하지 않습니다.")
        tier = self.get_stock_tier_as_of(TICKER, signal_date)
        if tier is not None and tier["tier"] == 1:
            return [TICKER], "TIER_1_SYNTHETIC_ASOF"
        return [], "NO_CANDIDATES_SYNTHETIC_ASOF"

    def get_stock_tier_as_of(self, ticker, signal_date):
        self._check_ticker(ticker)
        row = self._as_of(self.tiers, signal_date)
        return None if row is None else {"tier": int(row["tier"])}

    def get_stock_row_as_of(self, ticker, date, start, end):
        self._check_ticker(ticker)
        return self._as_of(self.prices.loc[pd.Timestamp(start):pd.Timestamp(end)], date)

    def get_ohlc_data_on_date(self, date, ticker, start, end):
        self._check_ticker(ticker)
        date = pd.Timestamp(date)
        frame = self.prices.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if date not in frame.index:
            raise ValueError(f"{TICKER} {date.date()}: 당일 가격이 없습니다.")
        return frame.loc[date].copy()

    def get_latest_price(self, date, ticker, start, end):
        row = self.get_stock_row_as_of(ticker, date, start, end)
        if row is None:
            raise ValueError(f"{ticker} {date}: 평가 가격이 없습니다.")
        return row["close_price"]

    def get_name_from_ticker(self, ticker):
        self._check_ticker(ticker)
        return "가상 분할투자 종목"


def _integer(value):
    if not np.isfinite(value) or float(value) != int(value):
        raise RuntimeError(f"정수 원화·수량이 아닙니다: {value}")
    return int(value)


def _trade_rows(portfolio):
    rows = []
    for trade in portfolio.trade_history:
        date = trade.date.strftime("%Y-%m-%d")
        previous = TRADING_DATES[TRADING_DATES.index(date) - 1]
        price = trade.buy_price if trade.trade_type == "BUY" else trade.sell_price
        values = (date, previous, trade.code, trade.trade_type, _integer(trade.order),
                  _integer(trade.quantity), _integer(price), _integer(trade.commission),
                  _integer(trade.tax), _integer(trade.cash_after))
        rows.append(dict(zip(TRADE_FIELDS, values)))
    return rows


def _daily_rows(portfolio):
    positions = {item["date"]: item["positions"]
                 for item in portfolio.daily_positions_snapshot_history}
    if len(positions) != len(TRADING_DATES):
        raise RuntimeError("일별 보유 기록이 누락됐습니다.")
    rows = []
    for snapshot in portfolio.daily_snapshot_history:
        held = positions[snapshot["date"]]
        quantity = _integer(sum(row["Quantity"] for row in held))
        market_value = _integer(sum(row["Total Value"] for row in held))
        cash, total = _integer(snapshot["cash"]), _integer(snapshot["total_value"])
        if cash + market_value != total:
            raise RuntimeError("현금·보유 평가액과 총자산이 다릅니다.")
        values = (snapshot["date"].strftime("%Y-%m-%d"), cash, quantity, market_value, total)
        rows.append(dict(zip(DAILY_FIELDS, values)))
    return rows


def run_simulation(prices=None, tiers=None):
    """새 객체로 실제 CPU 경로를 실행한다. 변형 자료의 회귀 검사에도 사용한다."""
    handler = InMemoryDataHandler(make_price_rows() if prices is None else prices,
                                  make_tier_rows() if tiers is None else tiers)
    start, end = TRADING_DATES[0], TRADING_DATES[-1]
    portfolio = Portfolio(INITIAL_CASH, start, end, capture_daily_positions_snapshot=True)
    engine = BacktestEngine(start, end, portfolio, MagicSplitStrategy(**STRATEGY_SETTINGS),
                            handler, BasicExecutionHandler(**EXECUTION_SETTINGS))
    output, errors = StringIO(), StringIO()
    try:
        with redirect_stdout(output), redirect_stderr(errors):
            engine.run()
    except BaseException:
        print(output.getvalue() + errors.getvalue(), file=sys.stderr, end="")
        raise
    trades, daily = _trade_rows(portfolio), _daily_rows(portfolio)
    summary = dict(initial_cash=INITIAL_CASH, final_cash=_integer(portfolio.cash),
                   final_value=daily[-1]["total_value"],
                   total_commission=sum(row["commission"] for row in trades),
                   total_tax=sum(row["tax"] for row in trades),
                   net_pnl=daily[-1]["total_value"] - INITIAL_CASH)
    return dict(trades=trades, daily_equity=daily, summary=summary)


def verify_result(result):
    expected = dict(trades=[dict(zip(TRADE_FIELDS, row)) for row in EXPECTED_TRADES],
                    daily_equity=[dict(zip(DAILY_FIELDS, row)) for row in EXPECTED_DAILY],
                    summary=EXPECTED_SUMMARY)
    if result != expected:
        raise RuntimeError("가상 시나리오의 거래·자산·비용이 고정 기대값과 다릅니다.")


def save_csv(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    created = []
    try:
        for name, fields, rows in (("trades.csv", TRADE_FIELDS, result["trades"]),
                                    ("daily_equity.csv", DAILY_FIELDS, result["daily_equity"])):
            path = directory / name
            with path.open("x", encoding="utf-8-sig", newline="") as stream:
                created.append(path)
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass  # 다른 작업이 놓은 파일은 삭제하지 않는다.
        raise


def print_result(result):
    print(f"가상 종목 {TICKER} / 명시적 7거래일 / 실제 CPU 분할투자 구현")
    print("초기 현금 1,000,000원 · 주문 비율 10% · 최대 2회 분할 · 종목 1개")
    print("추가 매수 -5% · 개별 익절 +10% · 손절 -15% · 재진입 대기 5거래일 · 비활성 90거래일")
    print("후보: 가상 등급 1 · 신규 진입 등급 1 / 추가 매수 등급 1~2 · 낮은 분할 차수 우선 · 조회 오류는 실패")
    print("계산 모델: 매수·매도 수수료 각 0.015%, 매도 세금 0.18% (현행 요율 주장 아님)")
    print("\n가상 입력 가격 (원)")
    prices = pd.DataFrame([dict(date=d, **dict(zip(PRICE_FIELDS, row)))
                           for d, row in zip(TRADING_DATES, OHLC)])
    print(prices.rename(columns=dict(date="날짜", open_price="시가", high_price="고가",
                                    low_price="저가", close_price="종가")).to_string(index=False))
    print("\n거래 내역 (전일 조건 → 당일 시가, 원)")
    trades = pd.DataFrame(result["trades"]).replace({"side": {"BUY": "매수", "SELL": "매도"}})
    print(trades.rename(columns=dict(trade_date="거래일", signal_date="기준일", ticker="종목",
                                    side="매매", split="차수", quantity="수량", price="체결가",
                                    commission="수수료", tax="세금", cash_after="현금")).to_string(index=False))
    print("\n일별 현금·보유·총자산 (원)")
    print(pd.DataFrame(result["daily_equity"]).rename(columns=dict(
        date="날짜", cash="현금", quantity="수량", market_value="보유평가액", total_value="총자산",
    )).to_string(index=False))
    summary = result["summary"]
    print(f"\n확인 완료: 거래 {len(result['trades'])}건 / 최종 현금·자산 {summary['final_value']:,}원")
    print(f"수수료 {summary['total_commission']:,}원 / 세금 {summary['total_tax']:,}원 / "
          f"비용 차감 순손익 {summary['net_pnl']:,}원")
    print("가상 입력의 계산 재현입니다. SQL·종목/지표 산정·GPU·실제 투자 성과를 검증하지 않습니다.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="CSV 두 개를 저장할 새 디렉터리 (기존 경로 거부)")
    args = parser.parse_args(argv)
    result = run_simulation()
    verify_result(result)
    if args.output_dir is not None:
        save_csv(result, args.output_dir)
    print_result(result)
    if args.output_dir is not None:
        print(f"CSV 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
