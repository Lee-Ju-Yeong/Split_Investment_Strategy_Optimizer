# backtester.py (수정 필수!)

from tqdm import tqdm
import pandas as pd
import warnings
import logging
import os
from typing import Optional

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class BacktestEngine:
    def __init__(
        self,
        start_date,
        end_date,
        portfolio,
        strategy,
        data_handler,
        execution_handler,
        *,
        logger: Optional[logging.Logger] = None,
        debug_ticker: Optional[str] = None,
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.portfolio = portfolio
        self.strategy = strategy
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.logger = logger or logging.getLogger(__name__)
        self.debug_ticker = debug_ticker

    def run(self):
        self.logger.info("백테스팅 엔진을 시작합니다...")
        
        trading_dates = self.data_handler.get_trading_dates(self.start_date, self.end_date)

        debug_ticker = self.debug_ticker
        if not debug_ticker:
            env_debug_ticker = os.getenv("BACKTEST_DEBUG_TICKER", "").strip()
            debug_ticker = env_debug_ticker or None
        
        # tqdm의 mininterval을 늘려 로그 출력이 밀리지 않게 함
        for i, current_date in enumerate(tqdm(trading_dates, desc="Backtesting Progress", mininterval=1.0)):
            if debug_ticker and self.logger.isEnabledFor(logging.DEBUG):
                try:
                    stock_data = self.data_handler.load_stock_data(debug_ticker, self.start_date, self.end_date)
                    # 데이터프레임이 유효하고, 현재 날짜의 데이터가 존재하는지 확인
                    if stock_data is not None and not stock_data.empty and current_date in stock_data.index:
                        ohlc = stock_data.loc[current_date]
                        self.logger.debug(
                            "[CPU_DATA_DEBUG] %s | %s | Open=%s, High=%s, Low=%s, Close=%s",
                            current_date.strftime("%Y-%m-%d"),
                            debug_ticker,
                            ohlc.get("open_price"),
                            ohlc.get("high_price"),
                            ohlc.get("low_price"),
                            ohlc.get("close_price"),
                        )
                except Exception:
                    # 디버그 경로는 백테스트를 깨지 않되, 조사 가능하도록 예외를 남깁니다.
                    self.logger.debug("debug ticker fetch skipped", exc_info=True)
            # --- 1. [변경] 신호 생성 및 실행 로직 분리 ---
            
            # [추가] 오늘 거래가 실행되기 전의 거래 내역 개수를 기록
            num_trades_before = len(self.portfolio.trade_history)

            # 단계 1-1: 매도 신호 생성 및 즉시 실행
            sell_signals = self.strategy.generate_sell_signals(current_date, self.portfolio, self.data_handler, trading_dates, i)
            if sell_signals:
                for signal in sell_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler, i)

            # 단계 1-2: 매수 신호를 '신규 진입' -> '추가 매수' 순으로 분리하여 실행
            # (1) 신규 진입 신호 생성 및 실행
            new_entry_signals = self.strategy.generate_new_entry_signals(current_date, self.portfolio, self.data_handler, trading_dates, i)
            if new_entry_signals:
                for signal in new_entry_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler, i)
            # (2) 추가 매수 신호 생성 및 실행 (신규 진입으로 자금/슬롯이 소진된 후의 상태 기준)
            additional_buy_signals = self.strategy.generate_additional_buy_signals(current_date, self.portfolio, self.data_handler, trading_dates, i)
            if additional_buy_signals:
                for signal in additional_buy_signals:
                    self.execution_handler.execute_order(signal, self.portfolio, self.data_handler, i)

            # --- 2. 일별 포트폴리오 가치 및 상태 기록 ---
            total_value = self.portfolio.get_total_value(current_date, self.data_handler)
            self.portfolio.record_daily_snapshot(current_date, total_value)

            # --- 3. [신규 로직] 당일 발생한 모든 거래에 최종 포트폴리오 가치 업데이트 ---
            num_trades_after = len(self.portfolio.trade_history)
            if num_trades_after > num_trades_before:
                for i in range(num_trades_before, num_trades_after):
                    self.portfolio.trade_history[i].total_portfolio_value = total_value

            # --- 4. [핵심] 일일 포트폴리오 스냅샷 로그 출력 ---
            # 상세 스냅샷은 DEBUG에서만 생성/출력합니다(기본 INFO에서는 tqdm 진행바만 유지).
            if total_value > 0 and self.logger.isEnabledFor(logging.DEBUG):
                stock_value = total_value - self.portfolio.cash
                # total_value가 0이 되는 엣지 케이스 방지
                cash_ratio = (self.portfolio.cash / total_value) * 100 if total_value else 0
                stock_ratio = (stock_value / total_value) * 100 if total_value else 0

                header = f"\n{'='*120}\n"
                footer = f"\n{'='*120}"

                date_str = pd.to_datetime(current_date).strftime("%Y-%m-%d")
                summary_str = (
                    f"Date: {date_str} | Day {i+1}/{len(trading_dates)}\n"
                    f"{'-'*120}\n"
                    f"Total Value: {total_value:,.0f} | "
                    f"Cash: {self.portfolio.cash:,.0f} ({cash_ratio:.1f}%) | "
                    f"Stocks: {stock_value:,.0f} ({stock_ratio:.1f}%)\n"
                    f"Holdings Count: {len(self.portfolio.positions)} Stocks"
                )

                log_message = header + summary_str

                positions_df = self.portfolio.get_positions_snapshot(current_date, self.data_handler, total_value)
                if not positions_df.empty:
                    positions_df["Avg Buy Price"] = positions_df["Avg Buy Price"].map("{:,.0f}".format)
                    positions_df["Current Price"] = positions_df["Current Price"].map("{:,.0f}".format)
                    positions_df["Unrealized P/L"] = positions_df["Unrealized P/L"].map("{:,.0f}".format)
                    positions_df["Total Value"] = positions_df["Total Value"].map("{:,.0f}".format)
                    positions_df["P/L Rate"] = positions_df["P/L Rate"].map("{:.2%}".format)
                    positions_df["Weight"] = positions_df["Weight"].map("{:.2%}".format)

                    log_message += f"\n{'-'*120}\n[Current Holdings]\n"
                    log_message += positions_df.to_string()

                log_message += footer
                self.logger.debug(log_message)

        self.logger.info("백테스팅이 완료되었습니다.")
        return self.portfolio
