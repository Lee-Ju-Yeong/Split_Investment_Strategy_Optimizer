"""같은 종목 대조(same-basket-r1) X 실행기: continuous_account_run.py → split_window_runs.py 경로를 그대로 쓰면서
신규 진입 후보 조회만 감싸 고정 명단과 교차한다. 제품 코드와 기존 도구는 바꾸지 않는다.

usage: python -B tools/research/same_basket_run.py START END OUT_PREFIX --basket FILE [--no-restrict] [--warmup]
       [--delist freeze|zero] [--only TICKER] [--cash WON] [--ratio R] [--eligible-all] [split_window_runs.py 옵션: --tax --div --hold ...]
3월 실행 체크아웃을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행한다. 앞의 세 인자는 위치 인자다.
- START는 등록한 시작일(첫 주문 가능일)이다. --warmup이면 엔진을 직전 거래일부터 돌린다. 엔진은 첫날에 전일 신호가 없어
  주문을 내지 않으므로 직전 거래일은 신호 준비만 하고, START 시가부터 주문할 수 있다(설계 4절). 시작일은 옮기지 않는다.
- --basket: 명단 파일. 후보 조회(get_candidates_with_tier_fallback_pit_gated) 결과를 명단과 먼저 교차한 뒤 엔진의 순위식이 돈다.
- --no-restrict: 교차하지 않는다(재현 대조용). --only: 명단 안의 한 종목만 허용(보조 시험).
- --delist zero: 상장폐지일(DB의 마지막 상장일) 다음 거래일부터 그 종목 평가 가격을 0원으로 둔다(월초 주문 예산에도 반영). freeze는 엔진 그대로.
- --cash / --ratio: 시작 현금과 주문율을 바꾼다(보조 시험: 50만 원, 44% 또는 10%).
기존 산출물(_curve·_fills·_divs·_state·_endpos·_meta 등)에 더해 아래 두 파일을 쓴다.
- OUT_PREFIX_eligible.csv: 날짜별 전일 신호일, 후보 출처, 그날 후보 수, 진입 가능 종목 수(허용 명단 ∩ 그날 후보), 보유 종목 수, 진입 신호 수.
  엔진은 빈자리가 있는 날만 후보를 조회한다. 조회하지 않은 날은 NOT_EVALUATED로 남기고, --eligible-all이면 그런 날도 같은 조회를
  한 번 더 해서 채운다(조회는 읽기만 하며 체결·현금을 바꾸지 않는다: 재현 대조가 이 옵션을 켠 채로 바이트 일치를 확인한다).
- OUT_PREFIX_sbmeta.json: 시작일·엔진 시작일·명단 해시·옵션·첫 체결일·0원 처리 종목·후보에 한 번이라도 든 명단 종목·
  시작일 주문액으로 1주도 못 사는 명단 종목(성과 값은 넣지 않는다)
- OUT_PREFIX_divinputs.csv: 배당 권리일마다 보유 종목의 배당 입력 상태(ok / no_dividend / delisted). 배당이 있는 보유분을
  원주로 환산할 수 없으면(수정계수 없음) 그 자리에서 중단한다(종료 코드 비0). 권리일 이전에 폐지된 종목은 배당을 받지 않는다.
하위 실행기(split_window_runs.py)의 표준 출력(최종 평가액·배당 합계)은 버린다.
전일 신호를 준비하지 못했으면 종료 코드 3, 명단 밖 매수가 있으면 4로 끝난다.
"""
import argparse, contextlib, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from src.data_handler import DataHandler
from src.backtest.cpu.portfolio import Portfolio
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.analysis import walk_forward_analyzer as wfa
import stage2_common as sc
import continuous_account_run as car
import same_basket_common as sb

EXIT_NO_SIGNAL, EXIT_OUTSIDE_BASKET = 3, 4
ELIGIBLE_COLS = ['date', 'signal_date', 'tier_source', 'n_candidates', 'n_eligible', 'n_held', 'n_entry_signals']


def parser():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('prefix')
    ap.add_argument('--basket', required=True); ap.add_argument('--no-restrict', action='store_true')
    ap.add_argument('--warmup', action='store_true'); ap.add_argument('--delist', choices=sb.DELISTS, default='freeze')
    ap.add_argument('--only'); ap.add_argument('--cash', type=float); ap.add_argument('--ratio', type=float)
    ap.add_argument('--eligible-all', action='store_true')
    return ap


def allowed_tickers(a):
    tickers = list(sb.read_basket(a.basket).ticker)
    if a.only is None: return set(tickers)
    if a.only not in tickers: raise SystemExit(f'--only {a.only} is not in the basket')
    return {a.only}


def account_dates(a):
    """(등록 시작일, 엔진 시작일). 시작일은 거래일이어야 하고 종료일은 조회 상한 이하여야 한다."""
    dates = sc.trading_dates(sc.connect()); s = pd.Timestamp(a.start)
    if pd.Timestamp(a.end) > pd.Timestamp(sb.END_CAP): raise SystemExit(f'END {a.end} exceeds price cap {sb.END_CAP}')
    if not a.warmup: return s, s
    if s not in dates: raise SystemExit(f'START {a.start} is not a trading day')
    return s, sb.prev_day(dates, s)


def delisted_dates(end):
    d = sc.q(sc.connect(), "SELECT stock_code, delisted_date FROM TickerUniverseHistory WHERE delisted_date IS NOT NULL AND delisted_date <= %s", [str(end)])
    return {str(r.stock_code): pd.Timestamp(r.delisted_date) for r in d.itertuples()}


def day_key(date):
    return str(pd.Timestamp(date).date())


def candidate_info(codes, source, allowed):
    """그날 후보 ∩ 허용 명단. 2등급 대체는 엔진(strict_hysteresis)처럼 0으로 본다."""
    blocked = str(source).startswith('TIER_2_FALLBACK')
    return {'tier_source': str(source), 'n_candidates': len(codes), 'n_eligible': 0 if blocked else len({str(c) for c in codes} & allowed)}


def install_basket_filter(allowed, restrict, seen, ever=None):
    """후보 조회를 감싼다. 엔진이 조회한 신호일의 결과를 seen에 남기고(ever: 1등급 후보에 든 명단 종목), 원래 함수를 돌려준다."""
    lookup0 = DataHandler.get_candidates_with_tier_fallback_pit_gated

    def lookup(self, date, min_liquidity_20d_avg_value=None, min_tier12_coverage_ratio=None):
        codes, source = lookup0(self, date, min_liquidity_20d_avg_value=min_liquidity_20d_avg_value, min_tier12_coverage_ratio=min_tier12_coverage_ratio)
        seen[day_key(date)] = candidate_info(codes, source, allowed)
        if ever is not None and not str(source).startswith('TIER_2_FALLBACK'): ever.update(str(c) for c in codes if str(c) in allowed)
        return ([c for c in codes if str(c) in allowed] if restrict else codes), source
    DataHandler.get_candidates_with_tier_fallback_pit_gated = lookup
    return lookup0


def extra_lookup(strategy, data_handler, signal_date, lookup0, allowed):
    """엔진이 조회하지 않은 날(빈자리 없음)의 후보를 같은 함수로 읽는다."""
    try:
        codes, source = lookup0(data_handler, signal_date, min_liquidity_20d_avg_value=strategy.min_liquidity_20d_avg_value,
                                min_tier12_coverage_ratio=strategy.min_tier12_coverage_ratio)
    except Exception as exc:
        return {'tier_source': f'LOOKUP_ERROR:{type(exc).__name__}', 'n_candidates': '', 'n_eligible': ''}
    return candidate_info(codes, source, allowed)


def install_entry_log(lookup0, allowed, seen, rows, log_all):
    entry0 = MagicSplitStrategy.generate_new_entry_signals
    blank = {'n_candidates': '', 'n_eligible': ''}

    def entry(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        sd = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
        row = {'date': day_key(current_date), 'signal_date': '' if sd is None else day_key(sd), 'n_held': len(portfolio.positions)}
        signals = entry0(self, current_date, portfolio, data_handler, trading_dates, current_day_idx)
        if sd is None: info = {'tier_source': 'NO_SIGNAL_DATE', **blank}
        elif day_key(sd) in seen: info = seen[day_key(sd)]
        elif log_all: info = extra_lookup(self, data_handler, sd, lookup0, allowed)
        else: info = {'tier_source': 'NOT_EVALUATED', **blank}
        rows.append({**row, **info, 'n_entry_signals': len(signals)})
        return signals
    MagicSplitStrategy.generate_new_entry_signals = entry


def install_delist_zero(delisted, zeroed):
    """상장폐지일 다음 거래일부터 평가 가격 0원(폐지일 = 마지막 상장일). 엔진의 평가 가격표(_resolve_price_map)를 감싸므로 평가액·월초 주문 예산·마지막 보유 기록
    (continuous_account_run.end_positions)이 모두 같은 0원을 쓴다."""
    resolve0 = Portfolio._resolve_price_map

    def resolve(self, evaluation_date, data_handler, price_map=None):
        prices = resolve0(self, evaluation_date, data_handler, price_map=price_map)
        day = pd.Timestamp(evaluation_date)
        dead = [t for t in self.positions if t in delisted and day > delisted[t]]      # 폐지일 다음 거래일부터
        if not dead: return prices
        if prices is None:
            prices = {t: data_handler.get_latest_price(evaluation_date, t, self.start_date, self.end_date) for t in self.positions}
        prices = dict(prices)
        for t in dead:
            prices[t] = 0.0; zeroed.setdefault(t, str(day.date()))
        return prices
    Portfolio._resolve_price_map = resolve


def keep_ratio(t, ratio, dps, day, delisted):
    """split_window_runs가 수량 × 수정계수 × DPS를 곱할 종목만 남긴다. 권리일 전에 폐지된 종목과, 무배당인데 곱이 0이 아닐 수 있는
    입력(수정계수나 DPS가 NaN, DPS 음수)은 빼서 기여액을 0으로 만든다. 무배당이고 DPS 0·계수 유한이면 기여 0인 행을 그대로 둔다(재현 대조 유지)."""
    import same_basket_ledger as ledger
    info = ledger.classify_dividend(t, dps, {t: ratio}, day, delisted)
    if info['status'] == 'delisted': return False
    if info['status'] == 'no_dividend': return info['dps'] == 0 and np.isfinite(info['ratio']) and np.isfinite(info['close'])
    return True


def install_dividend_guard(delisted, log):
    """split_window_runs.py가 쓰는 배당 입력(stage2_common.dps_table → adj_ratio_on)을 감싼다. 권리일 보유 종목마다 상태를 남기고,
    배당이 있는데 수정계수가 없으면 중단한다. 권리일 이전 폐지 종목은 수정계수 결과에서 빼서 배당을 받지 않게 한다(H와 같은 규칙)."""
    import same_basket_ledger as ledger
    dps0, ratio0 = sc.dps_table, sc.adj_ratio_on
    last = {}

    def dps_table(conn, tickers, years):
        df = dps0(conn, tickers, years); last['dps'] = df; last['fy'] = list(years)[-1]
        return df

    def adj_ratio_on(conn, tickers, day):
        res = ratio0(conn, tickers, day)
        d = last.get('dps'); d = d[d.fy == last['fy']] if d is not None else pd.DataFrame(columns=['ticker', 'dps', 'dps_fallback'])
        dps = {r.ticker: (float(r.dps), bool(r.dps_fallback)) for r in d.itertuples()}
        for t in tickers:
            info = ledger.classify_dividend(t, dps, res, day, delisted)
            log.append({'record_date': day_key(day), 'fy': last.get('fy'), 'ticker': t, 'status': info['status']})
            if info['status'] == 'ratio_missing': raise RuntimeError(f'dividend for held {t} at {day_key(day)} cannot be converted to raw shares (adj_ratio missing)')
        return {t: v for t, v in res.items() if keep_ratio(t, v, dps, day, delisted)}
    sc.dps_table, sc.adj_ratio_on = dps_table, adj_ratio_on


def install_start_affordability(start, allowed, out):
    """시작일 주문액(엔진이 정한 investment_per_order)으로 1주도 못 사는 명단 종목을 기록한다(읽기만 한다)."""
    entry0 = MagicSplitStrategy.generate_new_entry_signals

    def entry(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
        signals = entry0(self, current_date, portfolio, data_handler, trading_dates, current_day_idx)
        if pd.Timestamp(current_date) == start and 'unaffordable' not in out:
            out['unaffordable'], out['no_open'] = [], []
            for t in sorted(allowed):
                row = data_handler.get_ohlc_data_on_date(current_date, t, self.backtest_start_date, self.backtest_end_date)
                o = None if row is None else row.get('open_price')
                if o is None or not pd.notna(o) or float(o) <= 0: out['no_open'].append(t); continue
                if int(float(self.investment_per_order) // self._adjust_price_up(float(o))) <= 0: out['unaffordable'].append(t)
        return signals
    MagicSplitStrategy.generate_new_entry_signals = entry


def install_account_override(cash, ratio, used):
    run0 = wfa.run_cpu_single_backtest

    def run(*args, **kwargs):
        if cash is not None: kwargs['initial_cash'] = float(cash)
        if ratio is not None: kwargs['params_dict'] = {**kwargs['params_dict'], 'order_investment_ratio': float(ratio)}
        used.update({'initial_cash': float(kwargs['initial_cash']), 'order_investment_ratio': float(kwargs['params_dict']['order_investment_ratio']),
                     'engine_start': str(kwargs['start_date']), 'engine_end': str(kwargs['end_date'])})
        return run0(*args, **kwargs)
    wfa.run_cpu_single_backtest = run


def signal_ready(rows, start, engine_start):
    """등록 시작일 행의 전일 신호가 엔진 시작일(직전 거래일)이고, 엔진이 그 신호일의 후보를 실제로 조회했는지."""
    row = next((r for r in rows if r['date'] == str(start.date())), None)
    return bool(row and row['signal_date'] == str(engine_start.date()) and not row['tier_source'].startswith(('LOOKUP_ERROR', 'NO_SIGNAL_DATE', 'NOT_EVALUATED')))


def post_checks(a, prefix, start, engine_start, allowed, rows):
    fills = pd.read_csv(prefix + '_fills.csv', dtype={'ticker': str}) if os.path.getsize(prefix + '_fills.csv') > 2 else pd.DataFrame(columns=['date', 'ticker', 'type'])
    buys = fills[fills.type == 'BUY']
    outside = sorted(set(buys.ticker) - allowed) if not a.no_restrict else []
    early = int((pd.to_datetime(fills.date) < start).sum()) if len(fills) else 0
    ready = signal_ready(rows, start, engine_start) if a.warmup else None
    info = {'first_fill_date': str(fills.date.min()) if len(fills) else None, 'n_fills': int(len(fills)), 'fills_before_start': early,
            'buys_outside_allowed': outside, 'signal_ready': ready}
    code = EXIT_NO_SIGNAL if (a.warmup and (not ready or early)) else (EXIT_OUTSIDE_BASKET if outside else 0)
    return info, code


def main():
    a, rest = parser().parse_known_args()
    allowed = allowed_tickers(a); start, engine_start = account_dates(a)
    rows, zeroed, used, seen, ever, afford, divlog = [], {}, {}, {}, set(), {}, []
    delisted = delisted_dates(a.end)
    lookup0 = install_basket_filter(allowed, not a.no_restrict, seen, ever)
    install_entry_log(lookup0, allowed, seen, rows, a.eligible_all)
    install_start_affordability(start, allowed, afford)
    if a.delist == 'zero': install_delist_zero(delisted, zeroed)
    install_dividend_guard(delisted, divlog)
    install_account_override(a.cash, a.ratio, used)
    sys.argv = [car.__file__, str(engine_start.date()), a.end, a.prefix] + rest
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        car.main()
    pd.DataFrame(rows, columns=ELIGIBLE_COLS).to_csv(a.prefix + '_eligible.csv', index=False)
    pd.DataFrame(divlog, columns=['record_date', 'fy', 'ticker', 'status']).to_csv(a.prefix + '_divinputs.csv', index=False)
    info, code = post_checks(a, a.prefix, start, engine_start, allowed, rows)
    meta = {'account_start': str(start.date()), 'engine_start': str(engine_start.date()), 'end': a.end, 'warmup': a.warmup,
            'restrict': not a.no_restrict, 'eligible_all': a.eligible_all, 'basket_file': os.path.basename(a.basket), 'basket_sha256': sb.sha256(a.basket),
            'allowed': sorted(allowed), 'only': a.only, 'delist': a.delist, 'delisted_zeroed': zeroed, **used, **info, 'swr_args': rest,
            'basket_delisted': {t: str(d.date()) for t, d in sorted(delisted.items()) if t in allowed}, 'ever_candidate': sorted(ever),
            'start_unaffordable': afford.get('unaffordable'), 'start_no_open': afford.get('no_open')}
    json.dump(meta, open(a.prefix + '_sbmeta.json', 'w'), indent=1)
    sys.exit(code)


if __name__ == '__main__':
    main()
