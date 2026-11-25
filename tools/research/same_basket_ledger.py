"""같은 종목 대조(same-basket-r1) H 장부(같은 종목 보유)와 K 계좌(KODEX 200 보유).

usage: python -B tools/research/same_basket_ledger.py START END OUT_PREFIX --basket FILE --div approx|none|upper --delist freeze|zero [--etf CSV]
3월 실행 체크아웃을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행한다. DB는 읽기만 하고 주가는 END(≤ 2025-12-30)까지만 읽는다.
설계 4·5절(v1.1):
- H: 시작일 시가(엔진과 같은 호가 올림)에 종목당 50만 원 한도로 수수료 포함 최대 정수 주. 그날 못 산 종목의 자리는 끝까지 현금.
  교체·재조정·매도 없음. 가격은 엔진 DataHandler의 수정가격(X와 같은 조회 경로), 평가는 그날 이하 마지막 종가.
  수정가격에 반영된 주식분할로 수량을 바꾸지 않는다. 배당은 X(split_window_runs.py)와 같은 근사: 연말 마지막 거래일 전날 보유
  수정 수량 × 그날 수정계수 = 원주 수량 × DPS, 비정상 배당률(30% 초과) 제외. approx는 이듬해 5/16 이후 첫 거래일 지급, upper는
  권리 확정 다음 거래일 지급. 지급된 배당은 지급일 시가에 그 종목이 거래 가능하면 그날, 아니면 이후 첫 거래 가능일 시가에
  그 종목 배당금만으로 재투자하고 남은 돈은 현금으로 둔다(다시 쓰지 않음). 미수금은 지급 전까지 평가액에만 더한다.
- 상폐: freeze는 마지막 가격 동결, zero는 폐지일부터 0원.
- K: continuous_account_verdict.etf_curve와 같은 식(수정 시가, 수수료 포함 최대 정수 주, 호가 올림 없음). ETF 파일은 상한 이하 줄만 읽는다.
산출: OUT_PREFIX_H_curve.csv(일별 평가액·현금·미수금·주식 평가액), _H_pos.csv(일별 보유 수량), _H_buys.csv, _H_divs.csv, _H_meta.json, _K_curve.csv
"""
import argparse, json, math, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import same_basket_common as sb

MAX_YIELD = 0.3                      # split_window_runs.py와 같은 비정상 배당률 문턱
CASH_EPS = 1e-6


# ---------------- 순수 계산 ----------------
def tick_size(price):
    for upper, tick in ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)):
        if price < upper: return tick
    return 1000


def tick_up(price):
    """엔진 BasicExecutionHandler._adjust_price_up과 같은 계산(float32 시가를 호가 단위로 올림)."""
    p = np.float32(price); t = tick_size(p)
    return int(math.ceil(round(p / t, 5)) * t)


COMM_NUM, COMM_DEN = 3, 20000          # 0.015% = 3/20000. 수수료는 정수 산식으로 절사한다(실수 곱셈은 1원 작게 나올 수 있다)
assert COMM_NUM / COMM_DEN == sb.COMMISSION


def commission(gross):
    return int(gross) * COMM_NUM // COMM_DEN


def buy_cost(n, price):
    gross = int(n) * int(price)
    return gross + commission(gross)


def max_shares(budget, price):
    """수수료를 포함한 금액이 budget 이하인 최대 정수 주."""
    n = int(budget * COMM_DEN // (price * (COMM_DEN + COMM_NUM)))
    while n > 0 and buy_cost(n, price) > budget: n -= 1
    while buy_cost(n + 1, price) <= budget: n += 1
    return n


def tradable(open_px):
    return open_px is not None and np.isfinite(open_px) and open_px > 0


def delisted_on(state, day, j):
    di = state['delist_i'][j]
    return di is not None and day >= di


def can_trade(state, day, j):
    """폐지일부터는 가격 행이 있어도 사지 않는다."""
    return not delisted_on(state, day, j) and tradable(state['open'][day, j])


def buy(state, day, j, budget, kind):
    """예산 안에서 최대 정수 주를 산다. 한 주도 못 사면 수량 0으로 기록만 남고 예산은 현금에 남는다."""
    price = tick_up(state['open'][day, j]); n = max_shares(budget, price)
    cost = buy_cost(n, price)
    state['cash'] -= cost; state['qty'][j] += n
    state['buys'].append({'date': state['dates'][day], 'ticker': state['tickers'][j], 'kind': kind, 'open': float(state['open'][day, j]), 'price': price,
                          'quantity': n, 'gross': n * price, 'commission': cost - n * price, 'cost': cost, 'budget': float(budget)})
    return n


def initial_buys(state, day):
    for j, t in enumerate(state['tickers']):
        if not can_trade(state, day, j) or buy(state, day, j, sb.SLOT_CASH, 'initial') == 0: state['skipped'].append(t)


def dividend_amounts(state, ev):
    """권리 수량(전일 종가 보유 수정 수량 × 수정계수) × DPS. split_window_runs.py의 계산·제외 규칙과 같다.
    보유 종목마다 권리일의 입력 상태(classify_dividend)를 본다: ok(계산), no_dividend·delisted(배당 없음, 사유 기록),
    ratio_missing(배당이 있는데 원주 환산 불가 → 중단)."""
    rows = []
    for j, t in enumerate(state['tickers']):
        q = int(state['qty'][j])
        if q <= 0: continue
        info = ev['per_ticker'].get(t)
        if info is None: raise ValueError(f'dividend input not classified for held {t} at {ev["record_date"]}')
        status = info['status']          # 권리일 분류를 따른다(폐지일 = 권리일이면 ok. 재투자는 can_trade가 폐지일 다음 거래일부터 막는다)
        if status == 'ratio_missing': raise ValueError(f'dividend for held {t} at {ev["record_date"]} cannot be converted to raw shares (adj_ratio missing)')
        if status == 'ok' and not (np.isfinite(info['ratio']) and np.isfinite(info['close'])): raise ValueError(f'adj_ratio missing for {t} at {ev["record_date"]}')
        if status == 'ok' and info['close'] > 0 and info['dps'] / info['close'] > MAX_YIELD: status = 'yield_over_30pct'
        if status != 'ok':
            state['div_skips'].append({'fy': ev['fy'], 'record_date': ev['record_date'], 'ticker': t, 'reason': status}); continue
        rows.append({'fy': ev['fy'], 'record_date': ev['record_date'], 'ticker': t, 'adj_qty': q, 'raw_qty': q * info['ratio'], 'dps': info['dps'],
                     'fallback': bool(info['fallback']), 'amount': q * info['ratio'] * info['dps']})
    return rows


def recognize(state, day, ev):
    rows = dividend_amounts(state, ev)
    state['divs'] += rows
    state['receivable'][ev['fy']] = {r['ticker']: r['amount'] for r in rows}
    state['events'].append({'fy': ev['fy'], 'event': 'receivable', 'date': state['dates'][day], 'amount': sum(r['amount'] for r in rows)})


def pay(state, day, fy):
    """미수금을 현금으로 넣고 종목별 재투자 대기 예산에 올린다."""
    amounts = state['receivable'].pop(fy)
    state['cash'] += sum(amounts.values())
    for t, amt in amounts.items():
        if amt <= 0: continue
        j = state['tickers'].index(t); state['pending'][j] = state['pending'].get(j, 0.0) + amt
    state['events'].append({'fy': fy, 'event': 'paid', 'date': state['dates'][day], 'amount': sum(amounts.values())})


def reinvest(state, day):
    """대기 예산이 있는 종목 가운데 오늘 시가에 거래 가능한 것만 그 예산으로 산다. 남은 돈은 현금에 남고 다시 쓰지 않는다."""
    for j in sorted(state['pending']):
        if can_trade(state, day, j): buy(state, day, j, state['pending'].pop(j), 'reinvest')


def price_today(state, day, j, delist_zero):
    """폐지일(DB의 마지막 상장일) 다음 거래일부터: 0원 가정은 0, 동결 가정은 폐지일 종가(그 뒤 가격 행은 쓰지 않는다)."""
    if not delisted_on(state, day, j): return state['close'][day, j]
    di = state['delist_i'][j]
    return 0.0 if delist_zero else (state['close'][di - 1, j] if di >= 1 else np.nan)


def mark(state, day, delist_zero):
    held = [j for j in range(len(state['tickers'])) if state['qty'][j] > 0]
    prices = {j: price_today(state, day, j, delist_zero) for j in held}
    if any(not np.isfinite(p) for p in prices.values()): raise ValueError(f'no valuation price on {state["dates"][day]}')
    stock = float(sum(state['qty'][j] * prices[j] for j in held))
    recv = float(sum(sum(v.values()) for v in state['receivable'].values()))
    if state['cash'] < -CASH_EPS: raise ValueError(f'negative cash on {state["dates"][day]}')
    state['curve'].append({'date': state['dates'][day], 'value': state['cash'] + stock + recv, 'cash': state['cash'], 'receivable': recv,
                           'stock_value': stock, 'n_held': len(held)})
    state['pos'] += [{'date': state['dates'][day], 'ticker': state['tickers'][j], 'quantity': int(state['qty'][j]), 'price': float(prices[j])} for j in held]


def run_hold(inp, div, delist):
    """inp: {'dates','tickers','open','close','start_i','events','delist_i'}. events: [{'fy','record_date','rights_i','pay_i','per_ticker'}].
    rights_i는 권리 확정 다음 거래일(그날 주문 전 보유 = 전일 종가 보유), pay_i는 approx 지급일(자료 밖이면 None)."""
    n = len(inp['tickers'])
    state = {**inp, 'cash': sb.CASH0, 'qty': [0] * n, 'receivable': {}, 'pending': {}, 'buys': [], 'skipped': [], 'divs': [], 'div_skips': [], 'events': [], 'curve': [], 'pos': []}
    for day in range(len(inp['dates'])):
        if day >= inp['start_i'] and div != 'none':
            for ev in inp['events']:
                if ev['rights_i'] == day:
                    recognize(state, day, ev)
                    if div == 'upper': pay(state, day, ev['fy'])
                if div == 'approx' and ev['pay_i'] == day and ev['fy'] in state['receivable']: pay(state, day, ev['fy'])
        if day == inp['start_i']: initial_buys(state, day)
        if day >= inp['start_i']: reinvest(state, day)
        mark(state, day, delist == 'zero')
    return state


def kodex_curve(etf, days, cash0=sb.CASH0, comm=sb.COMMISSION):
    """continuous_account_verdict.etf_curve와 같은 식. etf: 날짜 색인의 open·close, days: 시작일부터의 거래일."""
    e = etf.reindex(days)
    px = e[['open', 'close']].to_numpy(dtype=float)
    if etf.index.has_duplicates or not np.isfinite(px).all() or (px <= 0).any(): raise ValueError('ETF price missing, duplicated or non-positive in the window')
    o = float(e.open.iloc[0]); n = np.floor(cash0 / (o * (1 + comm)))
    return pd.Series(cash0 - (n * o + np.floor(n * o * comm)) + n * e.close.values, index=days), int(n)


# ---------------- 자료 읽기 ----------------
def read_etf_capped(path, cap=sb.END_CAP):
    """날짜 문자열만 보고 상한 이하 줄만 읽는다(상한 뒤 가격은 해석하지 않음)."""
    with open(path, newline='') as f:
        header = f.readline(); di = header.rstrip('\r\n').split(',').index('date')
        lines = [header] + [l for l in f if l.split(',', di + 1)[di] <= cap]
    import io
    return pd.read_csv(io.StringIO(''.join(lines)), parse_dates=['date']).set_index('date')


def engine_data_handler(start, end):
    """엔진(run_backtest_from_config)과 같은 방식으로 DataHandler를 만든다. strict_pit·가격 정책이 X와 같아야 한다."""
    import yaml
    from src.analysis import walk_forward_analyzer as wfa
    from src import main_backtest as mb
    from src.data_handler import DataHandler
    base = yaml.safe_load(open('config/config.yaml'))
    params = dict(json.load(open('results/wfo_run_20260324_125326/final_candidate_manifest.json'))['champion_params'])
    cfg = wfa._build_cpu_backtest_config(base, start_date=start, end_date=end, initial_cash=sb.CASH0,
                                         strategy_params=wfa._extract_strategy_params(params, base['strategy_params']))
    spc = cfg['strategy_params']
    mode = mb.resolve_universe_mode(spc, universe_mode=os.environ.get('MAGICSPLIT_UNIVERSE_MODE'))
    if mode != 'strict_pit': raise SystemExit(f'universe mode must be strict_pit (got {mode})')
    basis, gate = mb.resolve_price_policy(spc)
    return DataHandler(db_config=cfg['database'], price_basis=basis, adjusted_price_gate_start_date=gate, universe_mode=mode, strategy_params=spc), basis


def price_matrices(dh, tickers, dates, cap=sb.END_CAP):
    """엔진과 같은 조회(load_stock_data)로 시가(그날 행)와 평가 종가(그날 이하 마지막 행)를 만든다."""
    start, end = str(dates[0].date()), str(dates[-1].date())
    if dates[-1] > pd.Timestamp(cap): raise SystemExit(f'END {end} exceeds price cap {cap}')
    opens, closes = [], []
    for t in tickers:
        df = dh.load_stock_data(t, start, end)
        if df is None or df.empty:
            opens.append(pd.Series(np.nan, index=dates)); closes.append(pd.Series(np.nan, index=dates)); continue
        if df.index.max() > dates[-1]: raise RuntimeError('price rows after END')
        opens.append(pd.to_numeric(df.open_price, errors='coerce').reindex(dates))
        closes.append(pd.to_numeric(df.close_price, errors='coerce').reindex(dates).ffill())
    return np.column_stack(opens).astype(float), np.column_stack(closes).astype(float)


def classify_dividend(t, dps, ratio, record_day, delisted):
    """한 종목의 권리일 배당 입력 상태. dps: {종목: (DPS, 대체 여부)}, ratio: {종목: (수정계수, 원주 종가)}."""
    d, fb = dps.get(t, (0.0, False)); r, c = ratio.get(t, (np.nan, np.nan))
    info = {'dps': float(d), 'fallback': bool(fb), 'ratio': float(r), 'close': float(c)}
    if t in delisted and pd.Timestamp(delisted[t]) < pd.Timestamp(record_day): return {**info, 'status': 'delisted'}   # 폐지일 = 마지막 상장일
    if not (np.isfinite(d) and d > 0): return {**info, 'status': 'no_dividend'}
    if not (np.isfinite(r) and r > 0 and np.isfinite(c)): return {**info, 'status': 'ratio_missing'}
    return {**info, 'status': 'ok'}


def dividend_events(conn, cal, tickers, dates, start_i, delisted=None):
    """권리 확정 다음 거래일이 (시작일, 종료일] 안에 있는 FY의 사건. DPS·수정계수는 split_window_runs.py와 같은 함수로 읽는다.
    명단의 모든 종목에 입력 상태(classify_dividend)를 남긴다. delisted: {종목: 폐지일}(폐지일이 권리일보다 앞이면 배당 없음)."""
    import stage2_common as sc
    delisted = delisted or {}
    pos = {d: i for i, d in enumerate(dates)}; out = []
    for fy in range(dates[0].year - 1, dates[-1].year + 1):
        rec = cal.record_date(fy) if (cal.dates.year == fy).sum() >= 2 else None
        if rec is None or pos.get(cal.shift(rec, 1), -1) <= start_i: continue
        dps = sc.dps_table(conn, tickers, [fy - 1, fy]); dps = dps[dps.fy == fy]
        dps = {r.ticker: (float(r.dps), bool(r.dps_fallback)) for r in dps.itertuples()}
        ratio = sc.adj_ratio_on(conn, tickers, rec)
        per = {t: classify_dividend(t, dps, ratio, rec, delisted) for t in tickers}
        out.append({'fy': fy, 'record_date': str(rec.date()), 'rights_i': pos[cal.shift(rec, 1)], 'pay_i': pos.get(cal.pay_date(fy, min_day=16)), 'per_ticker': per})
    return out


def delist_positions(conn, tickers, dates):
    import stage2_common as sc
    ph = ','.join(['%s'] * len(tickers))
    d = sc.q(conn, f"SELECT stock_code, delisted_date FROM TickerUniverseHistory WHERE stock_code IN ({ph}) AND delisted_date IS NOT NULL AND delisted_date <= %s",
             list(tickers) + [str(dates[-1].date())])
    day = {str(r.stock_code): pd.Timestamp(r.delisted_date) for r in d.itertuples()}
    # 위치 = 폐지일 다음 거래일(폐지일은 DB의 마지막 스냅숏·마지막 가격 행과 같은 마지막 상장일로 본다. 조정자 결정 B)
    return [int(dates.searchsorted(day[t], side='right')) if t in day else None for t in tickers], {t: str(v.date()) for t, v in day.items()}


def load_inputs(basket_file, start, end):
    import stage2_common as sc
    tickers = list(sb.read_basket(basket_file).ticker)
    conn = sc.connect(); cal = sc.Calendar(sc.trading_dates(conn))
    s = pd.Timestamp(start); base = sb.prev_day(cal.dates, s)
    dates = cal.dates[(cal.dates >= base) & (cal.dates <= pd.Timestamp(end))]
    dh, basis = engine_data_handler(str(base.date()), str(dates[-1].date()))
    opens, closes = price_matrices(dh, tickers, dates)
    delist_i, delisted = delist_positions(conn, tickers, dates)
    inp = {'dates': [str(d.date()) for d in dates], 'tickers': tickers, 'open': opens, 'close': closes, 'start_i': 1,
           'events': dividend_events(conn, cal, tickers, dates, 1, delisted), 'delist_i': delist_i}
    return inp, dates, {'price_basis': basis, 'base_date': str(base.date()), 'delisted': delisted}


# ---------------- 쓰기 ----------------
def write_hold(prefix, state, meta):
    cols = {'curve': ['date', 'value', 'cash', 'receivable', 'stock_value', 'n_held'], 'pos': ['date', 'ticker', 'quantity', 'price'],
            'buys': ['date', 'ticker', 'kind', 'open', 'price', 'quantity', 'gross', 'commission', 'cost', 'budget'],
            'divs': ['fy', 'record_date', 'ticker', 'adj_qty', 'raw_qty', 'dps', 'fallback', 'amount', 'event', 'date']}
    pd.DataFrame(state['curve'], columns=cols['curve']).to_csv(prefix + '_H_curve.csv', index=False)
    pd.DataFrame(state['pos'], columns=cols['pos']).to_csv(prefix + '_H_pos.csv', index=False)
    pd.DataFrame(state['buys'], columns=cols['buys']).to_csv(prefix + '_H_buys.csv', index=False)
    pd.DataFrame(state['divs'] + state['events'], columns=cols['divs']).to_csv(prefix + '_H_divs.csv', index=False)
    pd.DataFrame(state['div_skips'], columns=['fy', 'record_date', 'ticker', 'reason']).to_csv(prefix + '_H_divskips.csv', index=False)
    json.dump({**meta, 'skipped_at_start': state['skipped'], 'unreinvested_budget': {state['tickers'][j]: v for j, v in state['pending'].items()},
               'n_initial_buys': sum(b['kind'] == 'initial' for b in state['buys']), 'n_reinvest_buys': sum(b['kind'] == 'reinvest' for b in state['buys'])},
              open(prefix + '_H_meta.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('prefix'); ap.add_argument('--basket', required=True)
    ap.add_argument('--div', choices=sb.DIVS, required=True); ap.add_argument('--delist', choices=sb.DELISTS, required=True)
    ap.add_argument('--etf', default=sb.ETF_CSV)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    inp, dates, info = load_inputs(a.basket, a.start, a.end)
    state = run_hold(inp, a.div, a.delist)
    write_hold(a.prefix, state, {**info, 'account_start': inp['dates'][1], 'end': inp['dates'][-1], 'div': a.div, 'delist': a.delist,
                                 'basket_file': os.path.basename(a.basket), 'basket_sha256': sb.sha256(a.basket)})
    k, shares = kodex_curve(read_etf_capped(a.etf), dates[1:])
    out = pd.concat([pd.Series([sb.CASH0], index=dates[:1]), k]).rename('value')
    out.to_csv(a.prefix + '_K_curve.csv', index_label='date')
    json.dump({'etf_file': os.path.basename(a.etf), 'etf_sha256': sb.sha256(a.etf), 'shares': shares, 'account_start': inp['dates'][1], 'end': inp['dates'][-1]},
              open(a.prefix + '_K_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
