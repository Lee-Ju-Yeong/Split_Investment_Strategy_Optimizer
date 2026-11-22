"""2단계 S20 재현 대조(과거 규칙 전용, v2.1 B): selection_vs_etf_sim.py의 simulate()를 div='approx', 폐지 0원 없음, 띠 없음으로 옮겨
2014~2015 구간을 다시 계산하고 보관된 sim_v2/curves_2014.csv의 2014_S20(일별)과 sim_v2/accounts_2014.csv의 S20/div_approx final에 대조한다.

usage: python -B tools/research/pool_vs_index_legacy_s20.py PANEL_v2.csv STAGE2_DIR OUT.json
3월 실행 체크아웃을 작업 디렉터리로 실행(DB 읽기, 호가 함수). 기존 스크립트는 import하지 않는다(가져오기만 해도 DB를 실행).
가격·배당 조회는 END_CAP 이하로 강제한다(원 스크립트는 2026-01-31까지 읽었으나 2014~15 구간 값에는 영향이 없어야 한다).
허용 오차: 일별 ±1원, 종료금 ±1원(사전 고정).
"""
import json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, yaml
import stage2_common as sc
import pool_vs_index_common as pc
from src.backtest.cpu.execution import BasicExecutionHandler

WINDOW = ('2014-01-01', '2015-12-31')
CASH0, W, NMAX, COMM = 1e7, 0.05, 20, 0.00015       # 2단계 규칙 값(그대로)
CA_MIN_CHANGE, DPS_MAX_YIELD = 0.002, 0.3          # 과거 규칙 재현 전용: 30% 초과 DPS 제외 유지
TOL_WON = 1.0


def ticks():
    base = yaml.safe_load(open('config/config.yaml'))
    exh = BasicExecutionHandler(**{k: v for k, v in base['execution_params'].items() if k == 'sell_price_tick_rounding'})
    sell = getattr(exh, '_adjust_sell_price', exh._adjust_price_up)
    return (lambda p: float(exh._adjust_price_up(np.float32(p)))), (lambda p: float(sell(np.float32(p))))


def load(conn, panel, cap):
    tick = sorted(panel.ticker.unique()); ph = ','.join(['%s'] * len(tick))
    px = pc.capped_query(conn, f"SELECT stock_code, date, open_price, close_price, adj_ratio FROM DailyStockPrice WHERE stock_code IN ({ph}) "
                               f"AND date BETWEEN '2013-12-01' AND %s", tick + [str(cap.date())], cap)
    px['date'] = pd.to_datetime(px.date)
    _, full = pc.trading_calendar(conn, cap)
    cal = sc.Calendar(full); D = cal.dates[(cal.dates >= '2013-12-01') & (cal.dates <= cap)]
    mat = lambda c: px.pivot(index='date', columns='stock_code', values=c).reindex(index=D, columns=tick).astype(float).values
    OPEN, CLOSE, RATIO = mat('open_price'), mat('close_price'), mat('adj_ratio')
    CLOSE_F, RATIO_F = pd.DataFrame(CLOSE).ffill().values, pd.DataFrame(RATIO).ffill().values
    CA = np.ones_like(RATIO_F)
    with np.errstate(invalid='ignore', divide='ignore'): rr = RATIO_F[1:] / RATIO_F[:-1]
    CA[1:] = np.where(np.isfinite(rr) & (np.abs(rr - 1) > CA_MIN_CHANGE), rr, 1.0)
    years = [y for y in range(2013, 2026) if pd.Timestamp(f'{y + 1}-05-15') <= cap]
    dps = sc.dps_table(conn, tick, years)
    return {'tick': tick, 'tix': {t: i for i, t in enumerate(tick)}, 'D': D, 'cal': cal, 'OPEN': OPEN, 'CLOSE_F': CLOSE_F, 'CA': CA,
            'DPS': {(r.ticker, r.fy): (r.dps, r.dps_fallback) for r in dps.itertuples()}}


def legacy_orders(X, q, cash, i, V, L, up, dn):
    tick, tix, OPEN = X['tick'], X['tix'], X['OPEN']
    tgt = {t: np.floor(W * V / up(OPEN[i, tix[t]])) for t in L if np.isfinite(OPEN[i, tix[t]]) and OPEN[i, tix[t]] > 0}
    for j in np.where(q > 0)[0]:
        t = tick[j]; o = OPEN[i, j]
        if not (np.isfinite(o) and o > 0) or (t in L and t not in tgt): continue
        tq = tgt.get(t, 0.0)
        if q[j] > tq:
            dq = q[j] - tq; cash += np.floor(dq * dn(o) * (1 - COMM - X['cal'].sell_tax(X['D'][i]))); q[j] -= dq
    for t in L:
        if t not in tgt: continue
        j = tix[t]; dq = tgt[t] - q[j]
        if dq <= 0: continue
        p = up(OPEN[i, j]); cost = lambda n: n * p + np.floor(n * p * COMM)
        if cost(dq) > cash: dq = np.floor(cash / (p * (1 + COMM)))
        while dq > 0 and cost(dq) > cash: dq -= 1
        if dq <= 0: continue
        cash -= cost(dq); q[j] += dq
    return q, cash


def legacy_dividend(X, q, i, fy):
    amt = 0.0
    for j in np.where(q > 0)[0]:
        v = X['DPS'].get((X['tick'][j], fy)); c = X['CLOSE_F'][i, j]
        if v is None or not np.isfinite(c) or c <= 0 or v[0] / c > DPS_MAX_YIELD: continue
        amt += q[j] * v[0]
    return amt


def simulate_legacy(X, lists, s, e, up, dn):
    D, cal = X['D'], X['cal']; idx = np.where((D >= s) & (D <= e))[0]
    q = np.zeros(len(X['tick'])); cash = CASH0; recv = {}; vals = []
    years = sorted(set(D[idx].year)); rec = {cal.record_date(y): y for y in years}; pay = {cal.pay_date(y): y for y in years}
    for k, i in enumerate(idx):
        d = D[i]; q = q * X['CA'][i]
        if d in pay and pay[d] in recv: cash += recv.pop(pay[d])
        if d in lists:
            V = CASH0 if k == 0 else cash + float(np.nansum(q * X['CLOSE_F'][i - 1])) + sum(recv.values())
            q, cash = legacy_orders(X, q, cash, i, V, lists[d], up, dn)
        if d in rec: recv[rec[d]] = recv.get(rec[d], 0.0) + legacy_dividend(X, q, i, rec[d])
        vals.append(cash + float(np.nansum(q * X['CLOSE_F'][i])) + sum(recv.values()))
    return pd.Series(vals, index=D[idx])


def compare(v, ref, ref_final):
    diff = (v.reindex(ref.index) - ref).abs()
    res = {'window': WINDOW, 'n_days_ref': len(ref), 'n_days_new': len(v), 'missing_days': int(diff.isna().sum()),
           'max_abs_daily_diff': float(diff.max()), 'final_new': float(v.iloc[-1]), 'final_ref': ref_final,
           'final_diff': float(v.iloc[-1] - ref_final), 'tolerance_won': TOL_WON}
    res['pass'] = bool(res['missing_days'] == 0 and len(v) == len(ref) and res['max_abs_daily_diff'] <= TOL_WON and abs(res['final_diff']) <= TOL_WON)
    return res


def main():
    panel_path, stage2, out = sys.argv[1:4]
    cap = pc.cap_date(pc.END_CAP); conn = sc.connect()
    panel = pd.read_csv(panel_path, dtype={'ticker': str}, parse_dates=['trade_date', 'signal_date'])
    X = load(conn, panel, cap); up, dn = ticks()
    s, e = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])
    lists = {d: list(g.sort_values('rank').ticker.iloc[:NMAX]) for d, g in panel.groupby('trade_date') if s <= d <= e}
    v = simulate_legacy(X, lists, s, e, up, dn)
    ref = pd.read_csv(os.path.join(stage2, 'sim_v2', 'curves_2014.csv'), parse_dates=['date']).set_index('date')['2014_S20'].dropna()
    acc = pd.read_csv(os.path.join(stage2, 'sim_v2', 'accounts_2014.csv'))
    ref_final = float(acc[(acc.account == 'S20') & (acc.variant == 'div_approx')]['final'].iloc[0])
    res = compare(v, ref, ref_final)
    v.rename('S20_legacy').to_frame().to_csv(out.replace('.json', '') + '_daily.csv', index_label='date')
    if os.path.exists(out): raise SystemExit(f'refuse to overwrite {out}')
    json.dump(res, open(out, 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if res['pass'] else 1)      # F10: 실패면 종료 코드 비0


if __name__ == '__main__':
    main()
