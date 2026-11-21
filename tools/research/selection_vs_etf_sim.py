
"""2단계 비교: 시장 ETF(KODEX 200), 종목 선정만(S20), 무작위 대조(R20)를 실제(수정 전) 가격으로 모의 실행한다.

usage: python -B tools/research/selection_vs_etf_sim.py PANEL.csv ETF_069500.csv OUT_DIR [--seeds 100]
저장소 루트(3월 실행 체크아웃)에서 실행. DB는 읽기만 한다. 설계는
todos/2026_11_21-stock-selection-and-additional-buys.md의 "2단계 실행 설계" 절을 따른다.
"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, yaml
import stage2_common as sc
from src.backtest.cpu.execution import BasicExecutionHandler

ap = argparse.ArgumentParser()
ap.add_argument('panel'); ap.add_argument('etf'); ap.add_argument('outdir')
ap.add_argument('--seeds', type=int, default=100)
ap.add_argument('--window', default='', help='시작 연도 하나만 실행(예: 2014)')
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
WINDOWS = [('2014-01-01', '2015-12-31'), ('2016-01-01', '2017-12-31'), ('2018-01-01', '2019-12-31'),
           ('2020-01-01', '2021-12-31'), ('2022-01-01', '2023-12-31'), ('2024-01-01', '2025-12-31')]
CASH0, W, NMAX, COMM = 1e7, 0.05, 20, 0.00015
base = yaml.safe_load(open('config/config.yaml'))
ex_kw = {k: v for k, v in base['execution_params'].items() if k in ('sell_price_tick_rounding',)}
exh = BasicExecutionHandler(**ex_kw)
conn = sc.connect(); cal = sc.Calendar(sc.trading_dates(conn))
panel = pd.read_csv(a.panel, dtype={'ticker': str}, parse_dates=['trade_date', 'signal_date'])
tick = sorted(panel.ticker.unique()); tix = {t: i for i, t in enumerate(tick)}; NT = len(tick)
ph = ','.join(['%s'] * NT)
px = sc.q(conn, f"SELECT stock_code, date, open_price, close_price, adj_ratio FROM DailyStockPrice WHERE stock_code IN ({ph}) AND date BETWEEN '2013-12-01' AND '2026-01-31'", tick)
px['date'] = pd.to_datetime(px.date)
D = cal.dates[(cal.dates >= '2013-12-01') & (cal.dates <= '2026-01-31')]; dix = {d: i for i, d in enumerate(D)}
def mat(col):
    m = px.pivot(index='date', columns='stock_code', values=col).reindex(index=D, columns=tick).astype(float); return m.values
OPEN, CLOSE, RATIO = mat('open_price'), mat('close_price'), mat('adj_ratio')
CLOSE_F = pd.DataFrame(CLOSE).ffill().values
RATIO_F = pd.DataFrame(RATIO).ffill().values
# 기업행위로 수량이 바뀌는 비율(수정계수 변화 0.2% 초과만)
CA = np.ones_like(RATIO_F)
with np.errstate(invalid='ignore', divide='ignore'):
    rr = RATIO_F[1:] / RATIO_F[:-1]
rr = np.where(np.isfinite(rr) & (np.abs(rr - 1) > 0.002), rr, 1.0); CA[1:] = rr
mk = sc.q(conn, f"SELECT stock_code, market_type, MIN(snapshot_date) a, MAX(snapshot_date) b FROM TickerUniverseSnapshot WHERE stock_code IN ({ph}) GROUP BY stock_code, market_type", tick)
def market_asof(t, d):
    g = mk[mk.stock_code == t]
    ok = g[pd.to_datetime(g.a) <= d]
    return None if ok.empty else ok.sort_values('b').iloc[-1].market_type
panel['market'] = [market_asof(t, d) for t, d in zip(panel.ticker, panel.signal_date)]
dl = sc.q(conn, f"SELECT stock_code, delisted_date FROM TickerUniverseHistory WHERE stock_code IN ({ph}) AND delisted_date IS NOT NULL", tick)
DELIST = {r.stock_code: pd.Timestamp(r.delisted_date) for r in dl.itertuples()}
DPS = sc.dps_table(conn, tick, list(range(2013, 2026)))
DPSd = {(r.ticker, r.fy): (r.dps, r.dps_fallback) for r in DPS.itertuples()}
etf = pd.read_csv(a.etf, parse_dates=['date']).set_index('date').reindex(D).ffill()
months = {d: g.sort_values('rank') for d, g in panel.groupby('trade_date')}

def up(p): return float(exh._adjust_price_up(np.float32(p)))
_sell_round = getattr(exh, '_adjust_sell_price', exh._adjust_price_up)   # 3월 체크아웃은 매도도 올림
def dn(p): return float(_sell_round(np.float32(p)))

def simulate(lists, s, e, div='approx', delist_zero=False, band=0.0):
    """lists: {trade_date: [tickers in buy order]} (빈 목록이면 그달 주문 없음)."""
    idx = np.where((D >= s) & (D <= e))[0]
    q = np.zeros(NT); cash = CASH0; recv = {}; vals = []; inv = []; traded = 0.0; ndelist = set(); events = []
    years = sorted(set(D[idx].year)); rec = {cal.record_date(y): y for y in years}; pay = {cal.pay_date(y): y for y in years}
    for k, i in enumerate(idx):
        d = D[i]
        q = q * CA[i]
        if delist_zero:
            for t, dd in DELIST.items():
                j = tix[t]
                if q[j] > 0 and d >= dd: q[j] = 0; ndelist.add(t)
        if div != 'none' and d in pay and pay[d] in recv:
            cash += recv.pop(pay[d])
        if d in lists:
            V = CASH0 if k == 0 else cash + float(np.nansum(q * CLOSE_F[i - 1])) + sum(recv.values())
            L = [t for t in lists[d]]
            tgt = {}
            for t in L:
                o = OPEN[i, tix[t]]
                if np.isfinite(o) and o > 0: tgt[t] = np.floor(W * V / up(o))
            held = np.where(q > 0)[0]
            for j in held:   # 매도 먼저
                t = tick[j]; o = OPEN[i, j]
                if not (np.isfinite(o) and o > 0): continue
                if t in L and t not in tgt: continue
                tq = tgt.get(t, 0.0)
                if q[j] > tq:
                    dq = q[j] - tq
                    if t in tgt and band > 0 and dq * o < band * V: continue
                    p = dn(o); rev = dq * p; net = np.floor(rev * (1 - COMM - cal.sell_tax(d)))
                    cash += net; q[j] -= dq; traded += rev
            for t in L:      # 목록 순서대로 매수
                if t not in tgt: continue
                j = tix[t]; dq = tgt[t] - q[j]
                if dq <= 0: continue
                p = up(OPEN[i, j])
                if band > 0 and q[j] > 0 and dq * p < band * V: continue
                cost = lambda n: n * p + np.floor(n * p * COMM)
                if cost(dq) > cash: dq = np.floor(cash / (p * (1 + COMM)))
                while dq > 0 and cost(dq) > cash: dq -= 1
                if dq <= 0: continue
                cash -= cost(dq); q[j] += dq; traded += dq * p
        if div != 'none' and d in rec:
            fy = rec[d]; amt = 0.0
            for j in np.where(q > 0)[0]:
                t = tick[j]; v = DPSd.get((t, fy))
                c = CLOSE_F[i, j]
                if v is None or not np.isfinite(c) or c <= 0 or v[0] / c > 0.3: continue
                amt += q[j] * v[0]
            if div == 'upper':
                nd = cal.shift(d, 1)
                pay[nd] = fy; recv[fy] = recv.get(fy, 0.0) + amt
            else: recv[fy] = recv.get(fy, 0.0) + amt
            events.append((str(d.date()), fy, amt))
        hv = float(np.nansum(q * CLOSE_F[i])); tv = cash + hv + sum(recv.values())
        vals.append(tv); inv.append(hv / tv if tv > 0 else 0.0)
    v = pd.Series(vals, index=D[idx])
    return v, {'avg_invested': float(np.mean(inv)), 'turnover_per_year': traded / 2 / float(v.mean()) / (len(v) / 252),
               'delisted_zeroed': len(ndelist), 'div_events': events}

def metrics(v):
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / CASH0) ** (1 / yrs) - 1
    mdd = float((v / v.cummax() - 1).min()); r = v.pct_change().dropna()
    return {'cagr': cagr * 100, 'mdd': mdd * 100, 'sharpe': float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan, 'final': float(v.iloc[-1])}

def etf_account(s, e):
    idx = np.where((D >= s) & (D <= e))[0]; o = float(etf.open.iloc[idx[0]])
    n = np.floor(CASH0 / (o * (1 + COMM))); cash = CASH0 - (n * o + np.floor(n * o * COMM))
    return pd.Series(cash + n * etf.close.values[idx], index=D[idx])

def s20_lists(s, e):
    out = {}
    for d, g in months.items():
        if s <= d <= e: out[d] = list(g.ticker.iloc[:NMAX])
    return out

def mkt(d, t):
    g = months[d]; return g.set_index('ticker').market.get(t)

def r20_lists(s, e, seed, kind):
    rng = np.random.default_rng(seed); prev_s, prev_r = set(), []; out = {}; gaps = 0
    score = {t: rng.random() for t in tick}
    for d in sorted(x for x in months if s <= x <= e):
        g = months[d]; S = list(g.ticker.iloc[:NMAX]); E = list(g.ticker)
        pick = []
        for m in ('KOSPI', 'KOSDAQ'):
            Sm = [t for t in S if mkt(d, t) == m]; Em = [t for t in E if mkt(d, t) == m]
            n = len(Sm)
            if kind == 'fixed':
                pick += sorted(Em, key=lambda t: score[t], reverse=True)[:n]; continue
            if kind == 'redraw':
                pick += list(rng.permutation(Em)[:n]) if n else []; continue
            new_n = len([t for t in Sm if t not in prev_s]); keep_n = n - new_n
            Hm = [t for t in prev_r if t in Em]
            kept = list(rng.permutation(Hm)[:keep_n]) if keep_n > 0 and Hm else []
            pool = [t for t in Em if t not in Hm]
            need = n - len(kept)
            newp = list(rng.permutation(pool)[:need]) if need > 0 and pool else []
            if len(newp) < need:
                rest = [t for t in Em if t not in kept and t not in newp]
                newp += list(rng.permutation(rest)[:need - len(newp)]) if rest else []
            if len(kept) != keep_n or len(newp) != need: gaps += 1
            pick += kept + newp
        pick = list(rng.permutation(pick)) if pick else []
        out[d] = pick; prev_s = set(S); prev_r = pick
    return out, gaps

rows = []; curves = {}
for s, e in WINDOWS:
    s, e = pd.Timestamp(s), pd.Timestamp(e); w = str(s.year)
    if a.window and a.window != w: continue
    ve = etf_account(s, e); me = metrics(ve); rows.append({'window': w, 'account': 'ETF', 'variant': 'total_return', **me, 'avg_invested': 1.0}); curves[(w, 'ETF')] = ve
    L = s20_lists(s, e)
    for div in ('none', 'approx', 'upper'):
        for dz in (False, True):
            v, x = simulate(L, s, e, div=div, delist_zero=dz)
            rows.append({'window': w, 'account': 'S20', 'variant': f'div_{div}' + ('_delist0' if dz else ''), **metrics(v),
                         **{k: x[k] for k in ('avg_invested', 'turnover_per_year', 'delisted_zeroed')}})
            if div == 'approx' and not dz: curves[(w, 'S20')] = v
    v, x = simulate(L, s, e, div='approx', band=0.01)
    rows.append({'window': w, 'account': 'S20', 'variant': 'div_approx_band1', **metrics(v), **{k: x[k] for k in ('avg_invested', 'turnover_per_year', 'delisted_zeroed')}})
    for kind in ('matched', 'fixed', 'redraw'):
        for seed in range(a.seeds):
            Lr, gaps = r20_lists(s, e, 20261121 + seed, kind)
            for div in (('none', 'approx') if kind == 'matched' else ('approx',)):
                v, x = simulate(Lr, s, e, div=div)
                rows.append({'window': w, 'account': f'R20_{kind}', 'variant': f'div_{div}', 'seed': seed, 'gaps': gaps, **metrics(v),
                             **{k: x[k] for k in ('avg_invested', 'turnover_per_year', 'delisted_zeroed')}})
    print(w, 'done', flush=True)
sfx = f'_{a.window}' if a.window else ''
res = pd.DataFrame(rows); res.to_csv(os.path.join(a.outdir, f'accounts{sfx}.csv'), index=False)
pd.concat({f'{w}_{k}': v for (w, k), v in curves.items()}, axis=1).to_csv(os.path.join(a.outdir, f'curves{sfx}.csv'), index_label='date')
print(res[res.account.isin(['ETF', 'S20'])].to_string())
