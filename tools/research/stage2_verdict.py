
"""2단계 판정표: ETF 대비 S20·현행 분할의 구간별 통과 여부(배당 하한·상한), 무작위 대조 대비 순위 위치.
usage: python -B tools/research/stage2_verdict.py RUN_DIR   (RUN_DIR/sim/accounts_*.csv, RUN_DIR/split/*_curve.csv)
"""
import glob, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
R = sys.argv[1]; CASH0 = 1e7
acc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, 'sim', 'accounts_*.csv')))], ignore_index=True)
acc['window'] = acc.window.astype(str)
def metrics(v):
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    r = v.pct_change().dropna()
    return {'cagr': ((v.iloc[-1] / CASH0) ** (1 / yrs) - 1) * 100, 'mdd': float((v / v.cummax() - 1).min()) * 100,
            'sharpe': float(r.mean() / r.std() * np.sqrt(252)), 'final': float(v.iloc[-1])}
conn = sc.connect()
split_rows = []
for f in sorted(glob.glob(os.path.join(R, 'split', '*_curve.csv'))):
    tag = os.path.basename(f)[:-10]; w, tax, div = tag.split('_')
    c = pd.read_csv(f, parse_dates=['date']).set_index('date')
    m = metrics(c.value)
    split_rows.append({'window': w, 'account': 'SPLIT', 'variant': f'tax_{tax}_div_{div}', **m})
    if tax == 'period' and div == 'approx':   # 상장폐지 0원 민감도
        fl = pd.read_csv(os.path.join(R, 'split', tag + '_fills.csv'), dtype={'ticker': str}, parse_dates=['date'])
        dl = sc.q(conn, "SELECT stock_code, delisted_date FROM TickerUniverseHistory WHERE delisted_date BETWEEN %s AND %s", [str(c.index[0].date()), str(c.index[-1].date())])
        hit = [t for t in dl.stock_code if t in set(fl.ticker)]
        adj = c.value.copy(); n_hit = 0
        for t in hit:
            dd = pd.Timestamp(dl[dl.stock_code == t].delisted_date.iloc[0])
            qty = fl[(fl.ticker == t) & (fl.date < dd)].dq.sum()
            if qty <= 0: continue
            px = sc.q(conn, "SELECT date, adj_close FROM DailyStockPrice WHERE stock_code=%s AND date < %s ORDER BY date DESC LIMIT 1", [t, str(dd.date())])
            last = float(px.adj_close.iloc[0]); adj[adj.index >= dd] -= qty * last; n_hit += 1
        split_rows.append({'window': w, 'account': 'SPLIT', 'variant': 'tax_period_div_approx_delist0', 'delisted_zeroed': n_hit, **metrics(adj)})
allr = pd.concat([acc, pd.DataFrame(split_rows)], ignore_index=True)
etf = allr[allr.account == 'ETF'].set_index('window')
def passes(r):
    e = etf.loc[r.window]; return bool(r.cagr >= e.cagr + 1.0 and r.mdd >= e.mdd - 5.0)
allr['pass'] = [passes(r) if r.account != 'ETF' else None for r in allr.itertuples()]
allr['cagr_excess'] = [r.cagr - etf.loc[r.window].cagr for r in allr.itertuples()]
allr['mdd_gap'] = [r.mdd - etf.loc[r.window].mdd for r in allr.itertuples()]
allr.to_csv(os.path.join(R, 'stage2_all.csv'), index=False)
W5 = ['2014', '2016', '2018', '2020', '2022']
def get(acc_, var):
    return allr[(allr.account == acc_) & (allr.variant == var)].set_index('window')
out = {}
for name, lo, mid, hi in [('S20', 'div_none', 'div_approx', 'div_upper'), ('SPLIT', 'tax_period_div_none', 'tax_period_div_approx', 'tax_period_div_upper')]:
    L, M, H = get(name, lo), get(name, mid), get(name, hi)
    rows = []
    for w in W5 + ['2024']:
        pl, ph = bool(L.loc[w, 'pass']), bool(H.loc[w, 'pass'])
        rows.append({'window': w, 'etf_cagr': etf.loc[w].cagr, 'etf_mdd': etf.loc[w].mdd, 'cagr_lo': L.loc[w, 'cagr'], 'cagr_mid': M.loc[w, 'cagr'], 'cagr_hi': H.loc[w, 'cagr'],
                     'mdd_mid': M.loc[w, 'mdd'], 'pass_lo': pl, 'pass_hi': ph, 'verdict': 'pass' if pl and ph else ('fail' if not pl and not ph else 'undetermined')})
    t = pd.DataFrame(rows); out[name] = t
    print(f'== {name} vs ETF (CAGR excess >= 1%p, MDD gap >= -5%p)'); print(t.round(2).to_string(index=False))
    t5 = t[t.window.isin(W5)]
    print('pass windows (robust):', int((t5.verdict == 'pass').sum()), '/5; undetermined:', int((t5.verdict == 'undetermined').sum()))
R20 = allr[(allr.account == 'R20_matched') & (allr.variant == 'div_approx')]
S = get('S20', 'div_approx')
rk = []
for w in W5 + ['2024']:
    r = R20[R20.window == w].cagr
    rk.append({'window': w, 's20_cagr': S.loc[w, 'cagr'], 'r20_median': r.median(), 'r20_p10': r.quantile(.1), 'r20_p90': r.quantile(.9),
               's20_percentile': float((r < S.loc[w, 'cagr']).mean() * 100), 'r20_mdd_median': R20[R20.window == w].mdd.median(),
               'r20_pass_share': float(R20[R20.window == w]['pass'].mean()),
               'r20_fixed_median': allr[(allr.account == 'R20_fixed') & (allr.window == w)].cagr.median(),
               'r20_redraw_median': allr[(allr.account == 'R20_redraw') & (allr.window == w)].cagr.median()})
rk = pd.DataFrame(rk); print('== S20 vs random (matched, div approx)'); print(rk.round(2).to_string(index=False))
print('S20 above R20 median in', int((rk[rk.window.isin(W5)].s20_cagr > rk[rk.window.isin(W5)].r20_median).sum()), '/5')
sens = allr[allr.account.isin(['S20', 'SPLIT'])][['window', 'account', 'variant', 'cagr', 'mdd', 'pass', 'avg_invested', 'turnover_per_year', 'delisted_zeroed']]
print('== sensitivities'); print(sens.round(2).to_string(index=False))
rk.to_csv(os.path.join(R, 'stage2_random.csv'), index=False)
pd.concat(out, names=['account']).to_csv(os.path.join(R, 'stage2_verdict.csv'))
