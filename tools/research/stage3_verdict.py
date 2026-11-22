
"""3단계 판정: 추가매수 보류 계좌(C1·C1 보정·추가매수 끔·무작위 보류)를 기준 계좌·ETF와 비교한다.
usage: python -B tools/research/stage3_verdict.py STAGE3_DIR STAGE2_ALL.csv [--p-only]
--p-only: 기준 계좌 신호에서 구간별 무작위 보류 확률(판정 가능 종목×상태 중 C1 비율)만 출력한다.
"""
import glob, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
D, S2 = sys.argv[1], sys.argv[2]; CASH0 = 1e7
W5 = ['2014', '2016', '2018', '2020', '2022']; WALL = W5 + ['2024']
def p_window(w):
    s = pd.read_csv(os.path.join(D, f'{w}_none_addsignals.csv'), dtype={'ticker': str})
    u = s.drop_duplicates('state_key'); u = u[u.determinable]
    return round(float(u.c1.mean()), 4), len(u)
if '--p-only' in sys.argv:
    print(json.dumps({w: p_window(w) for w in WALL})); sys.exit(0)
def metrics(v):
    yrs = (v.index[-1] - v.index[0]).days / 365.25; r = v.pct_change().dropna()
    return {'cagr': ((v.iloc[-1] / CASH0) ** (1 / yrs) - 1) * 100, 'mdd': float((v / v.cummax() - 1).min()) * 100,
            'sharpe': float(r.mean() / r.std() * np.sqrt(252)), 'final': float(v.iloc[-1])}
s2 = pd.read_csv(S2, dtype={'window': str}); etf = s2[s2.account == 'ETF'].set_index('window')
def vs_etf(w, m):
    e = etf.loc[w]; ex = m['cagr'] - e.cagr; gap = m['mdd'] - e.mdd
    orig = ex >= 1 and gap >= -5
    calmar = m['cagr'] > 0 and e.cagr > 0 and m['cagr'] / abs(m['mdd']) >= e.cagr / abs(e.mdd)
    return orig, bool(ex >= 1 and (gap >= -5 or calmar))
rows = []
for f in sorted(glob.glob(os.path.join(D, '*_curve.csv'))):
    tag = os.path.basename(f)[:-10]; w, kind = tag.split('_', 1)
    c = pd.read_csv(f, parse_dates=['date']).set_index('date')
    fl = pd.read_csv(os.path.join(D, tag + '_fills.csv'))
    dv = pd.read_csv(os.path.join(D, tag + '_divs.csv')) if os.path.getsize(os.path.join(D, tag + '_divs.csv')) > 5 else pd.DataFrame()
    paid = dv[dv.get('event', pd.Series(dtype=str)) == 'paid'] if len(dv) and 'event' in dv else pd.DataFrame(columns=['date', 'amount'])
    cf = pd.concat([fl[['date', 'dcash']].rename(columns={'dcash': 'x'}), paid[['date', 'amount']].rename(columns={'amount': 'x'})])
    cf['date'] = pd.to_datetime(cf.date); cash = CASH0 + cf.groupby('date').x.sum().reindex(c.index, fill_value=0).cumsum()
    inv = float((1 - cash / c.value_engine).mean())
    m = metrics(c.value); o, s = vs_etf(w, m)
    hd = pd.read_csv(os.path.join(D, tag + '_holds.csv'))
    uf = pd.read_csv(os.path.join(D, tag + '_unfilled.csv'))
    ne = fl[fl.reason.astype(str).str.contains('신규')]
    rows.append({'window': w, 'kind': kind, **m, 'avg_invested': inv, 'pass_orig': o, 'pass_calmar': s,
                 'blocked_signals': len(hd), 'blocked_states': hd.state_key.nunique() if len(hd) else 0,
                 'blocked_fundable': int((hd.cash >= hd.amount).sum()) if len(hd) else 0,
                 'blocked_fundable_amt': float(hd[hd.cash >= hd.amount].amount.sum()) if len(hd) else 0.0,
                 'addbuy_fills': int(((fl.type == 'BUY') & ~fl.reason.astype(str).str.contains('신규')).sum()),
                 'new_entry_amt': float(-ne.dcash.sum()), 'unfilled_buys': len(uf)})
r = pd.DataFrame(rows); r.to_csv(os.path.join(D, 'stage3_all.csv'), index=False)
pd.set_option('display.width', 250)
main = r[~r.kind.str.startswith('random')]
print('== 계좌별(구간 x 규칙)'); print(main.sort_values(['window', 'kind']).round(2).to_string(index=False))
out = []
for w in WALL:
    b = r[(r.window == w) & (r.kind == 'none')].iloc[0]
    rnd = r[(r.window == w) & r.kind.str.match(r'random_s\d+$')]
    rnd2 = r[(r.window == w) & r.kind.str.startswith('random5855')]
    for k in ('c1', 'c1adj', 'noadd'):
        x = r[(r.window == w) & (r.kind == k)].iloc[0]
        better = (x.cagr - b.cagr >= 0.5) and (x.mdd - b.mdd >= -2)
        n_ge = int((rnd.cagr >= x.cagr).sum())
        out.append({'window': w, 'rule': k, 'cagr_vs_base': x.cagr - b.cagr, 'mdd_vs_base': x.mdd - b.mdd, 'better_than_base': better,
                    'random_n': len(rnd), 'random_ge_cagr': n_ge, 'top4': n_ge <= 3, 'candidate_window': bool(better and n_ge <= 3),
                    'random_cagr_med': rnd.cagr.median(), 'random_cagr_p10': rnd.cagr.quantile(.1), 'random_cagr_p90': rnd.cagr.quantile(.9),
                    'random5855_cagr_med': rnd2.cagr.median() if len(rnd2) else np.nan,
                    'blocked_fundable': x.blocked_fundable, 'random_blocked_fundable_med': rnd.blocked_fundable.median(),
                    'random_blocked_fundable_p10_p90': [rnd.blocked_fundable.quantile(.1), rnd.blocked_fundable.quantile(.9)]})
o = pd.DataFrame(out); o.to_csv(os.path.join(D, 'stage3_rules.csv'), index=False)
print('== 보류 규칙 vs 기준·무작위'); print(o.round(2).to_string(index=False))
for k in ('c1', 'c1adj', 'noadd'):
    t = o[(o.rule == k) & o.window.isin(W5)]
    print(k, 'candidate windows', int(t.candidate_window.sum()), '/5')
# 연도·차수별 보류(기준 경로에서 C1이면 보류됐을 신호 vs 실제 C1 계좌의 보류)
by = []
for w in WALL:
    for k in ('c1', 'c1adj'):
        h = pd.read_csv(os.path.join(D, f'{w}_{k}_holds.csv'))
        if len(h):
            h['year'] = h.date.str[:4]; h['fundable'] = h.cash >= h.amount
            g = h.groupby(['year', 'order']).agg(signals=('ticker', 'size'), fundable=('fundable', 'sum'), amount=('amount', lambda s: s[h.loc[s.index, 'fundable']].sum()))
            g['rule'] = k; by.append(g.reset_index())
pd.concat(by).to_csv(os.path.join(D, 'stage3_holds_by_year_order.csv'), index=False)
print('== C1 보류(연도x차수, 체결 가능 = 그때 현금 >= 주문액)')
print(pd.concat(by).query("rule=='c1'").pivot_table(index='year', columns='order', values='fundable', aggfunc='sum', fill_value=0).to_string())
