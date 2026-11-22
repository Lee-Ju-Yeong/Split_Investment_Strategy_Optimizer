
"""기준 계좌의 최대낙폭 구간(고점->저점) 손익을 종목별·차수별로 나누고 합계를 계좌 평가액 변화와 대사한다.
usage: python -B tools/research/stage3_drawdown.py PREFIX   (split_window_runs.py 산출물 접두어)
"""
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
P = sys.argv[1]
c = pd.read_csv(P + '_curve.csv', parse_dates=['date']).set_index('date')
v = c.value; dd = v / v.cummax() - 1; trough = dd.idxmin(); peak = v[:trough].idxmax()
fl = pd.read_csv(P + '_fills.csv', dtype={'ticker': str}, parse_dates=['date'])
conn = sc.connect(); tk = sorted(fl.ticker.unique()); ph = ','.join(['%s'] * len(tk))
px = sc.q(conn, f"SELECT stock_code, date, adj_close FROM DailyStockPrice WHERE stock_code IN ({ph}) AND date <= %s", tk + [str(trough.date())])
px['date'] = pd.to_datetime(px.date); px['adj_close'] = px.adj_close.astype(float)
def close_asof(t, d):
    s = px[(px.stock_code == t) & (px.date <= d)]; return float(s.adj_close.iloc[-1]) if len(s) else np.nan
# 차수별 보유: 매수 체결의 order로 들어가고, 매도는 그 position의 order로 빠진다
fl['key_order'] = fl.order
def holding(d):
    h = fl[fl.date <= d].groupby(['ticker', 'key_order']).dq.sum(); return h[h != 0]
h0, h1 = holding(peak), holding(trough)
mid = fl[(fl.date > peak) & (fl.date <= trough)]
keys = sorted(set(h0.index) | set(h1.index) | set(zip(mid.ticker, mid.key_order)))
rows = []
for t, o in keys:
    q0, q1 = h0.get((t, o), 0), h1.get((t, o), 0)
    flow = mid[(mid.ticker == t) & (mid.key_order == o)].dcash.sum()
    rows.append({'ticker': t, 'order': o, 'value_peak': q0 * close_asof(t, peak), 'value_trough': q1 * close_asof(t, trough), 'trade_cash': flow})
r = pd.DataFrame(rows); r['pnl'] = r.value_trough - r.value_peak + r.trade_cash
dv = pd.read_csv(P + '_divs.csv') if os.path.getsize(P + '_divs.csv') > 5 else pd.DataFrame()
paid = 0.0
if len(dv) and 'event' in dv:
    p_ = dv[dv.event == 'paid']; paid = float(p_[(pd.to_datetime(p_.date) > peak) & (pd.to_datetime(p_.date) <= trough)].amount.sum())
recv_chg = float(c.receivable.loc[trough] - c.receivable.loc[peak])
total = float(c.value_engine.loc[trough] - c.value_engine.loc[peak])
print('peak', peak.date(), round(v[peak]), 'trough', trough.date(), round(v[trough]), 'mdd%', round(dd.min() * 100, 2))
print('engine value change', round(total), '| sum pnl', round(r.pnl.sum()), '| dividends paid', round(paid), '| residual', round(total - r.pnl.sum() - paid),
      '| receivable change', round(recv_chg))
bo = r.groupby('order').pnl.sum(); print('by order:', bo.round(0).to_dict())
bt = r.groupby('ticker').pnl.sum().sort_values(); print('worst 10 tickers:', bt.head(10).round(0).to_dict())
print('worst-5 share of total loss', round(bt.head(5).sum() / r.pnl.sum(), 3), '| tickers', r.ticker.nunique())
r.to_csv(P + '_drawdown_decomp.csv', index=False)
