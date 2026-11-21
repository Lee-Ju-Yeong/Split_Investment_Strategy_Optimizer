
"""addbuy_signal_audit.py가 남긴 추가매수 신호에 수익성 악화 조건을 붙여 연도별로 집계한다.

조건은 2026-11-21 신호와 대조하기 전에 고정했다(todos/2026_11_21-stock-selection-and-additional-buys.md).
- 상태: 같은 (EPS, BPS)가 이어진 FinancialData 행 묶음. 신호일 이하의 행만 쓴다.
- C1(주): 최신 상태와 직전 상태가 모두 EPS>0·BPS>0이고 근사 ROE(100*EPS/BPS)가 내려갔다.
- C2(보조): C1이면서 최신 근사 ROE < 5.
- EPS 0(적자·결측 혼재) 또는 BPS 0 상태는 판정 불가. 양수→EPS 0 전환은 pos_to_zero로 따로 센다.
DB는 config/config.yaml의 연결로 읽기만 한다.
"""
import sys, json, yaml, numpy as np, pandas as pd, mysql.connector
sig_path, out_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(sys.argv[3] if len(sys.argv) > 3 else 'config/config.yaml'))['database']
sig = pd.read_csv(sig_path, dtype={'ticker': str})
sig['signal_date'] = pd.to_datetime(sig.signal_date); sig['year'] = sig.order_date.str[:4]
tickers = sorted(sig.ticker.unique())
conn = mysql.connector.connect(host=cfg['host'], port=cfg.get('port',3306), user=cfg['user'], password=cfg['password'], database=cfg['database'], use_pure=True)
q = "select stock_code, date, eps, bps from FinancialData where stock_code in (%s) and date <= %%s order by stock_code, date" % ','.join(['%s']*len(tickers))
fin = pd.read_sql(q, conn, params=tickers + [str(sig.signal_date.max().date())]); conn.close()
fin['date'] = pd.to_datetime(fin.date)
key = fin.eps.astype(str) + '|' + fin.bps.astype(str)
fin['new_state'] = (key != key.groupby(fin.stock_code).shift()).astype(int)
fin['state_idx'] = fin.groupby('stock_code').new_state.cumsum()
st = fin[fin.new_state == 1][['stock_code','date','eps','bps','state_idx']].rename(columns={'date':'first_date'})
prev = st.copy(); prev['state_idx'] += 1
st = st.merge(prev.rename(columns={'first_date':'s0_first','eps':'eps0','bps':'bps0'}), on=['stock_code','state_idx'], how='left')
st = st.sort_values('first_date')
sig = sig.sort_values('signal_date')
m = pd.merge_asof(sig, st, left_on='signal_date', right_on='first_date', left_by='ticker', right_by='stock_code', direction='backward')
def ok(e,b): return np.isfinite(e) & np.isfinite(b) & (e > 0) & (b > 0)
m['has_prior'] = m.eps0.notna()
m['determinable'] = m.has_prior & ok(m.eps, m.bps) & ok(m.eps0, m.bps0)
m['roe1'] = 100*m.eps/m.bps; m['roe0'] = 100*m.eps0/m.bps0
m['C1'] = m.determinable & (m.roe1 < m.roe0)
m['C2'] = m.C1 & (m.roe1 < 5)
m['pos_to_zero'] = m.has_prior & ok(m.eps0, m.bps0) & (m.eps == 0) & (m.bps > 0)
m['undet_reason'] = np.where(m.determinable, '', np.where(~m.has_prior, 'no_prior_state', 'eps_or_bps_nonpositive'))
m['days_since_s1'] = (m.signal_date - m.first_date).dt.days
m['state_key'] = m.ticker + '@' + m.first_date.dt.strftime('%Y-%m-%d')
m.to_csv(out_path, index=False)
def table(df):
    g = df.groupby('year')
    t = pd.DataFrame({'signals': g.size(), 'tickers': g.ticker.nunique(), 'filled': g.filled.sum(),
        'determinable': g.determinable.sum(), 'undet_no_prior': g.apply(lambda x:(x.undet_reason=='no_prior_state').sum()),
        'undet_eps_bps': g.apply(lambda x:(x.undet_reason=='eps_or_bps_nonpositive').sum())})
    for c in ['C1','C2','pos_to_zero']:
        t[c] = g[c].sum(); t[c+'_filled'] = g.apply(lambda x:(x[c] & x.filled).sum())
        t[c+'_cash'] = g.apply(lambda x:(x[c] & (x.unfilled_reason=='cash_shortfall')).sum())
        t[c+'_other'] = g.apply(lambda x:(x[c] & ~x.filled & (x.unfilled_reason!='cash_shortfall')).sum())
    t.loc['total'] = t.sum(); t.loc['total','tickers'] = df.ticker.nunique()
    return t.astype(int)
def conc(df, c):
    f = df[df[c] & df.filled]; n = len(f); tf = int(df.filled.sum())
    if n == 0: return {'filled': 0, 'share_of_all_filled_addbuys': 0.0}
    vc = f.ticker.value_counts(); yc = f.year.value_counts()
    return {'filled': n, 'share_of_all_filled_addbuys': round(n/tf, 4), 'tickers': int(f.ticker.nunique()),
        'ticker_state_transitions': int(f.state_key.nunique()), 'years': int(f.year.nunique()),
        'top5_ticker_share': round(vc.head(5).sum()/n, 4), 'max_year_share': round(yc.max()/n, 4),
        'top5': vc.head(5).to_dict(), 'median_days_since_state_change': float(f.days_since_s1.median()),
        'median_roe0_roe1': [round(float(f.roe0.median()),2), round(float(f.roe1.median()),2)]}
pd.set_option('display.width', 250)
print(table(m).to_string())
print(m[~m.filled].groupby(['year','unfilled_reason']).size().unstack(fill_value=0).to_string())
print(json.dumps({c: conc(m, c) for c in ['C1','C2','pos_to_zero']}, ensure_ascii=False, indent=1, default=str))
print('fin_updated_rows', len(fin), 'tickers', len(tickers))
