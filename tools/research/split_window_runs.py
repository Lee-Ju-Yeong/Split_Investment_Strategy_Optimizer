
"""3월 승인 후보(분할매매)를 구간별로 다시 실행하고 일별 평가액·체결 수량·배당 사건을 남긴다.

usage: python -B tools/research/split_window_runs.py START END OUT_PREFIX [--tax period|fixed] [--div none|approx|upper]
저장소 루트(3월 실행 체크아웃)에서 MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행. 제품 코드는 바꾸지 않고 실행 중에만
체결·매도 신호 함수를 감싼다.
- --tax period: 매도세율을 결제일(T+2 거래일)의 세율로 바꾼다. fixed는 설정값(0.18%) 그대로.
- --div approx: FY y 배당을 y년 마지막 거래일 전날 종가 보유 수량(수정 수량 x 그날 adj_ratio)으로 계산해
  미수금으로 기록하고 y+1년 5월 첫 거래일에 현금으로 넣는다(그 뒤 전략이 쓸 수 있음).
  --div upper: 같은 금액을 권리 확정 다음 거래일에 바로 현금으로 넣는다(상한 근사). none은 배당 없음.
- --hold (3단계): 추가매수 신호를 버리는 규칙. none(기준), c1, c1adj(주식 수 변화 보정), random(--seed, 확률 0.5855), noadd(전부).
  버린 신호는 OUT_PREFIX_holds.csv에 남는다.
"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hashlib
import numpy as np, pandas as pd, yaml
from src.backtest.cpu.execution import BasicExecutionHandler
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.analysis import walk_forward_analyzer as wfa
import stage2_common as sc

ap = argparse.ArgumentParser()
ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('prefix')
ap.add_argument('--tax', choices=['period', 'fixed'], default='period')
ap.add_argument('--div', choices=['none', 'approx', 'upper'], default='none')
ap.add_argument('--hold', choices=['none', 'c1', 'c1adj', 'random', 'noadd'], default='none')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--hold-prob', type=float, default=0.5855)
ap.add_argument('--manifest', default='results/wfo_run_20260324_125326/final_candidate_manifest.json')
a = ap.parse_args()
base = yaml.safe_load(open('config/config.yaml'))
params = dict(json.load(open(a.manifest))['champion_params'])
conn = sc.connect(); cal = sc.Calendar(sc.trading_dates(conn))
s, e = pd.Timestamp(a.start), pd.Timestamp(a.end)
years = list(range(s.year, e.year + 1))
rec = {cal.record_date(y): y for y in years}
after_rec = {cal.shift(r, 1): y for r, y in rec.items()}
pay = {y: cal.pay_date(y, min_day=16) for y in years}   # 3단계: DPS를 5/15까지 읽으므로 그 뒤 첫 거래일에 지급(2단계는 5월 첫 거래일)
fills, divs = [], []
pending = {}   # fy -> amount
orig_exec = BasicExecutionHandler.execute_order
def exec_wrap(self, ev, portfolio, dh, idx):
    t = ev['ticker']; q0 = sum(p.quantity for p in portfolio.positions.get(t, [])); c0 = float(portfolio.cash)
    if a.tax == 'period': self.sell_tax_rate = cal.sell_tax(ev['date'])
    orig_exec(self, ev, portfolio, dh, idx)
    q1 = sum(p.quantity for p in portfolio.positions.get(t, []))
    if q1 == q0 and ev['type'] == 'BUY':
        unfilled.append({'date': str(pd.Timestamp(ev['date']).date()), 'ticker': t, 'reason': ev.get('reason_for_trade', ''),
                         'amount': float(ev.get('investment_amount', 0)), 'cash': c0})
    if q1 != q0:
        pos = ev.get('position'); od = getattr(pos, 'order', None)
        fills.append({'date': str(pd.Timestamp(ev['date']).date()), 'ticker': t, 'type': ev['type'], 'dq': int(q1 - q0), 'dcash': float(portfolio.cash) - c0,
                      'order': int(od) if od is not None else None, 'reason': ev.get('reason_for_trade', '')})
BasicExecutionHandler.execute_order = exec_wrap
# --- 3단계: 추가매수 보류 ---
_fin, _shr = {}, {}
def states(t):
    if t not in _fin:
        f = sc.q(conn, "SELECT date, eps, bps FROM FinancialData WHERE stock_code=%s AND date <= %s ORDER BY date", [t, str(e.date())])
        f['date'] = pd.to_datetime(f.date); key = f.eps.astype(str) + '|' + f.bps.astype(str)
        st = f[key != key.shift()].reset_index(drop=True)
        _fin[t] = st[['date', 'eps', 'bps']].astype({'eps': float, 'bps': float})
    return _fin[t]
def shares_asof(t, d):
    k = (t, d)
    if k not in _shr:
        r = sc.q(conn, "SELECT shares_outstanding FROM MarketCapDaily WHERE stock_code=%s AND date <= %s ORDER BY date DESC LIMIT 1", [t, str(pd.Timestamp(d).date())])
        _shr[k] = float(r.shares_outstanding.iloc[0]) if len(r) and r.shares_outstanding.iloc[0] else np.nan
    return _shr[k]
pos_ = lambda r: r is not None and r.eps > 0 and r.bps > 0
_ratio = {}
def ratio_asof(t, d):
    if t not in _ratio:
        r = sc.q(conn, "SELECT date, adj_ratio FROM DailyStockPrice WHERE stock_code=%s AND date <= %s ORDER BY date", [t, str(e.date())])
        r['date'] = pd.to_datetime(r.date); _ratio[t] = r.set_index('date').adj_ratio.astype(float)
    s = _ratio[t]; s = s[s.index <= pd.Timestamp(d)].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan
def rescale(t, new, old):
    """new 상태가 old 상태의 분할·무상증자·주식배당 비율 k(수정계수 변화, 신호일 이전 사건)로만 설명되면 True.
    원 단위 반올림 허용: |k*new - old| <= (1+k)/2 (EPS·BPS 각각)."""
    if not (pos_(new) and pos_(old)): return False
    k = ratio_asof(t, new.date) / ratio_asof(t, old.date)
    if not np.isfinite(k) or abs(k - 1) <= 0.005: return False
    tol = (1 + k) / 2
    return abs(k * new.eps - old.eps) <= tol and abs(k * new.bps - old.bps) <= tol
def judge(t, sd):
    """(state_key, determinable, c1, c1adj_determinable, c1adj)"""
    st = states(t); st = st[st.date <= pd.Timestamp(sd)].reset_index(drop=True)
    if len(st) == 0: return None, False, False, False, False
    s1 = st.iloc[-1]; s0 = st.iloc[-2] if len(st) >= 2 else None
    key = f"{t}@{s1.date.date()}"
    det = pos_(s1) and pos_(s0)
    c1 = bool(det and s1.eps / s1.bps < s0.eps / s0.bps)
    j = len(st) - 1                       # 중립 전환(순수 주식 수 변화)은 연속 병합
    while j >= 1 and rescale(t, st.iloc[j], st.iloc[j - 1]): j -= 1
    if j == len(st) - 1: return key, det, c1, det, c1
    if j < 1: return key, det, c1, False, False
    a_, b_ = st.iloc[j], st.iloc[j - 1]
    dadj = pos_(a_) and pos_(b_)
    return key, det, c1, dadj, bool(dadj and a_.eps / a_.bps < b_.eps / b_.bps)
def rnd_block(key):
    h = hashlib.sha256(f"{a.seed}|{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12) < a.hold_prob
holds, sigrec, unfilled = [], [], []
orig_add = MagicSplitStrategy.generate_additional_buy_signals
def add_wrap(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
    sigs = orig_add(self, current_date, portfolio, data_handler, trading_dates, current_day_idx)
    if not sigs: return sigs
    sd = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
    keep = []
    for sg in sigs:
        t = sg['ticker']
        key, det, c1, dadj, c1adj = judge(t, sd)
        sigrec.append({'date': str(pd.Timestamp(current_date).date()), 'ticker': t, 'state_key': key, 'order': int(sg['position'].order),
                       'determinable': det, 'c1': c1, 'c1adj_determinable': dadj, 'c1adj': c1adj})
        blk = {'none': False, 'noadd': True, 'c1': c1, 'c1adj': c1adj, 'random': det and rnd_block(key)}[a.hold]
        sigrec[-1]['blocked'] = blk
        if blk:
            holds.append({'date': str(pd.Timestamp(current_date).date()), 'ticker': t, 'state_key': key, 'order': int(sg['position'].order),
                          'amount': float(sg['investment_amount']), 'cash': float(portfolio.cash)})
        else: keep.append(sg)
    return keep
MagicSplitStrategy.generate_additional_buy_signals = add_wrap
orig_sell = MagicSplitStrategy.generate_sell_signals
def sell_wrap(self, current_date, portfolio, data_handler, trading_dates, idx=None):
    d = pd.Timestamp(current_date)
    if a.div != 'none':
        if d in after_rec:     # 전날 종가 보유 = 권리 확정 보유
            fy = after_rec[d]; r = cal.shift(d, -1)
            held = {t: sum(p.quantity for p in ps) for t, ps in portfolio.positions.items()}
            dp = sc.dps_table(conn, held.keys(), [fy - 1, fy]); dp = dp[dp.fy == fy]; dp = dict(zip(dp.ticker, zip(dp.dps, dp.dps_fallback)))
            ar = sc.adj_ratio_on(conn, held.keys(), r)
            amt = 0.0
            for t, qa in held.items():
                if t not in dp or t not in ar: continue
                raw = qa * ar[t][0]; v = raw * dp[t][0]
                if ar[t][1] > 0 and dp[t][0] / ar[t][1] > 0.3: continue   # 비정상 배당률은 넣지 않음
                amt += v; divs.append({'fy': fy, 'record_date': str(r.date()), 'ticker': t, 'adj_qty': qa, 'raw_qty': raw, 'dps': dp[t][0], 'fallback': bool(dp[t][1]), 'amount': v})
            if a.div == 'upper': portfolio.update_cash(amt); divs.append({'fy': fy, 'event': 'paid', 'date': str(d.date()), 'amount': amt})
            else: pending[fy] = pending.get(fy, 0.0) + amt; divs.append({'fy': fy, 'event': 'receivable', 'date': str(d.date()), 'amount': amt})
        for fy, pdte in pay.items():
            if pdte is not None and d == pdte and fy in pending:
                amt = pending.pop(fy); portfolio.update_cash(amt); divs.append({'fy': fy, 'event': 'paid', 'date': str(d.date()), 'amount': amt})
    return orig_sell(self, current_date, portfolio, data_handler, trading_dates, idx)
MagicSplitStrategy.generate_sell_signals = sell_wrap
curve, res = wfa.run_cpu_single_backtest(base, start_date=a.start, end_date=a.end, params_dict=params, initial_cash=1e7)
curve = curve.rename('value')
# 미수 배당(권리 확정~지급 전)은 평가액에 더한다
ev = pd.DataFrame([x for x in divs if 'event' in x])
recv = pd.Series(0.0, index=curve.index)
if len(ev):
    for fy, g in ev.groupby('fy'):
        r = g[g.event == 'receivable']; p = g[g.event == 'paid']
        if len(r):
            r0 = pd.Timestamp(r.date.iloc[0]); p0 = pd.Timestamp(p.date.iloc[0]) if len(p) else None
            m = (curve.index >= r0) & ((curve.index < p0) if p0 is not None else True)
            recv[m] += float(r.amount.iloc[0])
out = pd.DataFrame({'value_engine': curve, 'receivable': recv}); out['value'] = out.value_engine + out.receivable
out.to_csv(a.prefix + '_curve.csv', index_label='date')
pd.DataFrame(fills).to_csv(a.prefix + '_fills.csv', index=False)
pd.DataFrame(divs).to_csv(a.prefix + '_divs.csv', index=False)
pd.DataFrame(holds, columns=['date', 'ticker', 'state_key', 'order', 'amount', 'cash']).to_csv(a.prefix + '_holds.csv', index=False)
pd.DataFrame(sigrec).to_csv(a.prefix + '_addsignals.csv', index=False)
pd.DataFrame(unfilled, columns=['date', 'ticker', 'reason', 'amount', 'cash']).to_csv(a.prefix + '_unfilled.csv', index=False)
print(json.dumps({'start': a.start, 'end': a.end, 'tax': a.tax, 'div': a.div, 'hold': a.hold, 'seed': a.seed, 'n_hold_signals': len(holds), 'final': float(out.value.iloc[-1]), 'n_fills': len(fills),
                  'div_total': float(ev[ev.event.isin(['receivable']) | ((ev.event == 'paid') & (a.div == 'upper'))].amount.sum()) if len(ev) else 0.0}, default=str))
