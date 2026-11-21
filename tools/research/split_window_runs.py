
"""3월 승인 후보(분할매매)를 구간별로 다시 실행하고 일별 평가액·체결 수량·배당 사건을 남긴다.

usage: python -B tools/research/split_window_runs.py START END OUT_PREFIX [--tax period|fixed] [--div none|approx|upper]
저장소 루트(3월 실행 체크아웃)에서 MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행. 제품 코드는 바꾸지 않고 실행 중에만
체결·매도 신호 함수를 감싼다.
- --tax period: 매도세율을 결제일(T+2 거래일)의 세율로 바꾼다. fixed는 설정값(0.18%) 그대로.
- --div approx: FY y 배당을 y년 마지막 거래일 전날 종가 보유 수량(수정 수량 x 그날 adj_ratio)으로 계산해
  미수금으로 기록하고 y+1년 5월 첫 거래일에 현금으로 넣는다(그 뒤 전략이 쓸 수 있음).
  --div upper: 같은 금액을 권리 확정 다음 거래일에 바로 현금으로 넣는다(상한 근사). none은 배당 없음.
"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, yaml
from src.backtest.cpu.execution import BasicExecutionHandler
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.analysis import walk_forward_analyzer as wfa
import stage2_common as sc

ap = argparse.ArgumentParser()
ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('prefix')
ap.add_argument('--tax', choices=['period', 'fixed'], default='period')
ap.add_argument('--div', choices=['none', 'approx', 'upper'], default='none')
ap.add_argument('--manifest', default='results/wfo_run_20260324_125326/final_candidate_manifest.json')
a = ap.parse_args()
base = yaml.safe_load(open('config/config.yaml'))
params = dict(json.load(open(a.manifest))['champion_params'])
conn = sc.connect(); cal = sc.Calendar(sc.trading_dates(conn))
s, e = pd.Timestamp(a.start), pd.Timestamp(a.end)
years = list(range(s.year, e.year + 1))
rec = {cal.record_date(y): y for y in years}
after_rec = {cal.shift(r, 1): y for r, y in rec.items()}
pay = {y: cal.pay_date(y) for y in years}
fills, divs = [], []
pending = {}   # fy -> amount
orig_exec = BasicExecutionHandler.execute_order
def exec_wrap(self, ev, portfolio, dh, idx):
    t = ev['ticker']; q0 = sum(p.quantity for p in portfolio.positions.get(t, [])); c0 = float(portfolio.cash)
    if a.tax == 'period': self.sell_tax_rate = cal.sell_tax(ev['date'])
    orig_exec(self, ev, portfolio, dh, idx)
    q1 = sum(p.quantity for p in portfolio.positions.get(t, []))
    if q1 != q0:
        fills.append({'date': str(pd.Timestamp(ev['date']).date()), 'ticker': t, 'type': ev['type'], 'dq': int(q1 - q0), 'dcash': float(portfolio.cash) - c0})
BasicExecutionHandler.execute_order = exec_wrap
orig_sell = MagicSplitStrategy.generate_sell_signals
def sell_wrap(self, current_date, portfolio, data_handler, trading_dates, idx=None):
    d = pd.Timestamp(current_date)
    if a.div != 'none':
        if d in after_rec:     # 전날 종가 보유 = 권리 확정 보유
            fy = after_rec[d]; r = cal.shift(d, -1)
            held = {t: sum(p.quantity for p in ps) for t, ps in portfolio.positions.items()}
            dp = sc.dps_table(conn, held.keys(), [fy]); dp = dict(zip(dp.ticker, zip(dp.dps, dp.dps_fallback)))
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
print(json.dumps({'start': a.start, 'end': a.end, 'tax': a.tax, 'div': a.div, 'final': float(out.value.iloc[-1]), 'n_fills': len(fills),
                  'div_total': float(ev[ev.event.isin(['receivable']) | ((ev.event == 'paid') & (a.div == 'upper'))].amount.sum()) if len(ev) else 0.0}, default=str))
