
"""승인 후보의 CPU 백테스트를 그대로 재실행하며 추가매수 신호별 체결·미체결 사유를 기록한다.

제품 코드는 바꾸지 않고 실행 중에만 신호 생성·체결 함수를 감싼다. 저장소 루트에서
`python -B tools/research/addbuy_signal_audit.py START END OUT.csv [--universe-mode strict_pit]`로 실행한다.
DB는 config/config.yaml의 연결로 읽기만 한다.
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, yaml
from src.backtest.cpu.strategy import MagicSplitStrategy
from src.backtest.cpu.execution import BasicExecutionHandler
from src.analysis import walk_forward_analyzer as wfa

ap = argparse.ArgumentParser()
ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('out')
ap.add_argument('--universe-mode', default='')
ap.add_argument('--manifest', default='results/wfo_run_20260324_125326/final_candidate_manifest.json')
ap.add_argument('--initial-cash', type=float, default=10000000)
a = ap.parse_args()
start, end, out, mode = a.start, a.end, a.out, a.universe_mode
if mode: os.environ['MAGICSPLIT_UNIVERSE_MODE'] = mode
base = yaml.safe_load(open('config/config.yaml'))
man = json.load(open(a.manifest))
params = dict(man['champion_params'])
records = []

orig_add = MagicSplitStrategy.generate_additional_buy_signals
def add_wrap(self, current_date, portfolio, data_handler, trading_dates, current_day_idx=None):
    sigs = orig_add(self, current_date, portfolio, data_handler, trading_dates, current_day_idx)
    sd = self._resolve_signal_date(current_date, trading_dates, current_day_idx, data_handler)
    for s in sigs:
        s['_audit'] = {'signal_date': str(pd.Timestamp(sd).date()) if sd is not None else None,
                       'target_order': int(s['position'].order),
                       'held_splits': len(portfolio.positions.get(s['ticker'], []))}
    return sigs
MagicSplitStrategy.generate_additional_buy_signals = add_wrap

orig_exec = BasicExecutionHandler.execute_order
def exec_wrap(self, order_event, portfolio, data_handler, current_day_idx):
    audit = order_event.get('_audit')
    if audit is None:
        return orig_exec(self, order_event, portfolio, data_handler, current_day_idx)
    n0 = len(portfolio.trade_history); cash0 = float(portfolio.cash)
    # diagnose before executing (same arithmetic as _execute_buy)
    reason = None
    ohlc = data_handler.get_ohlc_data_on_date(order_event['date'], order_event['ticker'], order_event['start_date'], order_event['end_date'])
    op = None if ohlc is None else ohlc.get('open_price')
    if op is None or not np.isfinite(op) or float(op) <= 0: reason = 'price_missing'
    else:
        px = self._adjust_price_up(np.float32(op))
        q = np.int32(math.floor(np.float32(order_event['investment_amount']) / px)) if px > 0 else 0
        if q <= 0: reason = 'zero_quantity'
        else:
            cost = np.float32(px) * np.float32(q); tc = cost + np.floor(cost * np.float32(self.buy_commission_rate))
            if cash0 < tc: reason = 'cash_shortfall'
    orig_exec(self, order_event, portfolio, data_handler, current_day_idx)
    filled = len(portfolio.trade_history) > n0
    records.append({'ticker': order_event['ticker'], 'order_date': str(pd.Timestamp(order_event['date']).date()),
        **audit, 'investment_amount': float(order_event['investment_amount']), 'cash_before': cash0,
        'filled': filled, 'unfilled_reason': None if filled else (reason or 'other')})
BasicExecutionHandler.execute_order = exec_wrap

curve, res = wfa.run_cpu_single_backtest(base, start_date=start, end_date=end, params_dict=params, initial_cash=a.initial_cash)
pd.DataFrame(records).to_csv(out, index=False)
th = res.get('trade_history') if isinstance(res, dict) else None
print(json.dumps({'metrics': res.get('metrics'), 'keys': sorted(res.keys()), 'n_records': len(records),
   'final_value': float(curve.iloc[-1]) if curve is not None and len(curve) else None, 'mode': mode or base['strategy_params'].get('universe_mode')}, default=str))
