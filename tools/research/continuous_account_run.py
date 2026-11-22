"""연속 계좌 진단(a): split_window_runs.py를 그대로 실행하면서 일별 상태·마지막 날 보유·엔진 상태를 남긴다.

usage: python -B tools/research/continuous_account_run.py START END OUT_PREFIX [split_window_runs.py 옵션...]
3월 실행 체크아웃을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행한다. 앞의 세 인자는 위치 인자로 준다.
split_window_runs.py의 산출물(_curve·_fills·_divs 등)은 그대로 나오고 아래 세 파일이 추가된다(읽기만 하는 감싸기).
- OUT_PREFIX_state.csv: date, engine_value(미수 배당 제외), cash, n_tickers, n_lots, n_full(차수 >= max_splits_limit), max_lots
- OUT_PREFIX_endpos.csv: 마지막 거래일 종가 평가 직전 보유(ticker, quantity, lots, price_used = 엔진이 평가에 쓴 가격)
- OUT_PREFIX_meta.json: 엔진 반환값의 success·degraded_run·universe_mode·후보 조회 오류 수·환경 변수
"""
import json, os, runpy, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src.backtest.cpu.portfolio import Portfolio
from src.analysis import walk_forward_analyzer as wfa
import stage2_common as sc

SWR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_window_runs.py')
DEFAULT_MANIFEST = 'results/wfo_run_20260324_125326/final_candidate_manifest.json'


def split_limit(argv):
    path = argv[argv.index('--manifest') + 1] if '--manifest' in argv else DEFAULT_MANIFEST
    return int(json.load(open(path))['champion_params']['max_splits_limit'])


def last_trading_day(end):
    dates = sc.trading_dates(sc.connect())
    return dates[dates <= pd.Timestamp(end)][-1]


def lot_state(positions, limit):
    lots = [len(ps) for ps in positions.values() if ps]
    return {'n_tickers': len(lots), 'n_lots': sum(lots), 'n_full': sum(n >= limit for n in lots), 'max_lots': max(lots, default=0)}


def end_positions(pf, date, dh, price_map):
    pm = pf._resolve_price_map(date, dh, price_map=price_map)
    rows = []
    for t, ps in pf.positions.items():
        if not ps: continue
        px = pm.get(t) if pm is not None else dh.get_latest_price(date, t, pf.start_date, pf.end_date)
        rows.append({'ticker': t, 'quantity': int(sum(p.quantity for p in ps)), 'lots': len(ps),
                     'price_used': float(px) if px is not None else float('nan')})
    return rows


def engine_meta(res):
    summary = res.get('candidate_lookup_summary') or {}
    return {'success': bool(res.get('success')), 'degraded_run': bool(res.get('degraded_run')), 'universe_mode': res.get('universe_mode'),
            'promotion_block_reasons': list(res.get('promotion_block_reasons') or []),
            'candidate_lookup_error_count': int(summary.get('error_count', 0) or 0),
            'source_lookup_error_days': int((res.get('run_metrics') or {}).get('source_lookup_error_days', 0) or 0),
            'env_universe_mode': os.environ.get('MAGICSPLIT_UNIVERSE_MODE')}


def install_hooks(limit, last_day, rec):
    snap0, total0, run0 = Portfolio.record_daily_snapshot, Portfolio.get_total_value, wfa.run_cpu_single_backtest

    def snap(self, date, total_value):
        rec['state'].append({'date': str(pd.Timestamp(date).date()), 'engine_value': float(total_value), 'cash': float(self.cash),
                             **lot_state(self.positions, limit)})
        return snap0(self, date, total_value)

    def total(self, evaluation_date, data_handler, price_map=None):
        value = total0(self, evaluation_date, data_handler, price_map=price_map)
        if pd.Timestamp(evaluation_date) == last_day:
            rec['endpos'] = end_positions(self, evaluation_date, data_handler, price_map)
        return value

    def run(*args, **kwargs):
        curve, res = run0(*args, **kwargs); rec['meta'] = engine_meta(res); return curve, res
    Portfolio.record_daily_snapshot, Portfolio.get_total_value, wfa.run_cpu_single_backtest = snap, total, run


def main():
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0].startswith('-'):
        sys.exit(__doc__)
    rec = {'state': [], 'endpos': [], 'meta': None}
    install_hooks(split_limit(argv), last_trading_day(argv[1]), rec)
    sys.argv = [SWR] + argv
    runpy.run_path(SWR, run_name='__main__')
    pd.DataFrame(rec['state']).to_csv(argv[2] + '_state.csv', index=False)
    pd.DataFrame(rec['endpos'], columns=['ticker', 'quantity', 'lots', 'price_used']).to_csv(argv[2] + '_endpos.csv', index=False)
    json.dump({**(rec['meta'] or {'success': False}), 'argv': argv}, open(argv[2] + '_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
