
"""매월 첫 거래일의 신규 진입 후보·순위를 3월 승인 후보와 같은 조회·순위식으로 기록한다(보유·재진입 제한 제외 없음).

usage: python -B tools/research/selection_panel.py START END OUT.csv [--manifest ...]
저장소 루트에서 MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행. DB는 읽기만 한다.
"""
import argparse, json, os, sys
from functools import lru_cache
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, yaml
from src.analysis import walk_forward_analyzer as wfa
from src import main_backtest as mb
from src.data_handler import DataHandler
from src.backtest.cpu.strategy import MagicSplitStrategy

ap = argparse.ArgumentParser()
ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('out')
ap.add_argument('--manifest', default='results/wfo_run_20260324_125326/final_candidate_manifest.json')
a = ap.parse_args()
base = yaml.safe_load(open('config/config.yaml'))
man = json.load(open(a.manifest))
sp = wfa._extract_strategy_params(dict(man['champion_params']), base['strategy_params'])
cfg = wfa._build_cpu_backtest_config(base, start_date=a.start, end_date=a.end, initial_cash=1e7, strategy_params=sp)
spc = cfg['strategy_params']
mode = mb.resolve_universe_mode(spc, universe_mode=os.environ.get('MAGICSPLIT_UNIVERSE_MODE'))
pb, gate = mb.resolve_price_policy(spc)
DataHandler._load_stock_data_cached = lru_cache(maxsize=6000)(DataHandler._load_stock_data_cached.__wrapped__)
dh = DataHandler(db_config=cfg['database'], price_basis=pb, adjusted_price_gate_start_date=gate, universe_mode=mode, strategy_params=spc)
kw = {k: v for k, v in spc.items() if k in mb._STRATEGY_PARAM_KEYS}
kw.update({'backtest_start_date': a.start, 'backtest_end_date': a.end})
strat = MagicSplitStrategy(**kw)
pre = (pd.Timestamp(a.start) - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
td = dh.get_trading_dates(pre, a.end)
td = [pd.Timestamp(x) for x in td]
s0 = pd.Timestamp(a.start)
firsts = []
for i, d in enumerate(td):
    if d < s0 or i == 0: continue
    if td[i-1].month != d.month or td[i-1] < s0 and not firsts:
        firsts.append((d, td[i-1]))
rows = []
meta = {'universe_mode': mode, 'price_basis': pb, 'gate': str(gate), 'min_liq': strat.min_liquidity_20d_avg_value,
        'min_cov': strat.min_tier12_coverage_ratio, 'strict_hysteresis': strat.strict_hysteresis_enabled}
print(json.dumps(meta, default=str), flush=True)
for d, sd in firsts:
    codes, used = dh.get_candidates_with_tier_fallback_pit_gated(sd, min_liquidity_20d_avg_value=strat.min_liquidity_20d_avg_value,
                                                                 min_tier12_coverage_ratio=strat.min_tier12_coverage_ratio)
    if strat.strict_hysteresis_enabled and str(used).startswith('TIER_2_FALLBACK'):
        codes = []
    rk = []
    for t in dict.fromkeys(str(c) for c in codes):
        r = dh.get_stock_row_as_of(t, sd, a.start, a.end)
        if r is None or 'atr_14_ratio' not in r.index: continue
        atr, close = r['atr_14_ratio'], r.get('close_price')
        if not (pd.notna(atr) and pd.notna(close)) or float(atr) <= 0: continue
        mc = r.get('market_cap'); mcq = int(float(mc) // 1_000_000) if pd.notna(mc) and float(mc) > 0 else 0
        cs, cc = r.get('cheap_score'), r.get('cheap_score_confidence')
        cs = float(cs) if pd.notna(cs) else 0.0; cc = float(cc) if pd.notna(cc) else 0.0
        ce = max(min(cs, 1.0), 0.0) * max(min(cc, 1.0), 0.0)
        f5 = r.get('flow5_mcap'); f5 = float(f5) if pd.notna(f5) else np.nan
        rk.append({'ticker': t, 'signal_close': float(close), 'cheap_effective': ce, 'flow5_mcap': f5, 'market_cap_q': mcq, 'atr_14_ratio': float(atr)})
    if rk:
        df = pd.DataFrame(rk)
        df['flow_score'] = pd.to_numeric(df.flow5_mcap, errors='coerce').rank(method='average', pct=True, ascending=True).fillna(0.0)
        df['atr_score'] = pd.to_numeric(df.atr_14_ratio, errors='coerce').rank(method='average', pct=True, ascending=True).fillna(0.0)
        df['score_q'] = ((0.5*df.cheap_effective + 0.3*df.flow_score + 0.2*df.atr_score).fillna(0.0) * 10000.0).round().astype(int)
        df = df.sort_values(['score_q', 'market_cap_q', 'ticker'], ascending=[False, False, True]).reset_index(drop=True)
        df['rank'] = np.arange(1, len(df)+1)
        df['trade_date'] = d.date(); df['signal_date'] = sd.date(); df['tier_source'] = used; df['raw_candidates'] = len(codes)
        rows.append(df)
    print(d.date(), sd.date(), used, len(codes), len(rk), flush=True)
out = pd.concat(rows, ignore_index=True)
out.to_csv(a.out, index=False)
print('rows', len(out), 'months', out.trade_date.nunique(), 'tickers', out.ticker.nunique())
