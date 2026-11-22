"""1등급 후보 패널: selection_panel.py와 같은 조회·순위식으로, 교체일을 달마다 (k+1)번째 거래일로 바꿔 만든다.

usage: python -B tools/research/pool_vs_index_panel.py START END OUT.csv --offset K [--manifest ...]
3월 실행 체크아웃(wt-79051b5)을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행. DB는 읽기만 한다.
- 교체일 = 그달 (K+1)번째 거래일, 신호일 = 그 직전 거래일(K=0이면 selection_panel.py와 같은 날짜).
- END는 END_CAP(2025-12-30)을 넘을 수 없다. 자료 조회 시작은 첫 신호일보다 LOAD_LEAD_DAYS 앞이다
  (selection_panel.py는 신호일이 START 이전인 첫 달을 빠뜨려 panel_v2는 그 달을 따로 만들어 붙였다).
OUT.months.csv에 달마다 status를 남긴다(v2.1 A4, F3):
  ok(순위 행 ≥ 1) | zero_candidates(상장 목록이 있는 정상 조회의 후보 0, 엄격 히스테리시스로 비운 TIER_2_FALLBACK 포함)
  | missing_universe(NO_CANDIDATES_NO_UNIVERSE: 목록 누락) | zero_ranked(후보는 있으나 순위 행 0: 자료 결측) | lookup_failed(조회 예외)
- 종목 행 조회의 끝은 신호일이다(F3): 신호일 뒤 자료(결측 등)가 그달 후보·순위·상태에 영향을 주지 않는다.
selection_panel.py는 바꾸지 않는다. 순위식은 그 파일과 같아야 하며 smoke에서 K=0 재생성을 panel_v2와 대조한다.
"""
import argparse, json, os, sys
from functools import lru_cache
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, yaml
from src.analysis import walk_forward_analyzer as wfa
from src import main_backtest as mb
from src.data_handler import DataHandler
from src.backtest.cpu.strategy import MagicSplitStrategy
from pool_vs_index_common import cap_date, rebalance_days

LOAD_LEAD_DAYS = 40
SCORE_WEIGHTS = (0.5, 0.3, 0.2)        # selection_panel.py와 같은 값(cheap·flow·atr)
SCORE_SCALE = 10000.0
MCAP_UNIT = 1_000_000


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('out')
    ap.add_argument('--offset', type=int, required=True)
    ap.add_argument('--manifest', default='results/wfo_run_20260324_125326/final_candidate_manifest.json')
    return ap


def build_engine(a, load_start):
    base = yaml.safe_load(open('config/config.yaml'))
    man = json.load(open(a.manifest))
    sp = wfa._extract_strategy_params(dict(man['champion_params']), base['strategy_params'])
    cfg = wfa._build_cpu_backtest_config(base, start_date=load_start, end_date=a.end, initial_cash=1e7, strategy_params=sp)
    spc = cfg['strategy_params']
    mode = mb.resolve_universe_mode(spc, universe_mode=os.environ.get('MAGICSPLIT_UNIVERSE_MODE'))
    if mode != 'strict_pit': raise SystemExit('universe_mode must be strict_pit')
    pb, gate = mb.resolve_price_policy(spc)
    DataHandler._load_stock_data_cached = lru_cache(maxsize=6000)(DataHandler._load_stock_data_cached.__wrapped__)
    dh = DataHandler(db_config=cfg['database'], price_basis=pb, adjusted_price_gate_start_date=gate, universe_mode=mode, strategy_params=spc)
    kw = {k: v for k, v in spc.items() if k in mb._STRATEGY_PARAM_KEYS}
    kw.update({'backtest_start_date': load_start, 'backtest_end_date': a.end})
    return dh, MagicSplitStrategy(**kw), {'universe_mode': mode, 'price_basis': pb, 'gate': str(gate)}


def rank_row(r, t):
    atr, close = r['atr_14_ratio'], r.get('close_price')
    if not (pd.notna(atr) and pd.notna(close)) or float(atr) <= 0: return None
    mc = r.get('market_cap'); mcq = int(float(mc) // MCAP_UNIT) if pd.notna(mc) and float(mc) > 0 else 0
    cs, cc = r.get('cheap_score'), r.get('cheap_score_confidence')
    cs = float(cs) if pd.notna(cs) else 0.0; cc = float(cc) if pd.notna(cc) else 0.0
    f5 = r.get('flow5_mcap'); f5 = float(f5) if pd.notna(f5) else np.nan
    return {'ticker': t, 'signal_close': float(close), 'cheap_effective': max(min(cs, 1.0), 0.0) * max(min(cc, 1.0), 0.0),
            'flow5_mcap': f5, 'market_cap_q': mcq, 'atr_14_ratio': float(atr)}


def score(rk):
    df = pd.DataFrame(rk)
    pct = lambda c: pd.to_numeric(df[c], errors='coerce').rank(method='average', pct=True, ascending=True).fillna(0.0)
    df['flow_score'], df['atr_score'] = pct('flow5_mcap'), pct('atr_14_ratio')
    wc, wf, wa = SCORE_WEIGHTS
    df['score_q'] = ((wc * df.cheap_effective + wf * df.flow_score + wa * df.atr_score).fillna(0.0) * SCORE_SCALE).round().astype(int)
    df = df.sort_values(['score_q', 'market_cap_q', 'ticker'], ascending=[False, False, True]).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df


NO_UNIVERSE_SUFFIX = 'NO_UNIVERSE'


def month_status(source, n_raw, n_ranked):
    if str(source).endswith(NO_UNIVERSE_SUFFIX): return 'missing_universe'
    if n_ranked > 0: return 'ok'
    return 'zero_candidates' if n_raw == 0 else 'zero_ranked'


PANEL_COLUMNS = ['ticker', 'signal_close', 'cheap_effective', 'flow5_mcap', 'market_cap_q', 'atr_14_ratio', 'flow_score', 'atr_score',
                 'score_q', 'rank', 'trade_date', 'signal_date', 'tier_source', 'raw_candidates']


def month_panel(dh, strat, d, sd, win):
    """(패널 행 또는 None, 달력 행). 조회 예외는 달력에 lookup_failed로 남기고 계속한다(판정 보류 사유)."""
    try:
        return _month_panel(dh, strat, d, sd, win)
    except Exception as e:      # noqa: BLE001 - 실패 달을 기록하고 판정기에서 막는다
        print(d.date(), sd.date(), 'LOOKUP_FAILED', type(e).__name__, flush=True)
        return None, {'trade_date': d.date(), 'signal_date': sd.date(), 'status': 'lookup_failed', 'tier_source': type(e).__name__,
                      'raw_candidates': -1, 'ranked': -1}


def _month_panel(dh, strat, d, sd, win):
    codes, used = dh.get_candidates_with_tier_fallback_pit_gated(sd, min_liquidity_20d_avg_value=strat.min_liquidity_20d_avg_value,
                                                                 min_tier12_coverage_ratio=strat.min_tier12_coverage_ratio)
    if strat.strict_hysteresis_enabled and str(used).startswith('TIER_2_FALLBACK'): codes = []
    rk = []
    for t in dict.fromkeys(str(c) for c in codes):
        r = dh.get_stock_row_as_of(t, sd, win[0], sd.strftime('%Y-%m-%d'))   # 조회 끝 = 신호일(F3)
        if r is None or 'atr_14_ratio' not in r.index: continue
        x = rank_row(r, t)
        if x is not None: rk.append(x)
    print(d.date(), sd.date(), used, len(codes), len(rk), flush=True)
    cal = {'trade_date': d.date(), 'signal_date': sd.date(), 'status': month_status(used, len(codes), len(rk)), 'tier_source': used,
           'raw_candidates': len(codes), 'ranked': len(rk)}
    if not rk: return None, cal
    df = score(rk)
    df['trade_date'] = d.date(); df['signal_date'] = sd.date(); df['tier_source'] = used; df['raw_candidates'] = len(codes)
    return df, cal


def assemble(res):
    """(패널, 달력). 모든 달이 후보 0이어도 빈 패널과 달력을 남긴다(F3)."""
    parts = [m for m, _ in res if m is not None]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PANEL_COLUMNS)
    return out, pd.DataFrame([c for _, c in res])


def main():
    a = parser().parse_args()
    end = cap_date(a.end)
    if pd.Timestamp(a.start) > end: raise SystemExit('start after end')
    load_start = (pd.Timestamp(a.start) - pd.Timedelta(days=LOAD_LEAD_DAYS)).strftime('%Y-%m-%d')
    dh, strat, meta = build_engine(a, load_start)
    td = pd.DatetimeIndex([pd.Timestamp(x) for x in dh.get_trading_dates(load_start, a.end)])
    days = [(d, sd) for d, sd in rebalance_days(td, a.offset, a.start) if d <= end]
    meta.update({'offset': a.offset, 'min_liq': strat.min_liquidity_20d_avg_value, 'min_cov': strat.min_tier12_coverage_ratio,
                 'strict_hysteresis': strat.strict_hysteresis_enabled, 'load_start': load_start, 'end': str(end.date())})
    print(json.dumps(meta, default=str), flush=True)
    res = [month_panel(dh, strat, d, sd, (load_start, a.end)) for d, sd in days]
    out, months = assemble(res)
    months_path = a.out.replace('.csv', '') + '.months.csv'
    for p in (a.out, months_path):
        if os.path.exists(p): raise SystemExit(f'refuse to overwrite {p}')
    out.to_csv(a.out, index=False)
    months.to_csv(months_path, index=False)
    print('rows', len(out), 'months_queried', len(months), 'months_with_candidates', out.trade_date.nunique(), 'tickers', out.ticker.nunique())


if __name__ == '__main__':
    main()
