"""종료일 절단 검사(v2 5절, v2.1 B, F4, G1·G2): 패널부터 상한을 CUT로 바꿔 독립적으로 다시 만든 실행(SIM_CUT)과
END_CAP 실행(SIM_FULL)을 대조한다. 기대값은 비교 대상 밖에서 정한다:
- 기대 k·기준일·CUT은 실행기가 넘긴다. 기대 날짜 목록은 긴 실행의 거래일 달력에서 [기준일, CUT]으로 만든다.
- 열 목록은 pool_vs_index_common의 고정 목록(곡선 156열, ETF 35열, 정수 1열, 표별 열, 패널 열)과 같아야 한다.
- 필수 파일이 한쪽이라도 없으면 실패. 양쪽 모두 0건인 표도 열 구조를 확인한다.
- 종목별 체결(모든 계좌·조합과 정수 계좌), 주문, 퇴출, 권리(수량·근거만), 표시, 월 상태, 관측 창이 CUT 전에 끝난 DPS 품질표, 패널 행을 비교한다.

- H1: 두 실행이 실제로 다른 조회 상한으로 만들어졌는지 확인한다. 실행기가 긴 실행의 기대 종료일(--full-end)·조회 상한(--full-cap)과
  절단 상한(--cut)을 따로 넘긴다. 양쪽 sim_checks의 cap·bundle_checks.cap·end_date, 패널 파일 SHA(= bundle_checks.panel_sha256),
  ETF 입력 폴더와 그 etf_cap_manifest.json의 cap을 대조한다. 같은 묶음·같은 패널·같은 ETF 입력이면 실패. 양쪽 calendar.csv 필수.
- H2: 비교 전에 각 실행 안에서 주문과 체결의 일관성을 먼저 검사한다(계좌·조합·날짜별 체결 금액 합 = 주문 금액, 정수 계좌 포함).

usage: python -I -B tools/research/pool_vs_index_truncation_check.py SIM_FULL SIM_CUT OUT.json --cut D --full-end D --full-cap D --k K --start S
       --panels FULL.csv CUT.csv
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pool_vs_index_common as pc

TOL_WON = 1e-6
PANEL_COLUMNS = ['ticker', 'signal_close', 'cheap_effective', 'flow5_mcap', 'market_cap_q', 'atr_14_ratio', 'flow_score', 'atr_score',
                 'score_q', 'rank', 'trade_date', 'signal_date', 'tier_source', 'raw_candidates']
PANEL_MONTH_COLUMNS = ['trade_date', 'signal_date', 'status', 'tier_source', 'raw_candidates', 'ranked']


def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('full'); ap.add_argument('cut_dir'); ap.add_argument('out')
    ap.add_argument('--cut', required=True); ap.add_argument('--k', type=int, required=True); ap.add_argument('--start', required=True)
    ap.add_argument('--full-end', required=True); ap.add_argument('--full-cap', required=True)
    ap.add_argument('--panels', nargs=2, required=True)
    return ap


def read(path):
    if path.endswith('.parquet'): return pd.read_parquet(path)
    try:
        return pd.read_csv(path, dtype={'ticker': str})
    except pd.errors.EmptyDataError:
        return None                       # 열 머리조차 없음 → 구조 실패


def expected_dates(full, start, cut):
    cal = pd.read_csv(os.path.join(full, 'calendar.csv'), parse_dates=['date'])
    priced = pd.DatetimeIndex(cal.date[cal.priced])
    base = priced[priced < pd.Timestamp(start)][-1]
    return priced[(priced >= base) & (priced <= pd.Timestamp(cut))]


def values_equal(a, b):
    num = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c]) and not pd.api.types.is_bool_dtype(b[c])]
    other = [c for c in b.columns if c not in num]
    nan_ok = bool((a[num].isna().values == b[num].isna().values).all()) if num else True
    d = float(np.nanmax(np.abs(a[num].values.astype(float) - b[num].values.astype(float)))) if num and len(b) else 0.0
    same = all(a[c].astype(str).equals(b[c].astype(str)) for c in other)
    return nan_ok and same and d <= TOL_WON, d


def frame_diff(full, cut, name, dates):
    pa, pb = os.path.join(full, name), os.path.join(cut, name)
    if not (os.path.exists(pa) and os.path.exists(pb)): return {'name': name, 'ok': False, 'reason': 'required file missing'}
    a, b = read(pa), read(pb); cols = pc.frame_columns(name)
    for side, df in (('full', a), ('cut', b)):
        if list(df.columns) != cols: return {'name': name, 'ok': False, 'reason': f'{side} columns differ from fixed list'}
    a = a[(a.index >= dates[0]) & (a.index <= dates[-1])]
    for side, df in (('full', a), ('cut', b)):
        if len(df) != len(dates) or not (df.index == dates).all(): return {'name': name, 'ok': False, 'reason': f'{side} dates differ from calendar'}
    ok, d = values_equal(a, b)
    return {'name': name, 'ok': ok, 'rows': len(b), 'cols': len(cols), 'max_abs_diff': d}


def table_diff(full, cut, name, cut_day):
    pa, pb = os.path.join(full, name), os.path.join(cut, name)
    if not (os.path.exists(pa) and os.path.exists(pb)): return {'name': name, 'ok': False, 'reason': 'required file missing'}
    a, b = read(pa), read(pb); cols = pc.TABLE_SCHEMAS[name]; keys = pc.TABLE_KEYS[name]; dcol = pc.TABLE_DATE_COL[name]
    for side, df in (('full', a), ('cut', b)):
        if df is None or list(df.columns) != cols: return {'name': name, 'ok': False, 'reason': f'{side} columns differ from fixed schema'}
    a = a[a[dcol].astype(str) <= cut_day]
    if (b[dcol].astype(str) > cut_day).any(): return {'name': name, 'ok': False, 'reason': 'cut side has rows after CUT'}
    a, b = [x.sort_values(keys, kind='stable').reset_index(drop=True) for x in (a, b)]
    if len(a) != len(b) or not a[keys].astype(str).equals(b[keys].astype(str)): return {'name': name, 'ok': False, 'rows_full': len(a), 'rows_cut': len(b)}
    ok, d = values_equal(a, b)
    return {'name': name, 'ok': ok, 'rows': len(b), 'max_abs_diff': d}


def panel_diff(pf, pcut, cut_day):
    a, b = pd.read_csv(pf, dtype={'ticker': str}), pd.read_csv(pcut, dtype={'ticker': str})
    for side, df in (('full', a), ('cut', b)):
        if list(df.columns) != PANEL_COLUMNS: return {'name': 'panel', 'ok': False, 'reason': f'{side} panel columns differ from fixed list'}
    a = a[a.trade_date <= cut_day]
    keys = ['trade_date', 'ticker']
    a, b = [x.astype(str).sort_values(keys).reset_index(drop=True) for x in (a, b)]
    ma, mb = [pd.read_csv(p.replace('.csv', '') + '.months.csv').astype(str) for p in (pf, pcut)]
    for side, df in (('full', ma), ('cut', mb)):
        if list(df.columns) != PANEL_MONTH_COLUMNS: return {'name': 'panel', 'ok': False, 'reason': f'{side} months columns differ'}
    ma = ma[ma.trade_date <= cut_day].reset_index(drop=True)
    return {'name': 'panel', 'rows': len(b), 'ok': bool(a.equals(b) and ma.equals(mb.reset_index(drop=True)))}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def etf_cap_of(ck):
    m = os.path.join(ck.get('etf_dir') or '', 'etf_cap_manifest.json')
    return json.load(open(m)).get('cap') if os.path.exists(m) else None


def provenance_checks(a, ck, dates):
    """H1: 각 쪽이 주장하는 상한·종료일·패널·ETF 입력이 실행기 기대와 맞고, 두 쪽이 서로 다른 입력인지."""
    d = lambda x: str(pd.Timestamp(x).date())
    full, cut = ck['full'], ck['cut']
    cal = pd.read_csv(os.path.join(a.full, 'calendar.csv'), parse_dates=['date']); priced = pd.DatetimeIndex(cal.date[cal.priced])
    full_end = d(priced[priced <= pd.Timestamp(a.full_end)][-1])          # 긴 실행의 기대 종료일 = full_end 이하 마지막 거래일
    exp = {'full': (d(a.full_cap), full_end, a.panels[0]), 'cut': (d(a.cut), d(a.cut), a.panels[1])}
    out = {}
    for side, c in (('full', full), ('cut', cut)):
        cap, end, panel = exp[side]
        out[f'{side}_k'] = c.get('k') == a.k
        out[f'{side}_cap'] = c.get('cap') == cap and c.get('bundle_checks', {}).get('cap') == cap
        out[f'{side}_end'] = c.get('end_date') == end and pd.Timestamp(c.get('bundle_checks', {}).get('end', '1900-01-01')) >= pd.Timestamp(end)
        out[f'{side}_panel_sha'] = c.get('bundle_checks', {}).get('panel_sha256') == sha256(panel)
        out[f'{side}_etf_cap'] = etf_cap_of(c) == cap
        out[f'{side}_calendar'] = os.path.exists(os.path.join(a.full if side == 'full' else a.cut_dir, 'calendar.csv'))
    out['base_same'] = full.get('base_date') == cut.get('base_date') == str(dates[0].date())
    out['cut_on_calendar'] = dates[-1] == pd.Timestamp(d(a.cut))
    out['different_bundles'] = full.get('bundle_checks', {}).get('cap') != cut.get('bundle_checks', {}).get('cap')
    out['different_panels'] = sha256(a.panels[0]) != sha256(a.panels[1]) and os.path.realpath(a.panels[0]) != os.path.realpath(a.panels[1])
    out['different_etf_inputs'] = os.path.realpath(full.get('etf_dir') or 'x') != os.path.realpath(cut.get('etf_dir') or 'y')
    out['different_dirs'] = os.path.realpath(a.full) != os.path.realpath(a.cut_dir)
    return out


def internal_consistency(sim_dir):
    """H2: 한 실행 안에서 주문 금액과 체결 금액(소수 주 156조합, 정수 계좌)이 맞는지."""
    res = {}
    for orders_n, fills_n, keys in (('orders.csv', 'fills.parquet', pc.KEY4 + ['date']), ('integer_orders.csv', 'integer_fills.csv', ['date'])):
        paths = [os.path.join(sim_dir, n) for n in (orders_n, fills_n)]
        if not all(os.path.exists(p) for p in paths): res[orders_n] = {'ok': False, 'reason': 'missing'}; continue
        o, f = read(paths[0]), read(paths[1])
        if o is None or f is None: res[orders_n] = {'ok': False, 'reason': 'unreadable'}; continue
        bad, n = pc.fills_consistency(o, f, keys)
        res[orders_n] = {'ok': n == 0, 'n_mismatch': n, 'examples': bad[:5]}
    return res


def run(a):
    for side, path in (('full', a.full), ('cut', a.cut_dir)):
        if not os.path.exists(os.path.join(path, 'calendar.csv')): return {'pass': False, 'reason': f'{side} calendar.csv missing (H1)'}
    dates = expected_dates(a.full, a.start, a.cut); cut_day = str(pd.Timestamp(a.cut).date())
    ck = {s: json.load(open(os.path.join(d, 'sim_checks.json'))) for s, d in (('full', a.full), ('cut', a.cut_dir))}
    meta = provenance_checks(a, ck, dates)
    inner = {s: internal_consistency(d) for s, d in (('full', a.full), ('cut', a.cut_dir))}
    inner_ok = all(v['ok'] for s in inner.values() for v in s.values())
    parts = [frame_diff(a.full, a.cut_dir, n, dates) for n in pc.FRAME_FILES]
    parts += [table_diff(a.full, a.cut_dir, n, cut_day) for n in pc.TABLE_SCHEMAS]
    parts.append(panel_diff(a.panels[0], a.panels[1], cut_day))
    ok = all(meta.values()) and inner_ok and all(p['ok'] for p in parts)
    return {'pass': bool(ok), 'tolerance_won': TOL_WON, 'expected': {'k': a.k, 'cut': cut_day, 'full_end': a.full_end, 'full_cap': a.full_cap,
            'base': str(dates[0].date()), 'n_dates': len(dates)}, 'provenance': meta, 'internal_consistency': inner, 'parts': parts}


def main():
    a = parser().parse_args()
    try:
        res = run(a)
    except Exception as e:      # noqa: BLE001 - 검사기 예외도 실패 결과로 남긴다(R3)
        res = {'pass': False, 'reason': f'checker error: {type(e).__name__}: {e}'}
    if os.path.exists(a.out): raise SystemExit(f'refuse to overwrite {a.out}')
    json.dump(res, open(a.out, 'w'), indent=1, default=str)
    print(json.dumps({'pass': res['pass'], 'failed': [p['name'] for p in res.get('parts', []) if not p['ok']],
                      'provenance_failed': [k for k, v in res.get('provenance', {}).items() if not v],
                      'internal_failed': [f'{s}:{n}' for s, v in res.get('internal_consistency', {}).items() for n, x in v.items() if not x['ok']],
                      'reason': res.get('reason')}, indent=1))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
