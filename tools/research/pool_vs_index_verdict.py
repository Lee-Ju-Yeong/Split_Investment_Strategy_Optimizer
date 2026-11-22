"""연구 (b) 판정: 두 k의 모의 실행 결과로 질문 1(서술)·질문 2(라벨)·질문 3(분해)·ETF 참고선을 낸다.

usage: python -B tools/research/pool_vs_index_verdict.py OUT_DIR --sim k0=SIM_DIR --sim k5=SIM_DIR --mode formal|smoke
       [--check panel=FILE.json] [--check truncation_k0=FILE.json] [--check truncation_k5=FILE.json] [--check stage2_s20=FILE.json]
       [--check db=FILE.json]
DB를 읽지 않는다. 판정 기준은 DESIGN-v2.md 4절 + DESIGN-v2.1.md A5·B + 구현 리뷰 F1·F5·F9·F10.
- 조합 라벨: 주기간(2014~2023) P1−M 비용 후 연복리 차(%p, 반올림 전) d와 다섯 구간 차 s_j.
  pool_edge: d ≥ +1 그리고 s_j > +1e-10 인 구간 ≥ 3. pool_drag: d ≤ −1 그리고 s_j < −1e-10 인 구간 ≥ 3. 나머지 no_clear_difference.
- 최종: 12조합(배당 3 × k 2 × 폐지 2) 모두 값이 유효하고 필수 검산이 모두 통과하고 라벨이 하나로 같을 때만 그 라벨. 아니면 undetermined + 사유.
- 구간 경계는 곡선이 아니라 거래일 달력에서 정한다. 곡선에 그 날짜가 없거나 값이 유한하지 않으면 그 구간은 무효(F10).
- smoke 모드는 구조 확인용이다. 기간이 짧아 구간 값이 없으므로 undetermined가 정상이고, 수치는 해석하지 않는다.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pool_vs_index_common as pc

PERIODS = {**{f'seg{y0}': (y0, y1) for y0, y1 in pc.SEGMENTS}, 'main': pc.MAIN_PERIOD, 'ext': pc.EXT_PERIOD, 'full': pc.FULL_PERIOD}
COMBO_DIVS, COMBO_DELISTS = ('none', 'observed', 'late'), ('last', 'zero')
COMBO_KS = {'k0': 0, 'k5': 5}
MAIN = ('observed', 'last')
Q1_PERIODS = ['main'] + [f'seg{y0}' for y0, _ in pc.SEGMENTS] + ['ext']
ARITH_TOL_PP = 1e-9
CONSERVATION_TOL = 1e-4          # 원
MDD_GAP_PP, ORIG_EXCESS_PP = -5.0, 1.0
EXPECTED_MONTHS = 144
KOSDAQ_REF_FROM = 2016
REQUIRED_CHECKS = ('panel', 'truncation_k0', 'truncation_k5', 'stage2_s20')
FORMAL_ONLY_CHECKS = ('db',)


def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir')
    ap.add_argument('--sim', action='append', required=True); ap.add_argument('--check', action='append', default=[])
    ap.add_argument('--mode', choices=('formal', 'smoke'), required=True)
    return ap


def load_sim(d):
    cal = pd.read_csv(os.path.join(d, 'calendar.csv'), parse_dates=['date'])
    return {'curves': pd.read_parquet(os.path.join(d, 'curves.parquet')), 'opens': pd.read_parquet(os.path.join(d, 'open_values.parquet')),
            'etf': pd.read_parquet(os.path.join(d, 'etf_curves.parquet')), 'checks': json.load(open(os.path.join(d, 'sim_checks.json'))),
            'full_cal': pd.DatetimeIndex(cal.date), 'priced_cal': pd.DatetimeIndex(cal.date[cal.priced])}


# ---------------- 지표 ----------------

def expected_bounds(sim, years, curve):
    """달력 기준 구간 경계: 기준 = 구간 첫해 직전 거래일(계좌 기준일보다 앞이면 기준일), 끝 = 마지막 해 마지막 거래일."""
    cal = sim['priced_cal']
    if not len(cal[cal.year == years[1]]) or curve.index[0] > pd.Timestamp(f'{years[0]}-01-01'): return None
    return pc.segment_bounds(cal, years, curve.index[0])


def period_row(curve, sim, years, dd_mask=None):
    bad = {'valid': False, 'cagr': np.nan, 'mdd': np.nan, 'calmar': np.nan}
    b = expected_bounds(sim, years, curve)
    if b is None: return bad
    base, end = b
    if base not in curve.index or end not in curve.index or end > curve.index[-1]: return bad
    if not (np.isfinite(curve.loc[base]) and np.isfinite(curve.loc[end])): return bad
    c = curve[(curve.index >= base) & (curve.index <= end)]
    if dd_mask is None and not np.isfinite(c.values).all(): return bad       # R3: 주식 계좌 내부 평가액 결측은 무효(ETF 무효일은 dd_mask로 제외)
    span = sim['priced_cal'][(sim['priced_cal'] >= base) & (sim['priced_cal'] <= end)]
    if len(c) != len(span) or not (c.index == span).all(): return bad
    return {'valid': True, **pc.period_metrics(curve, base, end, dd_mask)}


def metrics_table(sims):
    rows = []
    for k, s in sims.items():
        for col in s['curves'].columns:
            a, div, dl, cm = col.split('|')
            for p, yrs in PERIODS.items():
                rows.append({'k': k, 'account': a, 'div': div, 'delist': dl, 'cost': cm, 'period': p, **period_row(s['curves'][col], s, yrs)})
        mask = s['etf'][f'{pc.E0}|ok']
        for cm in ('net', 'gross'):      # 스타일 ETF는 style_rows에서만(5구간 승수 제외)
            for p, yrs in PERIODS.items():
                r = period_row(s['etf'][f'{pc.E0}|{cm}'], s, yrs, mask)
                rows += [{'k': k, 'account': pc.E0, 'div': div, 'delist': dl, 'cost': cm, 'period': p, **r}
                         for div in COMBO_DIVS for dl in COMBO_DELISTS]
    return pd.DataFrame(rows)


def pick(mt, k, a, div, dl, cm, p):
    r = mt[(mt.k == k) & (mt.account == a) & (mt['div'] == div) & (mt.delist == dl) & (mt.cost == cm) & (mt.period == p)]
    return r.iloc[0] if len(r) == 1 else None


# ---------------- 질문 2 ----------------

def label(main_diff, seg_diffs):
    pos = sum(d > pc.TIE_EPS_PP for d in seg_diffs); neg = sum(d < -pc.TIE_EPS_PP for d in seg_diffs)
    if main_diff >= pc.EDGE_PP and pos >= pc.MIN_SAME_SIGN: return 'pool_edge', pos, neg
    if main_diff <= -pc.EDGE_PP and neg >= pc.MIN_SAME_SIGN: return 'pool_drag', pos, neg
    return 'no_clear_difference', pos, neg


def combo_diff(mt, k, div, dl, p):
    a, b = pick(mt, k, 'P1', div, dl, 'net', p), pick(mt, k, 'M', div, dl, 'net', p)
    if a is None or b is None or not (a.valid and b.valid) or a.base != b.base or a.end != b.end or not np.isfinite(a.cagr - b.cagr): return None
    return float(a.cagr - b.cagr)


def combo_rows(mt):
    rows = []
    for k in COMBO_KS:
        for div in COMBO_DIVS:
            for dl in COMBO_DELISTS:
                main = combo_diff(mt, k, div, dl, 'main'); segs = [combo_diff(mt, k, div, dl, f'seg{y0}') for y0, _ in pc.SEGMENTS]
                valid = main is not None and all(s is not None for s in segs)
                lab, pos, neg = label(main, segs) if valid else ('invalid', None, None)
                rows.append({'k': k, 'div': div, 'delist': dl, 'valid': valid, 'main_diff_pp': main, **{f'seg{y0}_diff_pp': s for (y0, _), s in zip(pc.SEGMENTS, segs)},
                             'n_pos': pos, 'n_neg': neg, 'label': lab})
    return pd.DataFrame(rows)


def final_label(combos, failed_checks):
    reasons = list(failed_checks)
    keys = combos[['k', 'div', 'delist']].apply(tuple, axis=1)
    if len(combos) != 12 or keys.duplicated().any(): reasons.append(f'combos: expected 12 unique, got {keys.nunique()}/{len(combos)}')
    if not combos.valid.all(): reasons.append(f'invalid combos: {int((~combos.valid).sum())}')
    labs = sorted(set(combos.label[combos.valid]))
    if len(labs) > 1: reasons.append(f'labels differ across combos: {labs}')
    return ('undetermined', reasons) if reasons or not labs else (labs[0], [])


# ---------------- 질문 1·3 ----------------

def q1_rows(mt):
    rows = []
    for k in COMBO_KS:
        for p in Q1_PERIODS:
            a, e = pick(mt, k, 'P1', *MAIN, 'net', p), pick(mt, k, pc.E0, *MAIN, 'net', p)
            if a is None or e is None or not (a.valid and e.valid) or a.base != e.base or a.end != e.end:
                rows.append({'k': k, 'period': p, 'valid': False}); continue
            ex, gap = a.cagr - e.cagr, a.mdd - e.mdd
            mdd_ok = gap >= MDD_GAP_PP
            calmar_ok = a.cagr > 0 and e.cagr > 0 and a.calmar >= e.calmar
            rows.append({'k': k, 'period': p, 'valid': True, 'base': a.base, 'end': a.end, 'cagr_excess_pp': ex, 'mdd_gap_pp': gap,
                         'original_pass': bool(ex >= ORIG_EXCESS_PP and mdd_ok), 'calmar_pass': bool(ex >= ORIG_EXCESS_PP and (mdd_ok or calmar_ok))})
    return pd.DataFrame(rows)


def q1_summary(q1):
    out = []
    for k, g in q1.groupby('k'):
        segs = g[g.period.str.startswith('seg') & g.valid]; main = g[(g.period == 'main') & g.valid]
        out.append({'k': k, 'valid_rows': int(g.valid.sum()), 'original_seg_passes': int(segs.original_pass.sum()) if len(segs) else None,
                    'calmar_seg_passes': int(segs.calmar_pass.sum()) if len(segs) else None,
                    'original_main_pass': bool(main.original_pass.iloc[0]) if len(main) else None,
                    'calmar_main_pass': bool(main.calmar_pass.iloc[0]) if len(main) else None})
    return pd.DataFrame(out)


def q3_rows(mt):
    """세 항과 산술 확인(F9). 네 계좌의 k·조합·기간·기준일·말일이 같은지 검사(inputs_consistent)."""
    rows = []
    for k in COMBO_KS:
        for cm in ('net', 'gross'):
            g = {a: pick(mt, k, a, *MAIN, cm, 'main') for a in ('M', 'P1', 'S20e', pc.E0)}
            if any(v is None or not v.valid for v in g.values()): rows.append({'k': k, 'cost': cm, 'valid': False}); continue
            same = len({(v.base, v.end, v.k, v['div'], v.delist, v.cost, v.period) for v in g.values()}) == 1
            c = {a: v.cagr for a, v in g.items()}
            t1, t2, t3 = c['M'] - c[pc.E0], c['P1'] - c['M'], c['S20e'] - c['P1']
            lhs = c['S20e'] - c[pc.E0]
            rows.append({'k': k, 'cost': cm, 'valid': True, 'inputs_consistent': same, 'base': g['P1'].base, 'end': g['P1'].end,
                         'M_minus_E0': t1, 'P1_minus_M': t2, 'S20e_minus_P1': t3, 'P1_minus_E0': t1 + t2, 'S20e_minus_E0': lhs,
                         'arithmetic_check_residual': lhs - (t1 + t2 + t3)})
    return pd.DataFrame(rows)


# ---------------- ETF 참고선 ----------------

def month_end_returns(v, ok):
    """인접한 두 월말 종가가 모두 유효할 때만 그달 수익률을 쓴다(v2.1 B)."""
    me = v.groupby([v.index.year, v.index.month]).tail(1)
    okm = ok.reindex(me.index).fillna(False).astype(bool) & np.isfinite(me)
    return me / me.shift(1) - 1, okm & okm.shift(1, fill_value=False)


def reference_line(sim, internal, code, years):
    v_e = sim['etf'].get(f'E_{code}|net'); entry = (sim['checks'].get('etf_entry') or {}).get(code)
    if v_e is None or entry is None: return {'internal': internal, 'etf': code, 'status': 'withheld', 'reason': 'etf missing'}
    v_in = sim['curves'][f'{internal}|{MAIN[0]}|{MAIN[1]}|net']
    ok_e = sim['etf'][f'E_{code}|ok'] & (sim['etf'].index >= pd.Timestamp(entry['date']))
    ri, _ = month_end_returns(v_in, pd.Series(True, index=v_in.index)); re_, valid = month_end_returns(v_e, ok_e)
    span = (re_.index.year >= years[0]) & (re_.index.year <= years[1]); sel = valid & span
    full = [y for y in range(years[0], years[1] + 1) if int(sel[re_.index.year == y].sum()) == 12]
    yd = [abs((np.prod(1 + ri[ri.index.year == y]) - np.prod(1 + re_[re_.index.year == y])) * 100) for y in full]
    n = int(sel.sum()); corr = float(np.corrcoef(ri[sel], re_[sel])[0, 1]) if n > 2 else np.nan
    row = {'internal': internal, 'etf': code, 'years': f'{years[0]}-{years[1]}', 'months_used': n, 'months_excluded': int(span.sum()) - n,
           'full_years': len(full), 'years_excluded': years[1] - years[0] + 1 - len(full), 'corr': corr,
           'mean_abs_annual_diff_pp': float(np.mean(yd)) if yd else np.nan, 'price_label': 'naver_adjusted_price_approximation'}
    enough = n >= pc.REF_MIN_MONTHS and len(full) >= pc.REF_MIN_YEARS
    similar = corr >= pc.REF_CORR_MIN and row['mean_abs_annual_diff_pp'] <= pc.REF_ANNUAL_DIFF_MAX_PP
    row['status'] = ('similar' if similar else 'not_similar') if enough else 'withheld'
    return row


def reference_rows(sims):
    specs = (('U_KOSPI_CW200', pc.KODEX200, pc.MAIN_PERIOD), ('U_KOSPI_CW200', pc.KODEX200, pc.EXT_PERIOD),
             ('U_KOSDAQ_CW150', '229200', (KOSDAQ_REF_FROM, pc.MAIN_PERIOD[1])), ('U_KOSDAQ_CW150', '229200', pc.EXT_PERIOD))
    return pd.DataFrame([{'k': k, **reference_line(s, *spec)} for k, s in sims.items() for spec in specs])


def e0_value_at_open(s, day):
    """E0 계좌의 day 시가 평가액 = 보유 주수 × 그날 계산용 시가(진입 전이면 현금, 무효면 NaN)."""
    ent = s['checks']['etf_entry'].get(pc.KODEX200)
    if ent is None: return pc.CASH0          # 끝까지 미진입이면 현금 상태
    if day <= pd.Timestamp(ent['date']): return pc.CASH0      # R2: E0 첫 매수일은 주문 전 금액
    return ent['shares'] * s['etf'][f'{pc.E0}|open'].loc[day]


def style_window(s, code, entry, end):
    """ETF 실제 유효 진입일 시가 평가액에서 P1·M·S20e·E0를 자른다(교체일이 아니어도, F5). 끝값은 계산용 가격만."""
    e = pd.Timestamp(entry['date']); rows = []
    for a in ('P1', 'M', 'S20e'):
        key = f'{a}|{MAIN[0]}|{MAIN[1]}|net'
        rows.append((a, pc.annualized(float(s['opens'][key].loc[e]), float(s['curves'][key].loc[end]), e, end)))
    rows.append((f'E_{code}', pc.annualized(pc.CASH0, s['etf'][f'E_{code}|net'].loc[end], e, end)))
    if code != pc.KODEX200:
        rows.append((pc.E0, pc.annualized(e0_value_at_open(s, e), s['etf'][f'{pc.E0}|net'].loc[end], e, end)))
    return [{'etf': code, 'entry': entry['date'], 'end': str(end.date()), 'account': a, 'cagr': c, 'valid': bool(np.isfinite(c)),
             'basis': 'cut from continuous accounts at ETF entry-day open value', 'price_label': 'naver_adjusted_price_approximation'} for a, c in rows]


def style_rows(sims):
    rows = []
    for k, s in sims.items():
        for code, ent in (s['checks'].get('etf_entry') or {}).items():
            if ent is None: rows.append({'k': k, 'etf': code, 'valid': False, 'basis': 'no valid entry'}); continue
            for end_y in (pc.MAIN_PERIOD[1], pc.EXT_PERIOD[1]):
                end = s['priced_cal'][s['priced_cal'] <= pd.Timestamp(f'{end_y}-12-31')][-1]
                if end > pd.Timestamp(ent['date']) and s['curves'].index[-1] >= end:
                    rows += [{'k': k, **r} for r in style_window(s, code, ent, end)]
    return pd.DataFrame(rows)


# ---------------- 검산 ----------------

def sim_failures(k, s, mode):
    c = s['checks']; b = c['bundle_checks']; f = []
    if c.get('k') != COMBO_KS.get(k): f.append(f'{k}: sim offset {c.get("k")} does not match key')
    if c['skipped_months']: f.append(f'{k}: unusable months {c["skipped_months"][:5]}')
    if b.get('month_problems'): f.append(f'{k}: month problems {b["month_problems"][:5]}')
    if not c['weights_all_ok']: f.append(f'{k}: target weight sums')
    if c['min_cash_after_orders'] is not None and c['min_cash_after_orders'] < 0: f.append(f'{k}: negative cash')
    if c['max_abs_conservation_err'] is not None and c['max_abs_conservation_err'] > CONSERVATION_TOL: f.append(f'{k}: order conservation')
    if not c['base_value_all_cash0']: f.append(f'{k}: base value')
    if c.get('held_without_book_days_max', 0): f.append(f'{k}: held without any valid close')
    for key in ('p1_not_in_U', 'p1_not_in_U0', 'snapshot_after_signal', 'tier_row_after_signal', 'signal_not_before_trade'):
        if b[key]: f.append(f'{k}: {key}={b[key]}')
    if b['p1_mcap_missing']: f.append(f'{k}: P1 market cap missing (auto verdict halted) {b["p1_mcap_missing_months"]}')
    if b['p1_missing_from_snapshot']: f.append(f'{k}: P1 missing from snapshot')
    if pd.Timestamp(b['price_max_date']) > pd.Timestamp(pc.END_CAP): f.append(f'{k}: price after END_CAP')
    if c['n_months'] != b['months']: f.append(f'{k}: months not unique')
    if mode == 'formal':
        if b['months'] != EXPECTED_MONTHS: f.append(f'{k}: months {b["months"]} != {EXPECTED_MONTHS}')
        if c['base_date'] != '2013-12-30' or c['end_date'] != pc.END_CAP: f.append(f'{k}: account span {c["base_date"]}~{c["end_date"]}')
    return f


def external_failures(checks, mode):
    f = []
    for name in REQUIRED_CHECKS + (FORMAL_ONLY_CHECKS if mode == 'formal' else ()):
        if name not in checks: f.append(f'{name} check missing'); continue
        x = checks[name]
        if not (isinstance(x, dict) and x.get('pass') is True): f.append(f'{name} check failed')   # H4: JSON 객체이고 pass가 정확히 true일 때만
    return f


def output_failures(q1, q3, mode):
    """F1: formal에서 질문 1은 k별 7행, 질문 3은 4행이 유효해야 한다. 질문 3 입력 조합 일치·산술 확인(F9)."""
    f = []
    if mode == 'formal':
        for k in COMBO_KS:
            n = int(q1[(q1.k == k)].valid.sum())
            if n != len(Q1_PERIODS): f.append(f'q1 {k}: {n}/{len(Q1_PERIODS)} valid rows')
        if int(q3.valid.sum()) != 4: f.append(f'q3: {int(q3.valid.sum())}/4 valid rows')
    v = q3[q3.valid] if len(q3) and 'valid' in q3 else q3.iloc[0:0]
    if len(v) and not v.inputs_consistent.all(): f.append('q3 inputs not from the same k/combo/period')
    if len(v) and (v.arithmetic_check_residual.abs() > ARITH_TOL_PP).any(): f.append('q3 arithmetic check')
    return f


def load_check(path):
    """R3: 하위 검사가 JSON 없이 끝났거나 읽을 수 없으면 실패로 기록하고 계속한다(verdict.json은 항상 쓴다)."""
    try:
        return json.load(open(path))
    except (OSError, ValueError) as e:
        return {'pass': False, 'reason': f'check output unreadable: {type(e).__name__}'}


def main():
    a = parser().parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    sims = {kv.split('=')[0]: load_sim(kv.split('=')[1]) for kv in a.sim}
    ext = {kv.split('=')[0]: load_check(kv.split('=')[1]) for kv in a.check}
    failed = []
    if set(sims) != set(COMBO_KS): failed.append(f'sims given {sorted(sims)} != {list(COMBO_KS)}')
    mt = metrics_table(sims)
    combos, q1, q3 = combo_rows(mt), q1_rows(mt), q3_rows(mt)
    failed += sum((sim_failures(k, s, a.mode) for k, s in sims.items()), []) + external_failures(ext, a.mode) + output_failures(q1, q3, a.mode)
    lab, reasons = final_label(combos, failed)
    out = lambda n: os.path.join(a.outdir, n)
    mt.to_csv(out('metrics_all.csv'), index=False); combos.to_csv(out('q2_combos.csv'), index=False)
    q1.to_csv(out('q1.csv'), index=False); q1_summary(q1).to_csv(out('q1_summary.csv'), index=False); q3.to_csv(out('q3.csv'), index=False)
    reference_rows(sims).to_csv(out('etf_reference.csv'), index=False); style_rows(sims).to_csv(out('style_etf.csv'), index=False)
    v = {'mode': a.mode, 'q2_label': lab, 'reasons': reasons, 'failed_checks': failed,
         'combo_labels': combos[['k', 'div', 'delist', 'label']].to_dict('records'),
         'q1_valid_rows': {k: int(q1[q1.k == k].valid.sum()) for k in COMBO_KS}, 'q3_valid_rows': int(q3.valid.sum()) if len(q3) else 0,
         'external_checks': {k: isinstance(x, dict) and x.get('pass') is True for k, x in ext.items()},
         'note': 'smoke: structure only, numbers not for interpretation' if a.mode == 'smoke' else ''}
    json.dump(v, open(out('verdict.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v[k] for k in ('mode', 'q2_label', 'reasons', 'external_checks', 'q1_valid_rows', 'q3_valid_rows')}, indent=1, default=str))


if __name__ == '__main__':
    main()
