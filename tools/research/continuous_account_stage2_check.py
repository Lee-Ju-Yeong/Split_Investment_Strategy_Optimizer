"""연속 계좌 진단(a) — 2단계 보관 결과와 k=0 재시작 다섯 구간 대조(DESIGN-v2.2 6절). 판정 발행 전에 실행한다.

usage: python -B tools/research/continuous_account_stage2_check.py OUT_DIR STAGE2_DIR ETF_CSV --mode formal|smoke [--approved SHA256_FILE]
  OUT_DIR: 연속 계좌 실행 폴더(jobs.tsv, runs/)  STAGE2_DIR: $R/runs/selection-vs-etf-r1 (split/, stage2_all.csv, 읽기만)
  --approved: 대조 원본 46개(곡선·체결·배당 45 + stage2_all.csv)의 승인 해시 목록. formal에서는 필수.
출력: OUT_DIR/stage2_reconcile.csv, stage2_reconcile.json(기준별 status: linked / withheld)
원래·칼마 두 기준 각각 구간별로 보관값 -> 새 경로 옛 계산법 -> 새 경로 새 계산법 순서로 대조한다. 15개 짝·원본 해시·미설명 차이 없음,
그리고 세 배당 가정 모두에서 통과 구간과 3/5 문턱 여부가 세 단계 내내 같을 때만 그 기준의 2단계 해석을 연결(linked)한다. 아니면 withheld.
3월 실행 체크아웃을 작업 디렉터리로(거래일 조회에 DB 읽기 전용). JSON은 완성 뒤 임시 파일 -> rename으로 원자적으로 발행한다.
이 도구가 실패하거나 JSON이 없거나 깨져도 직접 비교 판정은 발행되며, 그 경우 두 기준 모두 2단계 해석은 withheld다(판정기가 처리).
배당 차이: 보유 확인일(권리 기준 전날, 종목 행의 record_date)·미수 인식일(receivable 사건)·지급일(paid 사건)을 따로 대조한다.
예정된 지급일 이동(같은 FY·같은 금액, 날짜만 다름)과 직전 연도 대체(새 실행의 fallback 종목·그 FY의 미수·지급 금액)는 설명된 차이로 빼고,
남은 배당 차이가 첫 예상 사건보다 앞서면 unexplained다.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

CASH0 = 1e7
FIVE = ['2014', '2016', '2018', '2020', '2022']
DIVS = ('none', 'approx', 'upper')
FILL_KEYS = ['date', 'ticker', 'type', 'dq']
TOL_WON = 0.5                       # 곡선·현금 변동·배당 금액 허용오차(원). 이보다 크면 다름
EXPLAINED = ('identical', 'pay_date_rule', 'dps_fallback')


# ---------- 순수 함수(단위 검사 대상) ----------
def first_curve_diff(old, new):
    """같은 날짜 위에서 처음 TOL_WON을 넘게 다른 날(없으면 None). 날짜가 다르면 호출하지 않는다."""
    d = (old - new).abs() > TOL_WON
    return str(d[d].index[0].date()) if d.any() else None


def first_fill_diff(fo, fn):
    a, b = fo.reset_index(drop=True), fn.reset_index(drop=True); n = min(len(a), len(b))
    for i in range(n):
        if [str(a.loc[i, c]) for c in FILL_KEYS] != [str(b.loc[i, c]) for c in FILL_KEYS] or abs(a.loc[i, 'dcash'] - b.loc[i, 'dcash']) > TOL_WON:
            return str(min(str(a.loc[i, 'date']), str(b.loc[i, 'date'])))
    return None if len(a) == len(b) else str((a if len(a) > n else b).loc[n, 'date'])


def div_events(d):
    if d is None or d.empty or 'event' not in d: return []
    e = d[d.event.notna()]
    return sorted((str(r.event), str(r.date), round(float(r.amount), 0)) for r in e.itertuples())


def _rows(d, kind):
    if d is None or d.empty: return pd.DataFrame()
    if kind == 'hold': return d[d.event.isna()] if 'event' in d else d
    return d[d.event == kind] if 'event' in d else pd.DataFrame()


def _is_true(x):
    return str(x).strip().lower() == 'true'


def dividend_diffs(do, dn):
    """배당 사건을 세 종류로 나눠 대조. 반환: pay_old/pay_new(예정된 지급일 이동), fallback_dates·fallback_fys(대체),
    first_hold/first_recv/first_paid(설명되지 않은 차이의 첫 날짜: 보유 확인일·미수 인식일·지급일), divs_equal."""
    ho, hn = _rows(do, 'hold'), _rows(dn, 'hold')
    key = lambda d: {(int(r.fy), str(r.ticker)): r for r in d.itertuples()} if len(d) else {}
    ko, kn = key(ho), key(hn)
    fb_rows = {k for k, r in kn.items() if _is_true(getattr(r, 'fallback', False))}
    fb_fys = {k[0] for k in fb_rows}
    same_h = lambda a, b: str(a.record_date) == str(b.record_date) and abs(float(a.adj_qty) - float(b.adj_qty)) < 1e-9 and abs(float(a.amount) - float(b.amount)) <= TOL_WON
    hold = [str((kn.get(k) or ko.get(k)).record_date) for k in set(ko) | set(kn)
            if k not in fb_rows and not (k in ko and k in kn and same_h(ko[k], kn[k]))]
    ev = lambda d, kind: {int(r.fy): (str(r.date), float(r.amount)) for r in _rows(d, kind).itertuples()}
    ro, rn, po, pn = ev(do, 'receivable'), ev(dn, 'receivable'), ev(do, 'paid'), ev(dn, 'paid')
    recv = [(rn.get(fy) or ro.get(fy))[0] for fy in set(ro) | set(rn)
            if fy not in fb_fys and not (fy in ro and fy in rn and ro[fy][0] == rn[fy][0] and abs(ro[fy][1] - rn[fy][1]) <= TOL_WON)]
    pay_old, pay_new, paid = [], [], []
    for fy in set(po) | set(pn):
        a, b = po.get(fy), pn.get(fy)
        if a and b and abs(a[1] - b[1]) <= TOL_WON:
            if a[0] != b[0]: pay_old.append(a[0]); pay_new.append(b[0])      # 예정된 지급일 이동
        elif fy in fb_fys and a and b:
            if a[0] != b[0]: pay_old.append(a[0]); pay_new.append(b[0])      # 대체로 금액이 다름(설명됨)
        else:
            paid.append(min(x[0] for x in (a, b) if x))
    first = lambda xs: min(xs) if xs else None
    return {'pay_old': sorted(pay_old), 'pay_new': sorted(pay_new), 'fallback_fys': sorted(fb_fys),
            'fallback_dates': sorted({str(kn[k].record_date) for k in fb_rows}),
            'first_hold': first(hold), 'first_recv': first(recv), 'first_paid': first(paid),
            'divs_equal': div_events(do) == div_events(dn) and not hold}


def classify(p):
    """p: dates_equal, first_curve, first_fill, divs_equal, pay_old, pay_new, fallback_dates, first_hold, first_recv, first_paid.
    예상 사건(예정된 지급일 이동 = 처음 다른 지급일, 대체 = 대체 종목의 보유 확인일) 이후에만 차이가 생겼을 때만 설명 가능.
    날짜 불일치, 곡선·체결 차이가 첫 예상 사건보다 앞서거나, 설명 안 된 배당 차이가 첫 예상 사건 당일 또는 그 전이면 unexplained."""
    if not p['dates_equal']: return 'unexplained'
    div_diffs = [p.get(k) for k in ('first_hold', 'first_recv', 'first_paid') if p.get(k)]
    if p['first_curve'] is None and p['first_fill'] is None and p['divs_equal'] and not div_diffs: return 'identical'
    sym = sorted(set(p['pay_old']) ^ set(p['pay_new']))
    events = [(sym[0], 'pay_date_rule')] if sym else []
    events += [(min(p['fallback_dates']), 'dps_fallback')] if p['fallback_dates'] else []
    if not events: return 'unexplained'
    first_event, cause = min(events)
    diffs = [x for x in (p['first_curve'], p['first_fill']) if x is not None]
    if diffs and min(diffs) < first_event: return 'unexplained'
    if div_diffs and min(div_diffs) <= first_event: return 'unexplained'     # 설명 안 된 배당 차이는 첫 예상 사건 당일도 불가
    return cause


STAGES = ('archived', 'new_run_old_method', 'new_run_new_method')
PASS_MIN = 3


def pass_windows(rows, crit, div, stage):
    return sorted(r['window'] for r in rows if r['div'] == div and r[f'{crit}_{stage}'])


def link_status(rows, crit, n_expected, sources_ok):
    """기준 crit('pass'|'pass_calmar')의 2단계 해석 상태. 직접 비교 판정과 따로 저장한다."""
    if not sources_ok: return 'withheld', 'stage2 source hashes differ from approved list'
    if len(rows) != n_expected: return 'withheld', f'{len(rows)} pairs compared, {n_expected} required'
    if any(r['cause'] not in EXPLAINED for r in rows): return 'withheld', 'unexplained differences'
    for div in sorted({r['div'] for r in rows}):
        sets = [pass_windows(rows, crit, div, st) for st in STAGES]
        if any(x != sets[0] for x in sets): return 'withheld', f'pass windows change across stages for div={div}: {sets}'
        if len({len(x) >= PASS_MIN for x in sets}) > 1: return 'withheld', f'threshold changes for div={div}'
    return 'linked', ''


# ---------- 자료 ----------
def load(prefix):
    c = pd.read_csv(prefix + '_curve.csv', parse_dates=['date']).set_index('date')
    d = pd.read_csv(prefix + '_divs.csv', dtype={'ticker': str}) if os.path.getsize(prefix + '_divs.csv') > 2 else pd.DataFrame()
    return c, d, pd.read_csv(prefix + '_fills.csv', dtype={'ticker': str})


def stage2_metrics(v):
    """stage2_verdict.py 계산법(시작 현금 미포함 낙폭, 첫날부터 연수)."""
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    cagr = ((v.iloc[-1] / CASH0) ** (1 / yrs) - 1) * 100; mdd = float((v / v.cummax() - 1).min()) * 100
    return {'cagr': cagr, 'mdd': mdd, 'calmar': cagr / abs(mdd) if mdd < 0 else np.nan}


def compare(ctx, w, div):
    import continuous_account_rules as rules, continuous_account_verdict as v
    co, do, fo = load(os.path.join(ctx['stage2'], 'split', f'{w}_period_{div}'))
    cn, dn, fn = load(os.path.join(ctx['outdir'], 'runs', f'restart_k0_{w}0101_period_{div}'))
    eq = co.index.equals(cn.index)
    p = {'window': w, 'div': div, 'dates_equal': eq, 'first_curve': first_curve_diff(co.value, cn.value) if eq else None,
         'first_fill': first_fill_diff(fo, fn), **dividend_diffs(do, dn), 'fills_old': len(fo), 'fills_new': len(fn)}
    p['cause'] = classify(p)
    s2 = ctx['s2all']; etf_row = s2[(s2.window == w) & (s2.account == 'ETF')].iloc[0]
    old = s2[(s2.window == w) & (s2.account == 'SPLIT') & (s2.variant == f'tax_period_div_{div}')].iloc[0]
    as_m = lambda r: {'cagr': float(r.cagr), 'mdd': float(r.mdd), 'calmar': float(r.cagr) / abs(float(r.mdd))}
    bd = v.prior_day(ctx['cal'], cn.index[0])
    judged = {'archived': rules.judge(as_m(old), as_m(etf_row)),
              'new_run_old_method': rules.judge(stage2_metrics(cn.value), as_m(etf_row)),
              'new_run_new_method': rules.judge(v.perf(cn.value, CASH0, bd), v.perf(v.etf_curve(ctx['etf'], cn.index), CASH0, bd))}
    if judged['archived']['pass'] != bool(old['pass']): raise ValueError('archived pass flag does not match stage2_all')
    for crit in rules.CRITERIA:
        for st in STAGES: p[f'{crit}_{st}'] = bool(judged[st][crit])
        p[f'{crit}_method_changes'] = p[f'{crit}_new_run_new_method'] != p[f'{crit}_new_run_old_method']
    return p


def sources_ok(approved):
    if not approved: return False
    for line in open(approved):
        h, path = line.split(maxsplit=1)
        path = path.strip()
        if not os.path.exists(path) or hashlib.sha256(open(path, 'rb').read()).hexdigest() != h: return False
    return True


def robust_counts(rows, key):
    """2단계 확정 통과(배당 none과 upper가 같은 구간에서 함께 통과) 구간 수."""
    by = {}
    for r in rows: by.setdefault(r['window'], {})[r['div']] = r[key]
    return sum(1 for x in by.values() if x.get('none') and x.get('upper'))


def criterion_summary(rows, crit, n_expected, ok):
    status, reason = link_status(rows, crit, n_expected, ok)
    return {'status': status, 'reason': reason,
            'pass_windows': {d: {st: pass_windows(rows, crit, d, st) for st in STAGES} for d in sorted({r['div'] for r in rows})},
            'robust_pass_count': {st: robust_counts(rows, f'{crit}_{st}') for st in STAGES}}


def publish_json(obj, path):
    """완성된 내용을 임시 파일에 쓰고 fsync 뒤 rename(원자적 발행)."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=str); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def publish_csv(df, path):
    tmp = path + '.tmp'; df.to_csv(tmp, index=False); os.replace(tmp, path)


def main():
    import stage2_common as sc
    ap = argparse.ArgumentParser(); ap.add_argument('outdir'); ap.add_argument('stage2'); ap.add_argument('etf')
    ap.add_argument('--mode', choices=['formal', 'smoke'], required=True); ap.add_argument('--approved')
    a = ap.parse_args()
    ctx = {'outdir': a.outdir, 'stage2': a.stage2, 's2all': pd.read_csv(os.path.join(a.stage2, 'stage2_all.csv'), dtype={'window': str}),
           'cal': sc.Calendar(sc.trading_dates(sc.connect())), 'etf': pd.read_csv(a.etf, parse_dates=['date']).set_index('date')}
    rows, errors = [], []
    for d in DIVS:
        for w in FIVE:
            if not os.path.exists(os.path.join(a.outdir, 'runs', f'restart_k0_{w}0101_period_{d}_curve.csv')): continue
            try: rows.append(compare(ctx, w, d))
            except Exception as exc: errors.append(f'{w}/{d}: {exc!r}')
    import continuous_account_rules as rules
    ok = sources_ok(a.approved) if a.mode == 'formal' else True
    n_expected = len(FIVE) * len(DIVS) if a.mode == 'formal' else len(rows)
    by_crit = {rules.CRITERION_NAME[c]: criterion_summary(rows, c, n_expected, ok) for c in rules.CRITERIA}
    for c in by_crit.values():
        if errors: c.update(status='withheld', reason='errors while comparing: ' + '; '.join(errors))
        if a.mode == 'smoke': c.update(status='smoke_not_applicable')
    publish_csv(pd.DataFrame(rows), os.path.join(a.outdir, 'stage2_reconcile.csv'))
    summ = {'mode': a.mode, 'by_criterion': by_crit, 'n_compared': len(rows), 'n_required': len(FIVE) * len(DIVS), 'sources_hash_ok': ok,
            'cause_counts': pd.Series([r['cause'] for r in rows]).value_counts().to_dict() if rows else {},
            'unexplained': [(r['window'], r['div']) for r in rows if r['cause'] not in EXPLAINED], 'errors': errors}
    publish_json(summ, os.path.join(a.outdir, 'stage2_reconcile.json'))
    print(json.dumps({c: x['status'] for c, x in by_crit.items()} | {'n_compared': len(rows)}))


if __name__ == '__main__':
    main()
