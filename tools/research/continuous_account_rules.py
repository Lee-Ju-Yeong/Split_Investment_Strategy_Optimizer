"""연속 계좌 진단(a)의 판정 규칙(순수 함수). DESIGN-v2.md 5절을 그대로 옮긴 것이며 DB·파일을 읽지 않는다.
continuous_account_verdict.py와 continuous_account_selftest.py가 함께 쓴다."""
import math

EXCESS_MIN, MDD_TOL = 1.0, 5.0      # 2단계 통과 기준(%p)
PASS_MIN, JUDGED = 3, 5             # 원래 다섯 구간 중 3개 이상 = 2단계 통과선
LARGE_GAP, SMALL_GAP = 2.0, 1.0     # 연복리 차이 표시용(%p/년). 판정에는 쓰지 않는다
OFFSETS = (0, 5, 20)
DIVS_K0 = ('none', 'approx', 'upper')
N_WINDOWS = 6
CRITERIA = ('pass', 'pass_calmar')
CRITERION_NAME = {'pass': 'original', 'pass_calmar': 'calmar'}
# 주판정 기준: 사용자 결정 대기(DESIGN-v2.1 5절). None이면 정식 판정은 필수 검사에서 멈춘다.
# 허용값: ('pass',) 원래만 / ('pass_calmar',) 칼마만 / ('pass', 'pass_calmar') 공동 주판정(기준별 최종 라벨을 함께 주판정으로)
PRIMARY_CRITERIA = ('pass', 'pass_calmar')   # 2026-11-22T06:33:59Z 사용자 확정: 원래·칼마 공동 주판정(OR로 합친 단일 결론 금지)
ALLOWED_PRIMARY = (('pass',), ('pass_calmar',), ('pass', 'pass_calmar'))
# 결론 첫 문장(고정, 사용자 확정)
STAGE2_NOTE = '원래 기준에서는 2단계도 2/5로 미통과였다. 결과를 본 뒤 추가한 칼마 보완 기준에서는 3/5였다.'
MECH_EPS = {'gap_cagr': 0.01, 'd_full': 0.05, 'd_invested': 0.001}


def judge(r, e):
    """r, e: {'cagr','mdd','calmar'} (%). 2단계 원래 기준과 칼마 보완 기준."""
    exc, gap = r['cagr'] - e['cagr'], r['mdd'] - e['mdd']
    calmar_ok = r['cagr'] > 0 and e['cagr'] > 0 and r['calmar'] >= e['calmar']
    return {'excess': exc, 'mdd_gap': gap, 'pass': bool(exc >= EXCESS_MIN and gap >= -MDD_TOL),
            'pass_calmar': bool(exc >= EXCESS_MIN and (gap >= -MDD_TOL or calmar_ok))}


def count_label(restart_count, cont_count):
    """원래 다섯 구간 통과 수의 전후 비교. 'flipped'는 재시작 >= 3이고 연속 < 3일 때만."""
    r_ok, c_ok = restart_count >= PASS_MIN, cont_count >= PASS_MIN
    change = 'down' if cont_count < restart_count else ('up' if cont_count > restart_count else 'same')
    if r_ok and not c_ok: return 'flipped', change
    if c_ok and not r_ok: return 'reverse_flipped', change
    return ('unchanged_pass' if r_ok else 'unchanged_fail'), change


def bool_label(restart_pass, cont_pass):
    """전체 기간 ETF 통과의 전후 비교."""
    if restart_pass and not cont_pass: return 'flipped'
    if cont_pass and not restart_pass: return 'reverse_flipped'
    return 'unchanged_pass' if restart_pass else 'unchanged_fail'


def gap_class(gaps):
    """세 시작일의 (재시작 - 연속) 연복리 차이. 정확히 +2는 large, 정확히 1은 mixed."""
    if all(g >= LARGE_GAP for g in gaps): return 'large'
    if all(g <= -LARGE_GAP for g in gaps): return 'reverse_large'
    if all(abs(g) < SMALL_GAP for g in gaps): return 'small'
    return 'mixed'


def across_offsets(labels):
    """시작일별 라벨(같은 기준·같은 배당 가정)을 하나로. 모두 같으면 그 라벨, 아니면 시작일에 따라 다름."""
    labels = list(labels)
    if all(l == 'flipped' for l in labels): return 'flipped'
    if all(l == 'reverse_flipped' for l in labels): return 'reverse_flipped'
    if all(l.startswith('unchanged') for l in labels): return 'unchanged'
    return 'depends_on_start_date'


def final_verdict(primary_by_k, k0_by_div):
    """primary_by_k: {k: 원래 기준 다섯 구간 라벨(배당 approx)}, k0_by_div: {div: k=0의 같은 라벨}.
    배당 가정 사이에서 k=0 라벨이 갈리면 배당 처리에 민감·미확정."""
    if len(set(k0_by_div.values())) > 1: return 'dividend_sensitive_undetermined'
    return {'flipped': 'stage2_verdict_flipped', 'reverse_flipped': 'stage2_verdict_reverse_flipped',
            'unchanged': 'stage2_verdict_unchanged'}.get(across_offsets(primary_by_k.values()), 'depends_on_start_date')


def relation(a, b, eps_a, eps_b):
    """기전 표: 두 차이의 방향 관계. 어느 하나가 0(허용오차 이내)이면 zero."""
    if abs(a) < eps_a or abs(b) < eps_b: return 'zero'
    return 'same' if (a > 0) == (b > 0) else 'opposite'


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def check_job_structure(rows):
    """rows: [{'kind','k','div','window','prefix'}]. 정식 실행 작업 목록의 필수 구성. 오류 문자열 목록을 돌려준다."""
    errs = []
    prefixes = [r['prefix'] for r in rows]
    if len(set(prefixes)) != len(prefixes): errs.append('duplicate prefix')
    if sum(r['kind'] == 'anchor' for r in rows) != 1: errs.append('anchor job missing or duplicated')
    for k in OFFSETS:
        for div in (DIVS_K0 if k == 0 else ('approx',)):
            if sum(r['kind'] == 'cont' and r['k'] == k and r['div'] == div for r in rows) != 1: errs.append(f'cont k={k} div={div} missing')
            n = sum(r['kind'] == 'restart' and r['k'] == k and r['div'] == div for r in rows)
            if n != N_WINDOWS: errs.append(f'restart k={k} div={div}: {n} windows (need {N_WINDOWS})')
    extra = {(r['k'], r['div']) for r in rows if r['kind'] != 'anchor'} - {(0, d) for d in DIVS_K0} - {(k, 'approx') for k in OFFSETS}
    if extra: errs.append(f'unexpected jobs {sorted(extra)}')
    return errs


def primary_errors(primary=None):
    p = PRIMARY_CRITERIA if primary is None else primary
    return [] if p in ALLOWED_PRIMARY else [f'PRIMARY_CRITERIA not decided or invalid: {p!r}']


def primary_final(finals, primary=None):
    """finals: {'pass': 라벨, 'pass_calmar': 라벨}. 주판정 기준이 정해지지 않았으면 pending_user_decision."""
    p = PRIMARY_CRITERIA if primary is None else primary
    if p not in ALLOWED_PRIMARY: return 'pending_user_decision'
    return finals[p[0]] if len(p) == 1 else {CRITERION_NAME[c]: finals[c] for c in p}


def boundary_sensitivity(table):
    """table: {(k, div, period, criterion): {variant: label}}. 세 이어 붙이기 방식의 라벨이 다른 위치 목록."""
    return [{'k': k, 'div': d, 'period': p, 'criterion': c, 'labels': labs}
            for (k, d, p, c), labs in sorted(table.items()) if len(set(labs.values())) > 1]


JOB_COLS = ['kind', 'k', 'window', 'start', 'end', 'tax', 'div', 'prefix']


def compare_job_rows(actual, expected):
    """승인 작업표와 행 단위 정확 대조(순서 포함). actual/expected: dict 목록."""
    norm = lambda rows: [tuple(str(r[c]) for c in JOB_COLS) for r in rows]
    a, e = norm(actual), norm(expected)
    if a == e: return []
    errs = [f'job rows differ: {len(a)} actual vs {len(e)} expected']
    errs += [f'unexpected job row {x}' for x in a if x not in e][:5] + [f'missing job row {x}' for x in e if x not in a][:5]
    return errs + (['job row order differs'] if sorted(a) == sorted(e) else [])


def check_argv(row, argv):
    """meta.argv가 작업표 행과 같은지: 시작·끝·세율·배당·보류."""
    want = [str(row['start']), str(row['end']), '--tax', str(row['tax']), '--div', str(row['div']), '--hold', 'none']
    got = [argv[0], argv[1]] + list(argv[3:]) if len(argv) >= 3 else []
    return [] if got == want else [f"{row['prefix']}: argv {got} != expected {want}"]
