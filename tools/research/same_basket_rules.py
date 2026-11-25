"""같은 종목 대조(same-basket-r1) 판정 규칙(순수 함수). DESIGN-v1.1.md 7절을 그대로 옮긴 것이며 DB·파일을 읽지 않는다.

a = X 연복리 − H 연복리, b = X 연복리 − K 연복리(%p). 반올림 전 부호로 읽는다.
same_basket_verdict.py와 same_basket_selftest.py가 함께 쓴다.
"""
import math

N_VARIANTS = 18                    # 시작일 3 × 배당 3 × 상폐 2
N_BASKETS = 6
MOSTLY_MIN = 5                     # '대체로'를 쓸 수 있는 같은 방향 바구니 수
CELLS = {('neg', 'neg'): '현행 규칙 계좌가 양쪽 모두에 졌다',
         ('neg', 'pos'): '같은 종목 보유가 가장 높고 분할도 ETF를 앞섰다',
         ('pos', 'neg'): '분할이 수익을 보탰고 ETF가 가장 높다',
         ('pos', 'pos'): '현행 규칙 계좌가 양쪽 모두를 앞섰다'}
S_ALL_NEG = '여섯 바구니 모두 같은 종목 보유가 현행 분할 계좌보다 더 벌었다'
S_ALL_POS = '여섯 바구니 모두 현행 분할 계좌가 같은 종목 보유보다 더 벌었다'
S_POS_NOTE = 'X가 K를 밑돈 곳에서는 H도 K를 밑돌았다'
S_MOSTLY = {'neg': '대체로 같은 종목 보유가 현행 분할 계좌보다 더 벌었다', 'pos': '대체로 현행 분할 계좌가 같은 종목 보유보다 더 벌었다'}
AUX_LABEL = '배정 방식에 따라 다름'
AUX_MISSING = '보조 값 누락'
AUX_RATIO_NAMES = ('r10', 'r44')
NOTE_OVERLAP = '여섯 기본 2년 구간은 서로 겹치지 않는다. 같은 종목이 다시 들어올 수 있고 연장 기간도 겹치므로 유의확률을 내지 않는다.'


def sign(x):
    """반올림 전 부호. 값이 없거나 유한하지 않으면 missing, 정확히 0이면 tie."""
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x): return 'missing'
    return 'pos' if x > 0 else ('neg' if x < 0 else 'tie')


def dependence(values):
    """18가지 값의 가정 의존 여부. 하나라도 빠지면 undetermined, 부호가 한 번이라도 바뀌면(0 포함) dependent."""
    signs = [sign(v) for v in values]
    if len(signs) != N_VARIANTS or 'missing' in signs: return 'undetermined'
    return 'stable' if len(set(signs)) == 1 else 'dependent'


def basket_label(main, values):
    """한 바구니의 한 비교(a 또는 b): 주 값 부호와 가정 의존 상태."""
    return {'sign': sign(main), 'dependence': dependence(values)}


def table_2x2(labels):
    """labels: {바구니: {'a': {...}, 'b': {...}}}. 기본값 부호로 칸을 채우고 동률·누락은 따로 센다."""
    cells = {f'{a}/{b}': [] for a, b in CELLS}; other = {'tie': [], 'missing': []}
    for y, l in sorted(labels.items()):
        sa, sb = l['a']['sign'], l['b']['sign']
        if 'missing' in (sa, sb): other['missing'].append(y)
        elif 'tie' in (sa, sb): other['tie'].append(y)
        else: cells[f'{sa}/{sb}'].append(y)
    return {'cells': cells, 'cell_names': {f'{a}/{b}': n for (a, b), n in CELLS.items()}, **other}


def count_signs(labels, key):
    out = {'neg': [], 'pos': [], 'tie': [], 'missing': [], 'dependent': [], 'undetermined': []}
    for y, l in sorted(labels.items()):
        out[l[key]['sign']].append(y)
        if l[key]['dependence'] != 'stable': out[l[key]['dependence']].append(y)
    return out


def plain_counts(c):
    return (f"같은 종목 보유가 더 번 바구니 {len(c['neg'])}개, 현행 분할 계좌가 더 번 바구니 {len(c['pos'])}개, 동률 {len(c['tie'])}개, "
            f"값 누락 {len(c['missing'])}개. 가정 의존 {len(c['dependent'])}개, 미확정 {len(c['undetermined'])}개.")


def sentence(labels):
    """쓸 문장(a 기준). 여섯 모두 완전하고 가정 의존이 없을 때만 '모두'·'대체로'를 쓴다."""
    c = count_signs(labels, 'a')
    clean = len(labels) == N_BASKETS and not (c['missing'] or c['dependent'] or c['undetermined'])
    if clean and len(c['neg']) == N_BASKETS: return {'kind': 'all_neg', 'text': S_ALL_NEG + '.'}
    if clean and len(c['pos']) == N_BASKETS:
        below = [y for y, l in sorted(labels.items()) if l['b']['sign'] == 'neg']
        return {'kind': 'all_pos', 'text': S_ALL_POS + '.' + (f' {S_POS_NOTE}({below}).' if below else ''), 'x_below_k': below}
    for s in ('neg', 'pos'):
        if clean and len(c[s]) >= MOSTLY_MIN: return {'kind': f'mostly_{s}', 'text': plain_counts(c) + ' ' + S_MOSTLY[s] + '.'}
    return {'kind': 'counts', 'text': plain_counts(c)}


def aux_flags(labels, aux):
    """aux: {바구니: {비율 이름: 보조 a}}. 주 값과 보조 값의 부호가 양쪽 모두 양·음으로 확인되고 서로 다를 때만 '배정 방식에 따라 다름'.
    어느 쪽이든 값이 없으면 '보조 값 누락'. 동률(0)은 방향 변화로 보지 않고 ties에 따로 남긴다."""
    out = {}
    for y, l in sorted(labels.items()):
        signs = {r: sign(aux.get(y, {}).get(r)) for r in AUX_RATIO_NAMES}
        main = l['a']['sign']
        missing = main == 'missing' or 'missing' in signs.values()
        diff = [] if missing else sorted(r for r, s in signs.items() if s in ('pos', 'neg') and main in ('pos', 'neg') and s != main)
        ties = sorted(r for r, s in signs.items() if s == 'tie')
        out[y] = {'main_sign': main, 'aux_signs': signs, 'differs': diff, 'ties': ties,
                  'label': AUX_MISSING if missing else (AUX_LABEL if diff else '')}
    return out


def summarize(values, aux):
    """values: {바구니: {'a': 주 값, 'b': 주 값, 'a_all': [18], 'b_all': [18]}}. 판정 요약 dict."""
    labels = {y: {'a': basket_label(v['a'], v['a_all']), 'b': basket_label(v['b'], v['b_all'])} for y, v in values.items()}
    return {'labels': labels, 'table': table_2x2(labels), 'a_counts': count_signs(labels, 'a'), 'b_counts': count_signs(labels, 'b'),
            'sentence': sentence(labels), 'aux': aux_flags(labels, aux), 'note': NOTE_OVERLAP}


def check_job_structure(rows, expected):
    """rows: [{'kind','prefix'}]. 종류별 개수와 실행명 중복. 오류 문자열 목록."""
    errs = []
    prefixes = [r['prefix'] for r in rows]
    if len(set(prefixes)) != len(prefixes): errs.append('duplicate prefix')
    for kind, n in expected.items():
        got = sum(r['kind'] == kind for r in rows)
        if got != n: errs.append(f'{kind}: {got} jobs (need {n})')
    extra = sorted({r['kind'] for r in rows} - set(expected))
    return errs + ([f'unexpected job kinds {extra}'] if extra else [])


def refusal_reasons(status, required):
    """조립 거부 조건. status: {실행명: [종료 코드, ...]}(기록 순서), required: 실행명 목록.
    모든 작업의 마지막 기록이 0이어야 하고, 기록에 없는 작업·작업표에 없는 기록이 있으면 거부한다."""
    errs = [f'job not recorded: {p}' for p in required if p not in status]
    errs += [f'job failed: {p} exit={status[p][-1]}' for p in required if p in status and status[p][-1] != 0]
    return errs + [f'status for unknown job: {p}' for p in sorted(set(status) - set(required))]
