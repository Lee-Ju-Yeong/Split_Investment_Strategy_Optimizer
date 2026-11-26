"""주문율 격자(same-basket-size-r1) 수익 판정 규칙(설계 4절, 순수 함수). DB·파일을 읽지 않는다.

입력 단위: 주문율 r·바구니 y마다 기본 조건(k=0·5월 지급·동결) 값 {'a','b','ok'}와 18가정 값 목록(같은 형식).
a = X_r 연복리 − H 연복리, b = X_r 연복리 − K 연복리(%p, 반올림 전 부호). ok = 설계 3절 세 기준을 함께 통과.
- 가정 의존: 18가정에서 a(또는 b)의 부호가 한 번이라도 바뀌거나(0 포함) 기준 통과가 바뀌면 dependent. 값이 빠지면 undetermined.
- 2×2 표: 기본 조건에서 기준을 통과하고 부호가 양·음인 바구니만 칸에 넣는다. 누락·기준 미달·동률은 따로 센다(조정자 결정).
- 문장: 경계 구간이 있으면 모든 구간의 주문율을 함께 본다(조정자 결정). 그 주문율 전부에서 여섯 바구니·18가정이 모두 기준을
  통과하고 a가 모두 양수면 S_WIN, 모두 음수면 S_LOSE, 그 밖에는 승패와 차이를 그대로 적는다. 경계 구간이 없으면 수익 문장을 쓰지 않는다.
"""
import same_basket_rules as r1rules
import same_basket_size_common as sz

N_VARIANTS = 18
N_BASKETS = 6
CELLS = r1rules.CELLS
S_WIN = '이 과거 구간에서는 현금을 거의 다 쓴 분할 계좌가 같은 종목 보유보다 더 벌었다'
S_LOSE = '현금을 거의 다 써도 같은 종목 보유가 유리했다'
S_NO_BOUNDARY = '이번 범위에서 경계 미확인. 수익 문장은 쓰지 않고 주문율별 표만 낸다.'
NOTE_PARTIAL = ('큰 주문율에서는 첫날 일부 종목만 사서 보유 종목이 H와 달라질 수 있다. 실제 매수 종목과 시점이 달라 분할 주문만의 효과는 '
                '남은 질문이다.')
NOTE_NO_P = '이미 본 기간의 탐색이며 유의확률을 내지 않는다. 격자 점 사이의 정확한 경계는 추정하지 않는다.'
sign = r1rules.sign


def dependence(variants, key):
    if len(variants) != N_VARIANTS or any(sign(v.get(key)) == 'missing' or v.get('ok') is None for v in variants): return 'undetermined'
    flips = len({sign(v[key]) for v in variants}) > 1 or len({bool(v['ok']) for v in variants}) > 1
    return 'dependent' if flips else 'stable'


def basket_label(base, variants):
    return {'a_sign': sign(base.get('a')), 'b_sign': sign(base.get('b')), 'ok': base.get('ok') is True,
            'a_dependence': dependence(variants, 'a'), 'b_dependence': dependence(variants, 'b')}


def table_2x2(labels):
    """labels: {바구니: basket_label}. 누락 → 기준 미달 → 동률 순으로 따로 세고, 나머지를 칸에 넣는다."""
    cells = {f'{a}/{b}': [] for a, b in CELLS}; other = {'missing': [], 'below_criteria': [], 'tie': []}
    for y, l in sorted(labels.items()):
        sa, sb_ = l['a_sign'], l['b_sign']
        if 'missing' in (sa, sb_): other['missing'].append(y)
        elif not l['ok']: other['below_criteria'].append(y)
        elif 'tie' in (sa, sb_): other['tie'].append(y)
        else: cells[f'{sa}/{sb_}'].append(y)
    return {'cells': cells, 'cell_names': {f'{a}/{b}': n for (a, b), n in CELLS.items()}, **other}


def ratio_summary(base, variants):
    """base: {바구니: {'a','b','ok'}}, variants: {바구니: [18개]}. 한 주문율의 바구니별 표시와 2×2 표."""
    labels = {y: basket_label(base[y], variants.get(y, [])) for y in sorted(base)}
    dep = {k: sorted(y for y, l in labels.items() if l[f'{k}_dependence'] == 'dependent') for k in ('a', 'b')}
    und = {k: sorted(y for y, l in labels.items() if l[f'{k}_dependence'] == 'undetermined') for k in ('a', 'b')}
    return {'labels': labels, 'table': table_2x2(labels), 'dependent': dep, 'undetermined': und}


def all_variants(ratios, data):
    """data: {주문율: {'base':…, 'variants':…}}. 그 주문율들의 여섯 바구니·18가정 값 목록(빠지면 None)."""
    out = []
    for r in ratios:
        v = data.get(r, {}).get('variants', {})
        if len(v) != N_BASKETS or any(len(v[y]) != N_VARIANTS for y in v): return None
        out += [x for y in sorted(v) for x in v[y]]
    return out


def counts_text(r, base, labels):
    t = table_2x2(labels); win = [y for y, l in labels.items() if l['a_sign'] == 'pos']; lose = [y for y, l in labels.items() if l['a_sign'] == 'neg']
    vals = ', '.join(f"{y} {base[y]['a']:+.2f}%p" if sign(base[y].get('a')) != 'missing' else f'{y} 누락' for y in sorted(base))
    dep = sorted(y for y, l in labels.items() if l['a_dependence'] != 'stable')
    return (f"주문율 {sz.ratio_text(r)}: 기본값에서 같은 종목 보유보다 더 번 바구니 {len(win)}개, 덜 번 바구니 {len(lose)}개, "
            f"동률 {len([y for y, l in labels.items() if l['a_sign'] == 'tie'])}개, 누락 {len(t['missing'])}개, 기준 미달 {len(t['below_criteria'])}개, "
            f"가정 의존·미확정 {len(dep)}개 {dep}. a = {vals}.")


def sentence(runs, data):
    """runs: 경계 구간 목록(주문율 목록들). data: {주문율: {'base':…, 'variants':…, 'summary':…}}."""
    if not runs: return {'kind': 'no_boundary', 'ratios': [], 'text': S_NO_BOUNDARY}
    ratios = [r for run in runs for r in run]
    every = all_variants(ratios, data)
    clean = every is not None and all(v.get('ok') is True for v in every)
    if clean and all(sign(v.get('a')) == 'pos' for v in every): return {'kind': 'win', 'ratios': ratios, 'text': S_WIN + '.'}
    if clean and all(sign(v.get('a')) == 'neg' for v in every): return {'kind': 'lose', 'ratios': ratios, 'text': S_LOSE + '.'}
    lines = [counts_text(r, data[r]['base'], data[r]['summary']['labels']) for r in ratios]
    lead = '경계 구간 안의 모든 주문율·바구니·18가정에서 기준 통과와 승패가 한 방향으로 모이지 않았다.' if clean or every is None else \
        '경계 구간 안에 18가정 중 기준 미달인 경우가 있다.'
    return {'kind': 'counts', 'ratios': ratios, 'text': lead + ' ' + ' '.join(lines)}


def runs_text(runs):
    if not runs: return '경계 구간 없음(이번 범위에서 경계 미확인)'
    return ', '.join('[' + ' · '.join(sz.ratio_text(r) for r in run) + ']' for run in runs)
