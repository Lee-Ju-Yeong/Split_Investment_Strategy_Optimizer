"""same-basket-r2 판정 규칙(순수 함수). DESIGN-v1.md 4·5절을 옮긴 것이며 DB·파일을 읽지 않는다.

a = S 연복리 − H 연복리, b = S 연복리 − K 연복리(%p), 반올림 전 부호. 부호·가정 의존은 r1 규칙(same_basket_rules.sign·dependence)을 쓴다.
조건마다 두 기준을 본다: 비중(S ≥ 80%, |S − H| ≤ 10%p; 미달이면 '현금 문제 미해소'), 거래(분할 몫 추가매수 또는 익절 0건이면 '분할 효과 보류').
가정 의존은 a와 b를 각각 18개 부호로 본다. 개수 문장에는 둘을 따로 적고, 부호 변화가 비중 미달 조건에서만 생긴 바구니 수도 적는다.
고정 문장('깎았다'/'보탰다')은 같은 종목 보유와의 비교(a)만 말하므로 a의 안정성만 요구한다(b의 가정 의존은 문장 옆에 따로 적는다).
"""
import same_basket_rules as r1

N_VARIANTS, N_BASKETS = r1.N_VARIANTS, r1.N_BASKETS
CELLS = {('neg', 'neg'): 'S<H·S<K', ('neg', 'pos'): 'S<H·S>K', ('pos', 'neg'): 'S>H·S<K', ('pos', 'pos'): 'S>H·S>K'}
S_ALL_NEG = '이 여섯 번은 높은 투자 비중에서도 20% 분할 운용이 같은 종목 보유보다 수익을 깎았다'
S_ALL_POS = '이 여섯 번은 높은 투자 비중에서도 20% 분할 운용이 같은 종목 보유보다 수익을 보탰다'
LABEL_CASH, LABEL_SPLIT = '현금 문제 미해소', '분할 효과 보류'
NOTE_A_ONLY = '고정 문장은 같은 종목 보유와의 비교(a)만 말하므로 a의 가정 의존만 요구한다'
NOTE_OVERLAP = '여섯 구간은 겹치는 종목이 있어 독립 표본이 아니다. 유의확률을 내지 않는다.'
sign, dependence = r1.sign, r1.dependence


def gate(cond):
    """cond: {'weight_ok': bool, 'split_held': bool}. 판정에 쓸 수 있는지와 걸린 사유."""
    reasons = ([LABEL_CASH] if not cond['weight_ok'] else []) + ([LABEL_SPLIT] if cond['split_held'] else [])
    return {'judged': not reasons, 'reasons': reasons}


def flips_only_in_weight_fail(values, conds):
    """18개가 가정 의존인데, 비중 합격 조건만 보면 부호가 하나뿐이면 True(부호 변화가 비중 미달 조건에서만 생김)."""
    if dependence(values) != 'dependent' or len(conds) != len(values): return False
    kept = {sign(x) for x, c in zip(values, conds) if c['weight_ok']}
    return len(kept) <= 1 and 'missing' not in kept


def side(main, values, conds):
    return {'sign': sign(main), 'dependence': dependence(values), 'only_weight_fail': flips_only_in_weight_fail(values, conds)}


def basket_label(v):
    """v: {'a','b': 주 값, 'a_all','b_all': 18개, 'main': 주 조건 기준, 'all': 18개 기준}."""
    g = gate(v['main'])
    failed = [i for i, c in enumerate(v['all']) if not gate(c)['judged']]
    return {'a': side(v['a'], v['a_all'], v['all']), 'b': side(v['b'], v['b_all'], v['all']),
            'judged': g['judged'], 'reasons': g['reasons'], 'n_conditions_failed': len(failed) if len(v['all']) == N_VARIANTS else None,
            'n_cash_unresolved': sum(not c['weight_ok'] for c in v['all']), 'n_split_held': sum(bool(c['split_held']) for c in v['all'])}


def table_2x2(labels):
    """기본 조건이 두 기준을 통과한 바구니만 칸에 넣는다. 비중 미달·분할 효과 보류(겹칠 수 있음)·동률·누락은 따로 센다."""
    cells = {CELLS[k]: [] for k in CELLS}; other = {'cash_unresolved': [], 'split_held': [], 'tie': [], 'missing': []}
    for y, l in sorted(labels.items()):
        if LABEL_CASH in l['reasons']: other['cash_unresolved'].append(y)
        if LABEL_SPLIT in l['reasons']: other['split_held'].append(y)
        if not l['judged']: continue
        sa, sb_ = l['a']['sign'], l['b']['sign']
        if 'missing' in (sa, sb_): other['missing'].append(y)
        elif 'tie' in (sa, sb_): other['tie'].append(y)
        else: cells[CELLS[(sa, sb_)]].append(y)
    return {'cells': cells, **other}


def counts(labels):
    """판정한 바구니(기본 조건 통과)의 a 부호 수와, 전체의 보류·가정 의존 수."""
    judged = {y: l for y, l in labels.items() if l['judged']}
    c = {s: sorted(y for y, l in judged.items() if l['a']['sign'] == s) for s in ('neg', 'pos', 'tie', 'missing')}
    c['cash_unresolved'] = sorted(y for y, l in labels.items() if LABEL_CASH in l['reasons'])
    c['split_held'] = sorted(y for y, l in labels.items() if LABEL_SPLIT in l['reasons'])
    for key in ('a', 'b'):
        p = '' if key == 'a' else 'b_'
        c[p + 'dependent'] = sorted(y for y, l in labels.items() if l[key]['dependence'] == 'dependent')
        c[p + 'undetermined'] = sorted(y for y, l in labels.items() if l[key]['dependence'] == 'undetermined')
        c[p + 'dependent_only_weight_fail'] = sorted(y for y, l in labels.items() if l[key]['only_weight_fail'])
    c['baskets_with_failed_conditions'] = sorted(y for y, l in labels.items() if l['n_conditions_failed'] != 0)
    return c


def plain_counts(c):
    return (f"판정한 바구니 가운데 S가 같은 종목 보유보다 더 번 바구니 {len(c['pos'])}개, 같은 종목 보유가 더 번 바구니 {len(c['neg'])}개, "
            f"동률 {len(c['tie'])}개, 값 누락 {len(c['missing'])}개. 기본 조건에서 {LABEL_CASH} {len(c['cash_unresolved'])}개, "
            f"{LABEL_SPLIT} {len(c['split_held'])}개. 18가지 조건 가운데 기준 미달 조건이 있는 바구니 {len(c['baskets_with_failed_conditions'])}개. "
            + dependence_text(c))


def dependence_text(c):
    """a(같은 종목 보유와의 차이)와 b(KODEX 보유와의 차이)의 가정 의존을 따로 적는다."""
    part = lambda p, name: (f"{name} 가정 의존 {len(c[p + 'dependent'])}개(그중 부호 변화가 비중 미달 조건에서만 생긴 바구니 "
                            f"{len(c[p + 'dependent_only_weight_fail'])}개), 미확정 {len(c[p + 'undetermined'])}개")
    return part('', 'a(같은 종목 보유와의 차이)') + '. ' + part('b_', 'b(KODEX 보유와의 차이)') + '.'


def sentence(labels):
    """설계 5절 문장. 여섯 바구니가 모두 18가지 조건에서 두 기준을 통과하고 a의 가정 의존이 없을 때만 '모두' 문장을 쓴다."""
    c = counts(labels)
    clean = (len(labels) == N_BASKETS and all(l['n_conditions_failed'] == 0 for l in labels.values())
             and all(l['a']['dependence'] == 'stable' for l in labels.values()))
    note = f" ({NOTE_A_ONLY}. b 가정 의존 {len(c['b_dependent'])}개, b 미확정 {len(c['b_undetermined'])}개.)"
    if clean and len(c['neg']) == N_BASKETS: return {'kind': 'all_neg', 'text': S_ALL_NEG + '.' + note}
    if clean and len(c['pos']) == N_BASKETS: return {'kind': 'all_pos', 'text': S_ALL_POS + '.' + note}
    return {'kind': 'counts', 'text': plain_counts(c)}


def control_flags(rows):
    """대조 C(판정 아님): 조건마다 sign(S − C)와 sign(S − H)가 다르면 잔돈이 승패를 바꾼 것으로 표시한다. rows: [{'basket','a','s_minus_c'}]."""
    out = []
    for r in rows:
        sa, sc_ = sign(r['a']), sign(r['s_minus_c'])
        out.append({**r, 'a_sign': sa, 's_minus_c_sign': sc_, 'flips': 'missing' not in (sa, sc_) and sa != sc_})
    return out


def summarize(values):
    """values: {바구니: basket_label의 입력}. 판정 요약 dict."""
    labels = {y: basket_label(v) for y, v in values.items()}
    return {'labels': labels, 'table': table_2x2(labels), 'counts': counts(labels), 'sentence': sentence(labels), 'note': NOTE_OVERLAP}
