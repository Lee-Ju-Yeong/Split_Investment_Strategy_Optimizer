"""same-basket-r2 자체 검사: 합성 자료로 보유 몫·작은 몫 장부(손계산), S 합산과 비중(손계산), 비중 합격선 경계, 분할 효과 보류,
판정 규칙(2×2 표·문장·가정 의존·따로 세기), 대조 C 표시, 기술 검사의 오류 검출, 조립, 조립 거부를 확인한다.

usage: python -B tools/research/same_basket_core_selftest.py [--no-db]
3월 실행 체크아웃을 작업 디렉터리로 실행한다. --no-db가 없으면 조립기를 임시 폴더에서 실제로 돌려 거부(종료 코드 2, 표·문장 없음)를 확인한다
(DB는 거래일 목록만 읽는다). 실제 성과 자료는 읽지 않는다. 모든 검사가 통과하면 'N tests passed'를 출력하고 0으로 끝난다.
r1 자체 검사(same_basket_selftest)의 합성 자료 함수(calendar·hold_input·real_x)를 불러 쓴다.
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import same_basket_common as sb
import same_basket_ledger as ledger
import same_basket_checks as chk
import same_basket_selftest as t1
import same_basket_core_common as cc
import same_basket_core_rules as rules
import same_basket_core_checks as cchk
import same_basket_core_verdict as cv

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = []
raises = t1.raises


def test(fn):
    TESTS.append(fn); return fn


def first(df):
    return next(df.itertuples())


# ---------------- 보유 몫·작은 몫 장부 ----------------
def two_stock_input():
    """A: 시가 10,000원(호가 10), B: 시가 33,000원(호가 50). 날짜 0은 시작 전날."""
    return t1.hold_input([[np.nan, np.nan], [10000, 33000], [10100, 33100]], [[np.nan, np.nan], [10050, 33050], [10200, 33500]], tickers=('AAA001', 'BBB002'))


def run_part(amounts, inp=None):
    with cc.account(amounts['cash'], amounts['slot']):
        return ledger.run_hold(inp or two_stock_input(), 'approx', 'freeze')


@test
def core_small_and_h_ledgers_match_hand_calculation():
    # 손계산: 수량 = 수수료(매수금액 × 3 // 20000) 포함 한도 이하 최대 정수 주
    # 40만 원: A 39주 390,000 + 58 = 390,058 / B 12주 396,000 + 59 = 396,059
    # 10만 원: A 9주 90,000 + 13 = 90,013 / B 3주 99,000 + 14 = 99,014
    # 50만 원(H): A 49주 490,000 + 73 = 490,073 / B 15주 495,000 + 74 = 495,074
    for amounts, qa, qb, cost in ((cc.CORE, 39, 12, 390058 + 396059), (cc.SMALL, 9, 3, 90013 + 99014), ({'cash': sb.CASH0, 'slot': sb.SLOT_CASH}, 49, 15, 490073 + 495074)):
        s = run_part(amounts)
        assert s['qty'] == [qa, qb], (amounts, s['qty'])
        assert abs(s['cash'] - (amounts['cash'] - cost)) < 1e-9 and all(b['budget'] == amounts['slot'] for b in s['buys'])
        assert s['curve'][0]['value'] == amounts['cash'] and s['curve'][1]['stock_value'] == qa * 10050 + qb * 33050
    assert (sb.CASH0, sb.SLOT_CASH) == (1e7, 5e5)            # 계좌 금액은 항상 되돌린다
    # 대조 C의 뜻: 40만 + 10만으로 나누면 정수 주 잔돈 때문에 50만 원 한 번보다 적게 살 수 있다(A 48주 < 49주)
    assert run_part(cc.CORE)['qty'][0] + run_part(cc.SMALL)['qty'][0] == 48


@test
def account_switch_restores_on_error():
    def boom():
        with cc.account(1.0, 2.0):
            assert sb.CASH0 == 1.0 and sb.SLOT_CASH == 2.0
            raise RuntimeError('x')
    assert raises(boom, RuntimeError) and (sb.CASH0, sb.SLOT_CASH) == (1e7, 5e5)


# ---------------- S 합산과 비중 ----------------
def three_days():
    return pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'])


def s_parts():
    d = three_days()
    core = pd.DataFrame({'value': [8e6, 8.1e6, 8.2e6], 'cash': [8e6, 0.2e6, 0.2e6], 'receivable': [0.0, 0.0, 0.0], 'stock_value': [0.0, 7.9e6, 8.0e6]}, index=d)
    x = pd.DataFrame({'value_engine': [2e6, 2e6, 2.05e6], 'receivable': [0.0, 0.0, 0.05e6], 'value': [2e6, 2e6, 2.1e6]}, index=d)
    st = pd.DataFrame({'cash': [2e6, 1.5e6, 1.45e6]}, index=d)
    return core, x, st


@test
def s_is_daily_sum_and_weight_matches_hand_calculation():
    core, x, st = s_parts()
    s = cc.combine(core, x, st)
    assert list(s.value) == [10e6, 10.1e6, 10.3e6] and list(s.stock_value) == [0.0, 8.4e6, 8.6e6] and list(s.split_stock) == [0.0, 0.5e6, 0.6e6]
    assert list(s.cash) == [10e6, 1.7e6, 1.65e6] and list(s.receivable) == [0.0, 0.0, 0.05e6]
    assert np.allclose(s.value, s.cash + s.stock_value + s.receivable)
    w = cc.weight_stats(s.stock_value.iloc[1:], s.value.iloc[1:], [9e6, 9.1e6], [10e6, 10e6])
    hand_s = (8.4 / 10.1 + 8.6 / 10.3) / 2; hand_h = (0.90 + 0.91) / 2
    assert abs(w['s_weight'] - hand_s) < 1e-12 and abs(w['h_weight'] - hand_h) < 1e-12 and abs(w['weight_gap'] - (hand_s - hand_h)) < 1e-12
    assert abs(w['daily_gap_min'] - min(8.4 / 10.1 - 0.90, 8.6 / 10.3 - 0.91)) < 1e-12
    assert raises(lambda: cc.combine(core, x.iloc[:-1], st), ValueError) and raises(lambda: cc.daily_weight([1.0], [0.0]), ValueError)


@test
def weight_gate_boundaries():
    ok = cc.weight_ok
    assert ok(0.80, 0.85) and ok(0.80, 0.70 + 1e-9) and ok(0.95, 0.99) and ok(0.80, 0.90)
    assert not ok(0.7999, 0.80) and not ok(0.80, 0.9001) and not ok(0.95, 0.80) and not ok(float('nan'), 0.9) and not ok(0.9, float('nan'))


@test
def split_effect_hold_needs_both_addbuy_and_takeprofit():
    f = lambda reasons: pd.DataFrame({'type': [t for t, _ in reasons], 'reason': [r for _, r in reasons], 'ticker': ['A'] * len(reasons)})
    both = cc.split_trades(f([('BUY', '신규 진입'), ('BUY', cc.REASON_ADDBUY), ('SELL', cc.REASON_PROFIT), ('SELL', '손절매 (평균가 기준)')]))
    assert both == {'split_addbuys': 1, 'split_takeprofits': 1, 'split_tickers_bought': 1} and not cc.split_effect_held(both)
    assert cc.split_effect_held(cc.split_trades(f([('BUY', '신규 진입'), ('SELL', cc.REASON_PROFIT)])))
    assert cc.split_effect_held(cc.split_trades(f([('BUY', '신규 진입'), ('BUY', cc.REASON_ADDBUY), ('SELL', '매매 미발생 기간 초과')])))
    assert cc.split_effect_held(cc.split_trades(pd.DataFrame())) and cc.split_effect_held(cc.split_trades(None))


# ---------------- 판정 규칙 ----------------
GOOD = {'weight_ok': True, 'split_held': False}


def bv(a=-1.0, b=-1.0, a_all=None, b_all=None, main=None, all_=None):
    return {'a': a, 'b': b, 'a_all': a_all if a_all is not None else [a] * 18, 'b_all': b_all if b_all is not None else [b] * 18,
            'main': main or GOOD, 'all': all_ or [GOOD] * 18}


@test
def sentences_and_table_follow_the_fixed_rules():
    ys = sb.YEARS
    s = rules.summarize({y: bv() for y in ys})
    assert s['sentence']['kind'] == 'all_neg' and s['sentence']['text'].startswith(rules.S_ALL_NEG + '.') and s['table']['cells']['S<H·S<K'] == list(ys)
    assert rules.summarize({y: bv(a=1.0, b=1.0) for y in ys})['sentence']['kind'] == 'all_pos'
    # 기본 조건은 통과했지만 18가지 가운데 한 조건이 비중 미달 → '모두' 문장 불가, 표에는 남는다
    one_bad = [GOOD] * 17 + [{'weight_ok': False, 'split_held': False}]
    s = rules.summarize({y: bv(all_=one_bad if y == 2016 else None) for y in ys})
    assert s['sentence']['kind'] == 'counts' and 2016 in s['table']['cells']['S<H·S<K'] and s['labels'][2016]['n_conditions_failed'] == 1
    assert s['counts']['baskets_with_failed_conditions'] == [2016]
    # 가정 의존(0 포함)이면 '모두' 문장 불가
    s = rules.summarize({y: bv(a_all=[-1.0] * 17 + [0.0] if y == 2018 else None) for y in ys})
    assert s['sentence']['kind'] == 'counts' and s['counts']['dependent'] == [2018]


@test
def gate_failures_are_counted_separately():
    ys = sb.YEARS
    vals = {2014: bv(-1, -1), 2016: bv(-1, 1), 2018: bv(1, -1), 2020: bv(1, 1),
            2022: bv(-1, -1, main={'weight_ok': False, 'split_held': True}, all_=[{'weight_ok': False, 'split_held': True}] * 18),
            2024: bv(0.0, -1)}
    s = rules.summarize(vals)
    t = s['table']
    assert t['cells'] == {'S<H·S<K': [2014], 'S<H·S>K': [2016], 'S>H·S<K': [2018], 'S>H·S>K': [2020]}
    assert t['cash_unresolved'] == [2022] and t['split_held'] == [2022] and t['tie'] == [2024] and t['missing'] == []
    c = s['counts']
    assert c['neg'] == [2014, 2016] and c['pos'] == [2018, 2020] and c['tie'] == [2024] and 2022 not in c['neg']
    assert s['labels'][2022]['reasons'] == [rules.LABEL_CASH, rules.LABEL_SPLIT] and s['labels'][2022]['n_cash_unresolved'] == 18
    assert s['sentence']['kind'] == 'counts' and f"{rules.LABEL_CASH} 1개" in s['sentence']['text'] and f"{rules.LABEL_SPLIT} 1개" in s['sentence']['text']
    m = rules.summarize({**{y: bv() for y in ys}, 2020: bv(a=None, a_all=[None] * 18)})
    assert m['table']['missing'] == [2020] and m['counts']['undetermined'] == [2020] and m['sentence']['kind'] == 'counts'
    only_split = rules.summarize({**{y: bv() for y in ys}, 2016: bv(main={'weight_ok': True, 'split_held': True})})
    assert only_split['table']['split_held'] == [2016] and only_split['table']['cash_unresolved'] == [] and only_split['sentence']['kind'] == 'counts'


@test
def control_flags_mark_sign_changes_only():
    out = rules.control_flags([{'a': -1.0, 's_minus_c': 0.5}, {'a': -1.0, 's_minus_c': -0.2}, {'a': 1.0, 's_minus_c': 0.0}, {'a': None, 's_minus_c': 1.0}])
    assert [o['flips'] for o in out] == [True, False, True, False]


# ---------------- 작업표 ----------------
@test
def job_table_has_main_and_control_for_every_condition():
    cal = t1.calendar(); jobs = cc.build_jobs(cal)
    assert jobs.groupby('kind').size().to_dict() == cc.N_JOBS and jobs.prefix.is_unique and list(jobs.id) == list(range(1, 217))
    f = jobs.iloc[0]; assert (f.kind, f.basket, f.k, f['div'], f.delist) == ('main', 2014) + sb.MAIN
    m, c = jobs[jobs.kind == 'main'], jobs[jobs.kind == 'ctrl']
    assert list(m.prefix.str.replace('main_', '')) == list(c.prefix.str.replace('ctrl_', '')) and (m.end.values == c.end.values).all()
    r1jobs = sb.build_jobs(cal, t1.fake_baskets()); r1m = r1jobs[r1jobs.kind == 'main']
    assert list(r1m.prefix) == list(m.prefix) and list(r1m.start) == list(m.start) and list(r1m.end) == list(m.end)    # r1 주 작업과 같은 이름·기간
    assert sb.compare_jobs(jobs, cc.build_jobs(cal)) == [] and sb.compare_jobs(jobs.iloc[:-1], jobs)
    smoke = cc.build_smoke_jobs(cal, 2014, '2014-02-28')
    assert smoke.groupby('kind').size().to_dict() == {'ctrl': 6, 'main': 6} and set(smoke.k) == {0} and set(smoke.end) == {'2014-02-28'}


# ---------------- 기술 검사 ----------------
def write_part(tmp, prefix, part, amounts, inp):
    state = run_part(amounts, inp)
    meta = {'price_basis': 'adjusted', 'base_date': inp['dates'][0], 'delisted': {}, 'account_start': inp['dates'][1], 'end': inp['dates'][-1], 'div': 'approx',
            'delist': 'freeze', 'basket_file': 'b.csv', 'basket_sha256': 'h', 'part': part, 'cash0': amounts['cash'], 'slot_cash': amounts['slot']}
    ledger.write_hold(os.path.join(tmp, prefix + '_' + part), state, meta)
    return chk.read_h(os.path.join(tmp, prefix + '_' + part))


class Job:
    def __init__(self, **kw): self.__dict__.update(kw)


@test
def slot_checks_use_the_part_amounts_and_catch_faults():
    inp = t1.hold_input([[np.nan, np.nan]] + [[10000, 33000]] * 20, [[np.nan, np.nan]] + [[10050, 33050]] * 20, tickers=('AAA001', 'BBB002'))
    dates = pd.DatetimeIndex(inp['dates']); job = Job(prefix='p', kind='ctrl', start=inp['dates'][1], end=inp['dates'][-1], div='approx', delist='freeze', k=0, basket=2014)
    tmp = tempfile.mkdtemp()
    try:
        core = write_part(tmp, 'p', 'core', cc.CORE, inp); small = write_part(tmp, 'p', 'small', cc.SMALL, inp)
        basket = ['AAA001', 'BBB002']
        assert cchk.slot_errors(job, 'core', core, dates, basket, 'h', cc.CORE) == [], cchk.slot_errors(job, 'core', core, dates, basket, 'h', cc.CORE)
        assert cchk.slot_errors(job, 'small', small, dates, basket, 'h', cc.SMALL) == []
        assert cchk.slot_errors(job, 'core', core, dates, basket, 'h', cc.SMALL)            # 작은 몫 금액으로 보면 틀린다
        bad = dict(core, buys=core['buys'].assign(quantity=core['buys'].quantity + 1))
        assert any('price/quantity/fee' in e for e in cchk.slot_errors(job, 'core', bad, dates, basket, 'h', cc.CORE))
        c = pd.DataFrame({'value': core['curve'].value + small['curve'].value, 'stock_value': core['curve'].stock_value + small['curve'].stock_value,
                          'cash': core['curve'].cash + small['curve'].cash}, index=core['curve'].index)
        assert cchk.c_errors(job, {'core': core, 'small': small, 'c': c}) == []
        assert cchk.c_errors(job, {'core': core, 'small': small, 'c': c.assign(value=c.value + (c.index == c.index[5]))})
        assert (sb.CASH0, sb.SLOT_CASH) == (1e7, 5e5)
    finally: shutil.rmtree(tmp)


@test
def split_checks_expect_two_million_and_s_checks_catch_tampering():
    cal = t1.calendar(); job = first(cc.build_jobs(cal)); tickers = t1.fake_baskets()[2014]
    with cc.account(cc.SPLIT_CASH, sb.SLOT_CASH):
        x = t1.real_x(job, cal, tickers)                  # r1 합성 X를 200만 원 계좌로
    assert cchk.x_errors(job, x, cal, tickers, 'h') == [], cchk.x_errors(job, x, cal, tickers, 'h')
    assert chk.x_errors(job, x, cal, tickers, 'h')       # r1 기대값(1,000만 원)으로 보면 틀린다 → 감싸기가 실제로 금액을 바꾼다
    assert chk.expect_account(job)['initial_cash'] == sb.CASH0
    days = x['curve'].index
    core = pd.DataFrame({'value': 8e6, 'cash': 8e6, 'receivable': 0.0, 'stock_value': 0.0}, index=days)
    core.iloc[1:, core.columns.get_loc('cash')] = 1e5; core.iloc[1:, core.columns.get_loc('stock_value')] = 7.9e6
    out = {'x': x, 'core': {'curve': core}, 's': cc.combine(core, x['curve'], x['state'])}
    assert cchk.s_errors(job, out) == []
    tampered = out['s'].copy(); tampered.iloc[40, tampered.columns.get_loc('value')] += 1.0
    assert cchk.s_errors(job, dict(out, s=tampered))
    moved = out['s'].copy(); moved.iloc[0, moved.columns.get_loc('core_cash')] = 7e6; moved.iloc[0, moved.columns.get_loc('split_cash')] = 3e6
    assert any('8,000,000 + 2,000,000' in e or 'differs' in e for e in cchk.s_errors(job, dict(out, s=moved)))


@test
def r1_byte_reference_check_catches_any_difference():
    tmp = tempfile.mkdtemp(); job = Job(prefix='main_b2014_k0_approx_freeze')
    try:
        ref = {}
        for f in cc.H_FILES:
            p = os.path.join(tmp, f'{job.prefix}_{f}'); open(p, 'w').write(f)
            ref[f'runs/{job.prefix}_{f}'] = hashlib.sha256(f.encode()).hexdigest()
        assert cchk.hk_repro_errors(job, tmp, ref) == []
        open(os.path.join(tmp, f'{job.prefix}_H_curve.csv'), 'a').write(' ')
        assert cchk.hk_repro_errors(job, tmp, ref) == [f'{job.prefix}: H/K differ from the r1 run (H_curve.csv)']
        sums = os.path.join(tmp, 'SUMS'); open(sums, 'w').write('abc  ./runs/x_H_curve.csv\ndef  ./code/y.py\n')
        assert cchk.r1_reference(sums) == {'runs/x_H_curve.csv': 'abc'}
    finally: shutil.rmtree(tmp)


@test
def control_job_without_main_is_refused():
    cal = t1.calendar(); jobs = cc.build_jobs(cal)
    ctrl_only = jobs[(jobs.kind == 'ctrl') & (jobs.basket == 2014)]
    errs = cchk.cross_errors(ctrl_only, {})
    assert len(errs) == 18 and all('main job missing' in e for e in errs)


# ================= r2 (review-reconciled-impl.md 고칠 것 1·2) =================
def ledger_weights(n, s_frac, h_frac):
    """일별 장부에서부터: 순자산 1,000만 원 근처를 매일 바꾸고 주식평가액 = 순자산 × 비중."""
    v = 1e7 + 12345.67 * np.arange(n)
    return cc.weight_stats(v * s_frac, v, v * h_frac, v)


@test
def r2_fix1_exact_weight_boundaries_pass_from_daily_ledger():
    w = ledger_weights(492, 0.80, 0.90)                      # 정확히 80%, 10%p: 설계상 합격
    assert cc.weight_ok(w['s_weight'], w['h_weight']), (w['s_weight'], w['h_weight'])
    w = ledger_weights(492, 0.85, 0.75)                      # S가 H보다 정확히 10%p 높음
    assert cc.weight_ok(w['s_weight'], w['h_weight'])
    for s, h in ((0.80 - 1e-9, 0.85), (0.80, 0.90 + 1e-9), (0.95, 0.85 - 1e-9), (0.79, 0.80)):   # 바로 바깥은 불합격
        w = ledger_weights(492, s, h)
        assert not cc.weight_ok(w['s_weight'], w['h_weight']), (s, h)


@test
def r2_fix2_counts_sentence_reports_a_and_b_dependence_separately():
    ys = sb.YEARS
    b_flip = [-1.0] * 17 + [1.0]
    s = rules.summarize({y: bv(b_all=b_flip if y == 2020 else None) for y in ys})          # a는 모두 안정·음수, b만 한 조건에서 뒤집힘
    assert s['sentence']['kind'] == 'all_neg' and s['sentence']['text'].startswith(rules.S_ALL_NEG + '.')
    assert rules.NOTE_A_ONLY in s['sentence']['text'] and 'b 가정 의존 1개' in s['sentence']['text']
    assert s['counts']['dependent'] == [] and s['counts']['b_dependent'] == [2020]
    mixed = {y: bv(a=(-1.0 if y < 2020 else 1.0), b_all=b_flip if y == 2016 else None) for y in ys}
    t = rules.summarize(mixed)['sentence']
    assert t['kind'] == 'counts' and 'b(KODEX 보유와의 차이) 가정 의존 1개' in t['text'] and 'a(같은 종목 보유와의 차이) 가정 의존 0개' in t['text']
    # 부호 변화가 비중 미달 조건에서만 생긴 경우
    conds = [GOOD] * 17 + [{'weight_ok': False, 'split_held': False}]
    u = rules.summarize({y: bv(a_all=[-1.0] * 17 + [1.0] if y == 2018 else None, all_=conds if y == 2018 else None) for y in ys})
    assert u['labels'][2018]['a']['only_weight_fail'] and u['counts']['dependent_only_weight_fail'] == [2018]
    assert '비중 미달 조건에서만 생긴 바구니 1개' in u['sentence']['text']
    v = rules.summarize({y: bv(a_all=[1.0] + [-1.0] * 17 if y == 2018 else None, all_=conds if y == 2018 else None) for y in ys})
    assert not v['labels'][2018]['a']['only_weight_fail'] and v['counts']['dependent'] == [2018]     # 합격 조건에서 뒤집히면 해당 없음


# ---------------- 조립(합성 산출물) ----------------
def days_of(job, cal):
    return cal[(cal >= sb.prev_day(cal, job.start)) & (cal <= pd.Timestamp(job.end))]


def curve(days, final, weight):
    v = np.linspace(sb.CASH0, final, len(days)); stock = v * weight; stock[0] = 0.0
    return pd.DataFrame({'value': v, 'stock_value': stock}, index=days)


def fake_main(days, s, h, k, sw=0.9, hw=0.95, trades=True):
    reasons = [('BUY', '신규 진입', 'A'), ('BUY', cc.REASON_ADDBUY, 'A'), ('SELL', cc.REASON_PROFIT, 'A')] if trades else [('BUY', '신규 진입', 'A')]
    fills = pd.DataFrame({'type': [r[0] for r in reasons], 'reason': [r[1] for r in reasons], 'ticker': [r[2] for r in reasons]})
    x = {'fills': fills, 'sbmeta': {'allowed': ['A', 'B', 'C'], 'start_unaffordable': ['C'], 'ever_candidate': ['A', 'B']}}
    return {'s': curve(days, s, sw), 'h': {'curve': curve(days, h, hw), 'meta': {'skipped_at_start': [], 'delisted': {}}},
            'k': {'curve': pd.Series(np.linspace(sb.CASH0, k, len(days)), index=days)}, 'x': x, 'core': {'meta': {'skipped_at_start': ['Z']}}}


def fake_outs(jobs, cal, finals, special=None):
    """finals: {바구니: (S, H, K, C)}. special(job) → fake_main 인자 덮어쓰기."""
    outs = {}
    for job in jobs.itertuples():
        d = days_of(job, cal); s, h, k, c = finals[job.basket]
        if job.kind == 'main': outs[job.prefix] = fake_main(d, s, h, k, **(special(job) if special else {}))
        else: outs[job.prefix] = {'c': curve(d, c, 0.9), 'small': {'meta': {'skipped_at_start': ['Y', 'Z']}}}
    return outs


@test
def assembly_applies_gates_dependence_and_control():
    cal = t1.calendar(); jobs = cc.build_jobs(cal)
    finals = {2014: (11e6, 12e6, 13e6, 12e6), 2016: (11e6, 12e6, 10e6, 10.5e6), 2018: (13e6, 12e6, 14e6, 12e6), 2020: (13e6, 12e6, 11e6, 12e6),
              2022: (11e6, 12e6, 13e6, 12e6), 2024: (11e6, 12e6, 13e6, 12e6)}
    def special(job):
        out = {}
        if job.basket == 2022: out['sw'] = 0.70                                   # 비중 미달(모든 조건)
        if job.basket == 2024 and (job.k, job.div, job.delist) == sb.MAIN: out['trades'] = False     # 기본 조건만 분할 효과 보류
        return out
    outs = fake_outs(jobs, cal, finals, special)
    flip = first(jobs[(jobs.kind == 'main') & (jobs.basket == 2014) & (jobs.k == 20) & (jobs['div'] == 'none') & (jobs.delist == 'zero')])
    outs[flip.prefix] = fake_main(days_of(flip, cal), 12.5e6, 12e6, 13e6)    # 2014: 한 조건에서 a 부호가 바뀐다
    tables, s = cv.assemble(jobs, outs)
    t = s['table']
    assert t['cells'] == {'S<H·S<K': [2014], 'S<H·S>K': [2016], 'S>H·S<K': [2018], 'S>H·S>K': [2020]}, t
    assert t['cash_unresolved'] == [2022] and t['split_held'] == [2024] and s['counts']['dependent'] == [2014] and s['sentence']['kind'] == 'counts'
    rows = tables['summary_jobs']; assert len(rows) == 108 and len(tables['control']) == 108 and len(tables['summary_baskets']) == 6
    r = rows[rows.prefix == 'main_b2014_k0_approx_freeze'].iloc[0]
    yrs = (pd.Timestamp(r.end) - sb.prev_day(cal, r.start)).days / 365.25
    want = lambda v: ((v / 1e7) ** (1 / yrs) - 1) * 100
    assert abs(r.s_cagr - want(11e6)) < 1e-9 and abs(r.a - (want(11e6) - want(12e6))) < 1e-9 and abs(r.b - (want(11e6) - want(13e6))) < 1e-9
    assert abs(r.s_weight - 0.9) < 1e-12 and abs(r.h_weight - 0.95) < 1e-12 and r.weight_ok and not r.split_held
    assert (r.split_addbuys, r.split_takeprofits, r.split_tickers_bought, r.split_never_bought, r.core_not_bought, r.small_not_bought) == (1, 1, 1, 2, 1, 2)
    assert (r.split_never_order_below_one_share, r.split_never_never_tier1_candidate, r.split_never_other) == (1, 0, 1)
    b = tables['summary_baskets'].set_index('basket')
    assert b.loc[2016, 'c_flips_of_18'] == 18 and b.loc[2014, 'c_flips_of_18'] == 0       # 2016: S < H이지만 S > C
    assert b.loc[2022, 'n_cash_unresolved'] == 18 and b.loc[2024, 'n_split_held'] == 1 and not b.loc[2024, 'judged']
    assert len(tables['weight_daily']) == sum(len(days_of(j, cal)) - 1 for j in jobs[(jobs.kind == 'main') & (jobs.k == 0) & (jobs['div'] == 'approx') & (jobs.delist == 'freeze')].itertuples())
    clean = {y: (11e6, 12e6, 13e6, 12e6) for y in sb.YEARS}
    assert cv.assemble(jobs, fake_outs(jobs, cal, clean))[1]['sentence']['kind'] == 'all_neg'
    tmp = tempfile.mkdtemp()
    try:
        cv.publish(tmp, 'smoke', tables, s)
        assert sorted(os.listdir(tmp)) == ['smoke_tables', 'verdict_smoke.json'] and json.load(open(os.path.join(tmp, 'verdict_smoke.json')))['official'] is False
        cv.publish(tmp, 'formal', tables, s)
        md = open(os.path.join(tmp, 'VERDICT.md')).read(); v = json.load(open(os.path.join(tmp, 'verdict.json')))
        assert s['sentence']['text'] in md and rules.NOTE_OVERLAP in md and v['official'] is True and v['table']['cash_unresolved'] == [2022]
    finally: shutil.rmtree(tmp)


# ---------------- 조립기 거부(실제 실행, DB는 달력만) ----------------
def run_verdict(out, mode, *extra):
    p = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'same_basket_core_verdict.py'), out, '--mode', mode, *extra], capture_output=True, text=True)
    vdir = os.path.join(out, 'verdict')
    files = sorted(os.listdir(out)) + (sorted(os.listdir(vdir)) if os.path.isdir(vdir) else [])
    path = os.path.join(vdir, 'checks.json')
    return p.returncode, files, json.load(open(path))['errors'] if os.path.exists(path) else None, p.stdout


def status_file(out, jobs, fail=None, drop=None):
    rows = [(j.id, j.prefix, 1 if j.prefix == fail else 0, 1, 't') for j in jobs.itertuples() if j.prefix != drop]
    pd.DataFrame(rows, columns=['id', 'prefix', 'exit', 'seconds', 'finished_utc']).to_csv(os.path.join(out, 'job_status.tsv'), sep='\t', index=False)


def verdict_refuses_end_to_end():
    import stage2_common as sc
    cal = sc.trading_dates(sc.connect()); jobs = cc.build_jobs(cal); tmp = tempfile.mkdtemp()
    published = {'verdict.json', 'VERDICT.md', 'summary_jobs.csv', 'summary_baskets.csv', 'control.csv', 'weight_daily.csv', 'smoke_tables'}
    try:
        shutil.copytree(cc.BASKET_DIR, os.path.join(tmp, 'baskets')); jobs.to_csv(os.path.join(tmp, 'jobs.tsv'), sep='\t', index=False)
        for rel in ('verdict.json', 'VERDICT.md', 'verdict/verdict.json', 'verdict/VERDICT.md', 'verdict/summary_jobs.csv'):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True); open(os.path.join(tmp, rel), 'w').write('old official verdict')
        cases = {'failed job': dict(fail=jobs.prefix[7]), 'unrecorded job': dict(drop=jobs.prefix[150]), 'all ok but no reproduction record and no outputs': {}}
        for name, kw in cases.items():
            status_file(tmp, jobs, **kw); rc, files, errs, stdout = run_verdict(tmp, 'formal')
            assert rc == 2 and errs and not published & set(files) and stdout == '', (name, rc, files)
        assert any('reproduction check' in e for e in errs)
        os.makedirs(os.path.join(tmp, 'repro')); json.dump({'passed': True}, open(os.path.join(tmp, 'repro', 'repro_check.json'), 'w'))
        rc, files, errs, _ = run_verdict(tmp, 'formal')
        assert rc == 2 and len(errs) == len(jobs) and all('cannot read outputs' in e for e in errs) and not published & set(files)
        jobs.iloc[:-1].to_csv(os.path.join(tmp, 'jobs.tsv'), sep='\t', index=False); status_file(tmp, jobs.iloc[:-1])
        rc, files, errs, _ = run_verdict(tmp, 'formal')
        assert rc == 2 and any('job rows differ' in e for e in errs) and any('ctrl: 107 jobs' in e for e in errs)
        rc, files, errs, _ = run_verdict(tmp, 'formal', '--fail', 'inputs differ')
        assert rc == 2 and errs == ['run_all: inputs differ'] and files[-1:] == ['checks.json'] and not published & set(files)
        os.remove(os.path.join(tmp, 'job_status.tsv')); rc, files, errs, _ = run_verdict(tmp, 'smoke')
        assert rc == 2 and errs == ['job_status.tsv unreadable (FileNotFoundError)']
    finally: shutil.rmtree(tmp)


def main():
    tests = TESTS + ([] if '--no-db' in sys.argv else [verdict_refuses_end_to_end])
    failed = []
    for fn in tests:
        try: fn(); print('ok', fn.__name__)
        except Exception as exc:
            failed.append(fn.__name__); print('FAIL', fn.__name__, type(exc).__name__, str(exc)[:300])
    print(f'{len(tests) - len(failed)} tests passed, {len(failed)} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
