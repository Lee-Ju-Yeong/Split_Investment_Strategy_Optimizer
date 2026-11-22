"""연속 계좌 진단(a) 판정 규칙 단위 검사(가상 표만 사용, DB·실행 결과 없음). Astra 검토의 반례를 포함한다.
usage: python -B tools/research/continuous_account_selftest.py   (실패 시 AssertionError, 종료 코드 1)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import continuous_account_rules as r
import continuous_account_verdict as v


def test_astra_counterexample_both_fail():
    # 연복리 차이 +2.1%p(세 시작일), 두 계좌 모두 ETF 미통과, 구간 통과 수 2/2 -> 뒤집힘 아님, 효과 크기만 large
    assert r.gap_class([2.1, 2.1, 2.1]) == 'large'
    assert r.count_label(2, 2) == ('unchanged_fail', 'same')
    assert r.bool_label(False, False) == 'unchanged_fail'
    assert r.final_verdict({0: 'unchanged_fail', 5: 'unchanged_fail', 20: 'unchanged_fail'}, {'approx': 'unchanged_fail'}) == 'stage2_verdict_unchanged'


def test_astra_counterexample_small_gap_but_flip():
    # 차이 +0.5%p라도 재시작 3, 연속 2이면 뒤집힘
    assert r.gap_class([0.5, 0.5, 0.5]) == 'small'
    assert r.count_label(3, 2) == ('flipped', 'down')
    assert r.final_verdict({0: 'flipped', 5: 'flipped', 20: 'flipped'}, {'none': 'flipped', 'approx': 'flipped', 'upper': 'flipped'}) == 'stage2_verdict_flipped'


def test_count_decrease_without_flip():
    assert r.count_label(2, 1) == ('unchanged_fail', 'down')
    assert r.count_label(4, 3) == ('unchanged_pass', 'down')
    assert r.count_label(2, 3) == ('reverse_flipped', 'up')


def test_gap_boundaries_and_reverse():
    assert r.gap_class([2.0, 2.0, 2.0]) == 'large'
    assert r.gap_class([1.0, 0.2, 0.1]) == 'mixed'
    assert r.gap_class([-2.5, -3.0, -2.0]) == 'reverse_large'
    assert r.gap_class([2.5, 0.5, 2.5]) == 'mixed'


def test_start_date_and_dividend_sensitivity():
    assert r.final_verdict({0: 'flipped', 5: 'unchanged_pass', 20: 'flipped'}, {'approx': 'flipped'}) == 'depends_on_start_date'
    assert r.final_verdict({0: 'flipped', 5: 'flipped', 20: 'flipped'}, {'none': 'unchanged_fail', 'approx': 'flipped', 'upper': 'flipped'}) == 'dividend_sensitive_undetermined'


def test_job_structure_requires_all_offsets_and_anchor():
    rows = [{'kind': 'cont', 'k': 0, 'div': 'approx', 'window': 'all', 'prefix': 'c0'}] + \
           [{'kind': 'restart', 'k': 0, 'div': 'approx', 'window': str(i), 'prefix': f'r{i}'} for i in range(6)]
    errs = r.check_job_structure(rows)
    assert any('anchor' in e for e in errs) and any('k=5' in e for e in errs) and any('k=20' in e for e in errs) and any('div=none' in e for e in errs)
    full = [{'kind': 'anchor', 'k': 0, 'div': 'none', 'window': 'a', 'prefix': 'a'}]
    for k in r.OFFSETS:
        for d in (r.DIVS_K0 if k == 0 else ('approx',)):
            full.append({'kind': 'cont', 'k': k, 'div': d, 'window': 'all', 'prefix': f'c{k}{d}'})
            full += [{'kind': 'restart', 'k': k, 'div': d, 'window': str(i), 'prefix': f'r{k}{d}{i}'} for i in range(6)]
    assert r.check_job_structure(full) == []
    assert any('duplicate' in e for e in r.check_job_structure(full + [full[1]]))
    assert any('windows' in e for e in r.check_job_structure(full[:-1]))


def test_mechanism_relation():
    assert r.relation(1.0, 0.5, 0.01, 0.05) == 'same'
    assert r.relation(1.0, -0.5, 0.01, 0.05) == 'opposite'
    assert r.relation(0.0, 0.0, 0.01, 0.05) == 'zero'
    assert r.relation(1.0, 0.01, 0.01, 0.05) == 'zero'


def test_judge_rules():
    e = {'cagr': 5.0, 'mdd': -20.0, 'calmar': 0.25}
    assert r.judge({'cagr': 6.0, 'mdd': -25.0, 'calmar': 0.24}, e)['pass']
    assert not r.judge({'cagr': 5.9, 'mdd': -20.0, 'calmar': 0.3}, e)['pass']
    x = r.judge({'cagr': 10.0, 'mdd': -30.0, 'calmar': 1 / 3}, e)
    assert not x['pass'] and x['pass_calmar']


def _win(values, recv, pay=None):
    idx = pd.bdate_range('2020-01-01', periods=len(values))
    df = pd.DataFrame({'value_engine': values, 'receivable': recv, 'cash': 0.0}, index=idx); df['value'] = df.value_engine + df.receivable
    return {'df': df, 'b': {'receivable': float(recv[-1]), 'recv_pay': pay, 'est_sell_cost': 0.0}}


def test_chain_variants():
    w1 = _win([1e7, 1.1e7], [0.0, 1e6])                     # 끝에 미수금 100만
    idx2 = pd.bdate_range('2020-01-03', periods=3)
    df2 = pd.DataFrame({'value_engine': [1e7, 1.2e7, 1.2e7], 'receivable': [0.0, 0.0, 0.0], 'cash': 0.0}, index=idx2); df2['value'] = df2.value_engine
    w1['b']['recv_pay'] = idx2[1]
    w2 = {'df': df2, 'b': {'receivable': 0.0, 'recv_pay': None, 'est_sell_cost': 0.0}}
    s2 = v.chain([w1, w2], 'stage2_style')[0]; rp = v.chain([w1, w2], 'recv_par')[0]
    assert np.isclose(s2.iloc[-1], 1.2e7 * 1.2)             # 미수금 포함 120만 배를 다음 구간 수익률로 굴림
    assert np.isclose(rp.iloc[2], 1.0e7 * 1.1 + 1e6)        # 다음 구간 첫날: 미수금 액면
    assert np.isclose(rp.iloc[-1], 1.1e7 * 1.2 + 1e6)       # 지급일 뒤 그 구간 수익률(1.0)로 재투자
    w1c = {**w1, 'b': {**w1['b'], 'est_sell_cost': 5e4}}
    lc = v.chain([w1c, w2], 'liqcost')[0]
    assert np.isclose(lc.iloc[1], 1.2e7 - 5e4)               # 경계일 종가에서 같은 날 차감
    assert np.isclose(lc.iloc[-1], (1.2e7 - 5e4) * 1.2)       # 마지막 구간 끝은 청산하지 않음


def _w(start, engine, recv, pay=None, cash=None):
    idx = pd.bdate_range(start, periods=len(engine))
    df = pd.DataFrame({'value_engine': engine, 'receivable': recv, 'cash': cash if cash is not None else [0.0] * len(engine)}, index=idx)
    df['value'] = df.value_engine + df.receivable
    return {'df': df, 'b': {'receivable': float(recv[-1]), 'recv_pay': pay, 'est_sell_cost': 0.0}}


def test_recv_par_unpaid_across_two_boundaries():
    # Astra 1-1: 원금 100·미수 10, 지급 전 다음 구간 +10% -> 120(기존 코드 121)
    w1 = _w('2020-01-01', [1e7, 1e7], [0.0, 1e6], pay=pd.Timestamp('2020-02-05'))
    w2 = _w('2020-01-03', [1e7, 1.1e7], [0.0, 0.0])
    w3 = _w('2020-02-03', [1e7, 1e7, 1e7, 1.2e7], [0.0] * 4)          # 2020-02-05(셋째 날) 지급
    vals, inv = v.chain_recv_par([w1, w2, w3])
    assert np.isclose(vals.loc['2020-01-06'], 1.2e7)                  # 두 번째 경계: 투자 110 + 미수 10(액면)
    assert np.isclose(vals.loc['2020-02-04'], 1.1e7 + 1e6)            # 지급 전날: 아직 액면
    assert np.isclose(vals.loc['2020-02-05'], 1.1e7 + 1e6)            # 지급일 종가에 지분 매입(값 연속)
    assert np.isclose(vals.loc['2020-02-06'], (1.1 + 0.1) * 1.2e7)    # 다음 날부터 재투자 지분도 수익 노출
    assert np.isclose(inv.loc['2020-01-06'], 1.1e7 / 1.2e7)           # 투자 비중 분모 = 미수 포함 이어 붙인 평가액


def test_recv_par_reinvested_units_get_dividend_rights():
    # Astra 1-2: 미수 10을 기본 계좌 100일 때 재투자 -> 지분 1.1. 구간 말 기본 계좌 엔진 100·새 미수 20
    #  -> 올바른 이월: 투자 110, 미수 22. 다음 구간 지급 전 +10% -> 143
    w1 = _w('2020-01-01', [1e7, 1e7], [0.0, 1e6], pay=pd.Timestamp('2020-01-03'))
    w2 = _w('2020-01-03', [1e7, 1e7], [0.0, 2e6], pay=pd.Timestamp('2020-01-10'))
    w3 = _w('2020-01-07', [1e7, 1.1e7, 1.1e7, 1.1e7], [0.0] * 4)
    vals, _ = v.chain_recv_par([w1, w2, w3])
    assert np.isclose(vals.loc['2020-01-06'], 1.1 * 1.2e7)
    assert np.isclose(vals.loc['2020-01-07'], 1.1e7 + 2.2e6)
    assert np.isclose(vals.loc['2020-01-08'], 1.43e7)


def test_liqcost_at_peak_boundary_drawdown():
    w1 = _w('2020-01-01', [1e7, 1.2e7], [0.0, 0.0]); w1['b']['est_sell_cost'] = 1e5
    w2 = _w('2020-01-03', [1e7, 1e7], [0.0, 0.0])
    vals, _ = v.chain_simple([w1, w2], liqcost=True)
    assert np.isclose(vals.iloc[1], 1.19e7) and np.isclose(vals.iloc[-1], 1.19e7)
    p = v.perf(vals, 1e7, pd.Timestamp('2019-12-31'))
    assert p['mdd'] == 0.0                                            # 비용을 경계일에 잡아 고점이 부풀지 않음


def test_boundary_sensitivity_single_offset_and_2025_only():
    lab = lambda a, b, c: {'stage2_style': a, 'recv_par': b, 'liqcost': c}
    tab = {(k, 'approx', p, 'pass'): lab('unchanged_fail', 'unchanged_fail', 'unchanged_fail') for k in (0, 5, 20) for p in ('to2023', 'to2025')}
    assert r.boundary_sensitivity(tab) == []
    tab[(5, 'approx', 'to2025', 'pass')] = lab('flipped', 'unchanged_fail', 'flipped')
    pos = r.boundary_sensitivity(tab)
    assert len(pos) == 1 and pos[0]['k'] == 5 and pos[0]['period'] == 'to2025'


def test_primary_joint():
    assert r.PRIMARY_CRITERIA == ('pass', 'pass_calmar') and r.primary_errors() == []
    assert r.primary_final({'pass': 'a', 'pass_calmar': 'b'}) == {'original': 'a', 'calmar': 'b'}   # OR로 합치지 않음
    assert r.primary_final({'pass': 'a', 'pass_calmar': 'b'}, primary=('undecided',)) == 'pending_user_decision'
    assert r.primary_errors(('pass', 'other')) != []
    assert r.STAGE2_NOTE.startswith('원래 기준에서는 2단계도 2/5로 미통과였다.')


def test_job_rows_exact_and_argv():
    exp = [{'kind': 'cont', 'k': 0, 'window': 'all', 'start': '2014-01-02', 'end': '2025-12-30', 'tax': 'period', 'div': 'approx', 'prefix': 'c'}]
    assert r.compare_job_rows(exp, exp) == []
    assert r.compare_job_rows([{**exp[0], 'tax': 'fixed'}], exp) != []
    assert r.compare_job_rows(exp + [{**exp[0], 'kind': 'odd', 'prefix': 'x'}], exp) != []
    argv = ['2014-01-02', '2025-12-30', '/x/c', '--tax', 'period', '--div', 'approx', '--hold', 'none']
    assert r.check_argv(exp[0], argv) == []
    assert r.check_argv(exp[0], argv[:4] + ['fixed'] + argv[5:]) != []


def test_etf_price_checks():
    idx = pd.bdate_range('2020-01-01', periods=3)
    good = pd.DataFrame({'open': [1.0, 1.0, 1.0], 'close': [1.0, 1.0, 1.0]}, index=idx)
    assert v.check_etf(good, idx, __file__, None) == []
    bad = good.copy(); bad.iloc[1, 0] = np.nan; bad.iloc[2, 1] = np.inf
    assert any('infinite' in e for e in v.check_etf(bad, idx, __file__, None))
    dup = pd.concat([good, good.iloc[[0]]])
    assert any('duplicate' in e for e in v.check_etf(dup, idx, __file__, None))
    assert any('missing' in e for e in v.check_etf(good.iloc[:2], idx, __file__, None))


def test_stage2_classify_and_link():
    import continuous_account_stage2_check as s2
    base = {'dates_equal': True, 'first_curve': None, 'first_fill': None, 'divs_equal': True, 'pay_old': [], 'pay_new': [], 'fallback_dates': []}
    assert s2.classify(base) == 'identical'
    assert s2.classify({**base, 'dates_equal': False, 'pay_old': ['2015-05-04'], 'pay_new': ['2015-05-18']}) == 'unexplained'   # Astra 반례
    pay = {'pay_old': ['2015-05-04'], 'pay_new': ['2015-05-18'], 'divs_equal': False}
    assert s2.classify({**base, **pay, 'first_curve': '2015-05-06', 'first_fill': '2015-05-06'}) == 'pay_date_rule'
    assert s2.classify({**base, **pay, 'first_curve': '2015-03-02'}) == 'unexplained'                    # 지급일 전 차이
    assert s2.classify({**base, 'divs_equal': False, 'fallback_dates': ['2015-12-29'], 'first_curve': '2015-12-30'}) == 'dps_fallback'
    assert s2.classify({**base, 'first_curve': '2015-02-02'}) == 'unexplained'
    idx = pd.bdate_range('2020-01-01', periods=3)
    assert s2.first_curve_diff(pd.Series([1e7, 1e7, 1e7], idx), pd.Series([1e7, 1e7 + 90, 1e7], idx)) == '2020-01-02'   # 90원 차이도 다름
    row = lambda w, d, a, b, c, cause='identical': {'window': w, 'div': d, 'cause': cause, 'pass_archived': a, 'pass_new_run_old_method': b, 'pass_new_run_new_method': c}
    rows = [row(w, d, w in ('2014', '2018'), w in ('2014', '2018'), w in ('2014', '2018')) for d in s2.DIVS for w in s2.FIVE]
    assert s2.link_status(rows, 'pass', 15, True) == ('linked', '')
    assert s2.link_status(rows[:-1], 'pass', 15, True)[0] == 'withheld'
    assert s2.link_status(rows, 'pass', 15, False)[0] == 'withheld'
    changed = [dict(x) for x in rows]; changed[1]['pass_new_run_new_method'] = True                   # 통과 구간 변화
    assert s2.link_status(changed, 'pass', 15, True)[0] == 'withheld'
    une = [dict(x) for x in rows]; une[0]['cause'] = 'unexplained'
    assert s2.link_status(une, 'pass', 15, True)[0] == 'withheld'


# ---------- v2.2 축소 검사 ----------
def _divs(hold, recv=(), paid=()):
    """hold: [(fy, record_date, ticker, adj_qty, amount, fallback)], recv/paid: [(fy, date, amount)]"""
    rows = [{'fy': fy, 'record_date': rd, 'ticker': tk, 'adj_qty': q, 'raw_qty': q, 'dps': 1.0, 'fallback': fb, 'amount': am, 'event': None, 'date': None}
            for fy, rd, tk, q, am, fb in hold]
    rows += [{'fy': fy, 'event': 'receivable', 'date': d, 'amount': am} for fy, d, am in recv]
    rows += [{'fy': fy, 'event': 'paid', 'date': d, 'amount': am} for fy, d, am in paid]
    return pd.DataFrame(rows)


def test_v22_dividend_diff_before_pay_shift_is_unexplained():
    import continuous_account_stage2_check as s2
    hold = [(2014, '2014-12-29', '000001', 10.0, 1000.0, False)]
    old = _divs(hold, recv=[(2014, '2014-12-30', 1000.0)], paid=[(2014, '2015-05-04', 1000.0)])
    # ④ 12월 미수 금액이 이미 다르고(대체 아님), 곡선 첫 차이는 5월 지급일 뒤 -> 예전엔 pay_date_rule, 이제 unexplained
    new = _divs([(2014, '2014-12-29', '000001', 10.0, 1500.0, False)], recv=[(2014, '2014-12-30', 1500.0)], paid=[(2014, '2015-05-18', 1500.0)])
    d = s2.dividend_diffs(old, new)
    assert d['first_hold'] == '2014-12-29' and d['first_recv'] == '2014-12-30'
    p = {'dates_equal': True, 'first_curve': '2015-05-06', 'first_fill': '2015-05-06', **d}
    assert s2.classify(p) == 'unexplained'
    # 정상: 같은 금액·지급일만 이동 -> pay_date_rule
    new2 = _divs(hold, recv=[(2014, '2014-12-30', 1000.0)], paid=[(2014, '2015-05-18', 1000.0)])
    d2 = s2.dividend_diffs(old, new2)
    assert d2['pay_old'] == ['2015-05-04'] and d2['pay_new'] == ['2015-05-18'] and not any(d2[k] for k in ('first_hold', 'first_recv', 'first_paid'))
    assert s2.classify({'dates_equal': True, 'first_curve': '2015-05-06', 'first_fill': None, **d2}) == 'pay_date_rule'
    # 대체로 설명: 새 실행에만 fallback 종목이 있고 그 FY 미수·지급 금액이 다름 -> dps_fallback
    new3 = _divs(hold + [(2014, '2014-12-29', '000002', 5.0, 300.0, True)], recv=[(2014, '2014-12-30', 1300.0)], paid=[(2014, '2015-05-04', 1300.0)])
    d3 = s2.dividend_diffs(old, new3)
    assert d3['fallback_dates'] == ['2014-12-29'] and not any(d3[k] for k in ('first_hold', 'first_recv', 'first_paid'))
    assert s2.classify({'dates_equal': True, 'first_curve': '2014-12-30', 'first_fill': None, **d3}) == 'dps_fallback'
    # 대체 종목 외의 보유 차이가 함께 있으면 unexplained
    new4 = _divs([(2014, '2014-12-29', '000001', 12.0, 1200.0, False), (2014, '2014-12-29', '000002', 5.0, 300.0, True)],
                 recv=[(2014, '2014-12-30', 1500.0)], paid=[(2014, '2015-05-04', 1500.0)])
    d4 = s2.dividend_diffs(old, new4)
    assert d4['first_hold'] == '2014-12-29'
    assert s2.classify({'dates_equal': True, 'first_curve': '2014-12-30', 'first_fill': None, **d4}) == 'unexplained'


def test_v22_fifteen_pairs_link_per_criterion():
    import continuous_account_stage2_check as s2
    rows = []
    for d in s2.DIVS:
        for w in s2.FIVE:
            r = {'window': w, 'div': d, 'cause': 'pay_date_rule' if d != 'none' else 'identical'}
            for st in s2.STAGES:
                r[f'pass_{st}'] = w in ('2014', '2018')
                r[f'pass_calmar_{st}'] = w in ('2014', '2018', '2020') if st != 'new_run_new_method' else w in ('2014', '2018')
            rows.append(r)
    o, c = s2.criterion_summary(rows, 'pass', 15, True), s2.criterion_summary(rows, 'pass_calmar', 15, True)
    assert o['status'] == 'linked' and c['status'] == 'withheld' and 'pass windows change' in c['reason']


def _direct_stub():
    return {'final_original': 'stage2_verdict_unchanged', 'final_calmar': 'stage2_verdict_flipped',
            'pass': {'final': 'stage2_verdict_unchanged'}, 'pass_calmar': {'final': 'stage2_verdict_flipped'}}


def test_v22_direct_verdict_survives_broken_stage2():
    import json, tempfile
    for case in ('truncated', 'exit_nonzero', 'missing', 'bad_structure', 'ok'):
        d = tempfile.mkdtemp(dir=os.environ.get('SELFTEST_TMP'))
        good = {'by_criterion': {'original': {'status': 'linked'}, 'calmar': {'status': 'withheld', 'reason': 'x'}}}
        if case == 'truncated': open(os.path.join(d, 'stage2_reconcile.json'), 'w').write('{"by_criterion": {"orig')
        if case == 'exit_nonzero':
            json.dump(good, open(os.path.join(d, 'stage2_reconcile.json'), 'w')); open(os.path.join(d, 'stage2_check.exit'), 'w').write('1\n')
        if case == 'bad_structure': json.dump({'by_criterion': {'original': {'status': 'yes'}}}, open(os.path.join(d, 'stage2_reconcile.json'), 'w'))
        if case == 'ok':
            json.dump(good, open(os.path.join(d, 'stage2_reconcile.json'), 'w')); open(os.path.join(d, 'stage2_check.exit'), 'w').write('0\n')
        v.publish(d, 'formal', _direct_stub())
        assert json.load(open(os.path.join(d, 'direct_verdict.json')))['final_calmar'] == 'stage2_verdict_flipped', case
        out = json.load(open(os.path.join(d, 'verdict.json')))['by_criterion']
        if case == 'ok':
            assert out['original']['stage2_link'] == 'linked' and out['calmar']['stage2_link'] == 'withheld'
            assert out['original']['stage2_sentence'] == v.SENTENCES['stage2_verdict_unchanged'] and out['calmar']['stage2_sentence'] == v.WITHHELD
        else:
            assert all(x['stage2_link'] == 'withheld' and x['stage2_sentence'] == v.WITHHELD for x in out.values()), case
        assert not [f for f in os.listdir(d) if f.endswith('.tmp')]


def test_v22_liqcost_boundary_exposure_pre_liquidation():
    w1 = _w('2020-01-01', [1e7, 1.2e7], [0.0, 0.0], cash=[0.0, 0.0]); w1['b']['est_sell_cost'] = 1e5
    w2 = _w('2020-01-03', [1e7, 1e7], [0.0, 0.0], cash=[1e7, 0.0])
    _, inv = v.chain_simple([w1, w2], liqcost=True)
    assert np.isclose(inv.iloc[1], 1.0) and inv.max() <= 1.0 + 1e-12      # 전액 투자 계좌가 100%를 넘지 않음


def test_v22_run_all_copy_verification():
    """① 실행 사본을 승인 목록 복사본으로 검증: 정상은 통과·실행 표시, 사본 변조·재사용·formal 시험 훅은 거부(실행 없음)."""
    import subprocess, tempfile, shutil
    here = os.path.dirname(os.path.abspath(__file__)); sh = os.path.join(here, 'continuous_account_run_all.sh')
    src = '/mnt/volume/workspace/split-research/repo/tools/research'
    names = ['continuous_account_jobs.py', 'continuous_account_rules.py', 'continuous_account_run.py', 'continuous_account_run_all.sh',
             'continuous_account_selftest.py', 'continuous_account_stage2_check.py', 'continuous_account_verdict.py', 'split_window_runs.py', 'stage2_common.py']
    parent = tempfile.mkdtemp(dir=os.environ.get('SELFTEST_TMP')); ap = os.path.join(parent, 'approved'); os.makedirs(ap)
    open(os.path.join(parent, 'DESIGN-v2.2.md'), 'w').write('test design\n')
    sha = lambda p: __import__('hashlib').sha256(open(p, 'rb').read()).hexdigest()
    open(os.path.join(ap, 'code.sha256'), 'w').write(''.join(f'{sha(os.path.join(src, n))}  {n}\n' for n in names))
    open(os.path.join(ap, 'design.sha256'), 'w').write(f"{sha(os.path.join(parent, 'DESIGN-v2.2.md'))}  DESIGN-v2.2.md\n")
    def run(name, mode='smoke', **env):
        out = os.path.join(parent, name); os.makedirs(out, exist_ok=True)
        if not os.path.exists(os.path.join(out, 'jobs.tsv')): open(os.path.join(out, 'jobs.tsv'), 'w').write('kind\n')
        e = {**os.environ, 'SKIP_DB_CHECKSUM': '1' if mode == 'smoke' else '0', **env}
        r = subprocess.run(['bash', sh, out, '1', mode], env=e, capture_output=True, text=True)
        return r, out
    r, out = run('ok', STOP_AFTER_COPY_VERIFY='1')
    assert r.returncode == 0 and os.path.exists(os.path.join(out, '.opened')), r.stderr
    assert 'copies_verified_against' in open(os.path.join(out, 'provenance', 'copy_check.txt')).read()
    r2, _ = run('ok', STOP_AFTER_COPY_VERIFY='1'); assert r2.returncode != 0 and 'fresh folder' in r2.stderr
    hook = "echo '# tampered' >> \"$OUT/code/continuous_account_rules.py\""
    r3, out3 = run('tampered', STOP_AFTER_COPY_VERIFY='1', TEST_HOOK_AFTER_COPY=hook.replace('$OUT', os.path.join(parent, 'tampered')))
    assert r3.returncode != 0 and 'execution copies differ' in r3.stderr, r3.stderr
    hook_d = "echo x >> \"%s/code/DESIGN-v2.2.md\"" % os.path.join(parent, 'tampered_design')
    r4, _ = run('tampered_design', STOP_AFTER_COPY_VERIFY='1', TEST_HOOK_AFTER_COPY=hook_d)
    assert r4.returncode != 0 and 'design differs' in r4.stderr, r4.stderr
    r5, _ = run('formal_hook', mode='formal', STOP_AFTER_COPY_VERIFY='1')
    assert r5.returncode != 0 and 'test hooks are not allowed' in r5.stderr, r5.stderr
    assert not os.path.exists(os.path.join(parent, 'tampered', 'runs', 'x')) and not os.listdir(os.path.join(parent, 'ok', 'runs'))
    shutil.rmtree(parent)


if __name__ == '__main__':
    tests = [f for n, f in sorted(globals().items()) if n.startswith('test_')]
    for f in tests: f()
    print(f'ok: {len(tests)} tests')
