"""주문율 격자(same-basket-size-r1) 자체 검사: 합성 자료로 작업표·자금 지표·기준·경계 구간·판정 규칙·문장·주문액 검사·r1 대조·조립 거부를 확인한다.

usage: python -B tools/research/same_basket_size_selftest.py [--no-db]
3월 실행 체크아웃을 작업 디렉터리로 실행한다. --no-db가 없으면 조립기를 임시 폴더에서 실제로 돌려 거부(종료 코드 2, 표·문장 없음)를 확인한다
(DB는 거래일 목록만 읽고, 명단은 r1 명단 파일을 복사해 쓴다). 실제 성과 자료는 읽지 않는다. 모두 통과하면 'N tests passed'.
"""
import json, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import same_basket_common as sb
import same_basket_size_common as sz
import same_basket_size_funds as funds
import same_basket_size_rules as rules
import same_basket_size_checks as szc

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = []


def test(fn):
    TESTS.append(fn); return fn


def calendar():
    return pd.bdate_range('2013-01-02', '2026-01-09')


# ---------------- 작업표 ----------------
@test
def job_table_has_756_rows_r1x_first_and_maps_to_r1():
    jobs = sz.build_jobs(calendar())
    assert len(jobs) == 756 and sz.check_job_structure(jobs) == []
    assert (jobs.kind.iloc[:108] == 'r1x').all() and (jobs.kind.iloc[108:] == 'size').all()
    assert list(dict.fromkeys(jobs.ratio)) == ['0.022', '0.03', '0.04', '0.05', '0.06', '0.08', '0.1']
    r1 = [r[9] for y in sb.YEARS for r in sb.main_rows(calendar(), y)]
    for ratio, g in jobs.groupby('ratio', sort=False):
        assert list(g.r1prefix) == r1, ratio
        assert all(p == f'{sz.ratio_name(float(ratio))}_' + q[len('main_'):] for p, q in zip(g.prefix, g.r1prefix))
    assert sz.ratio_name(0.1) == 'r100' and sz.ratio_name(0.022) == 'r022' and sz.ratio_text(0.022) == '2.2%'
    assert sz.compare_jobs(jobs.iloc[:-1], jobs) and sz.compare_jobs(jobs, jobs) == []
    assert sz.check_job_structure(jobs.iloc[1:]) == ['r1x: 107 jobs (need 108)']
    smoke = sz.build_smoke_jobs(calendar(), 2014, '2014-02-28')
    assert len(smoke) == 24 and set(smoke.end) == {'2014-02-28'} and set(smoke.k) == {0}


# ---------------- 자금 지표 ----------------
def synth_x(n=10, invest=0.95, short_days=2, add_share=0.3, profit=1, value0=sb.CASH0, start='2020-01-02'):
    """시작 전날 + n일. 현금 = 순자산 × (1 − invest), 주문액은 처음 short_days일만 현금보다 크다."""
    days = pd.bdate_range(start, periods=n + 1)
    value = pd.Series(np.linspace(value0, value0 * 1.1, n + 1), index=days); value.iloc[0] = value0
    cash = value * (1 - invest); cash.iloc[0] = value0
    curve = pd.DataFrame({'value_engine': value, 'receivable': 0.0, 'value': value})
    state = pd.DataFrame({'engine_value': value, 'cash': cash, 'n_tickers': 1, 'n_lots': 1})
    ipo = cash + np.where(np.arange(n + 1) <= short_days, 1.0, -1.0)
    log = pd.DataFrame({'date': days, 'recomputed': False, 'basis': np.nan, 'investment_per_order': ipo.to_numpy()})
    new, add = 1000.0, 1000.0 * add_share / (1 - add_share) if add_share < 1 else 1000.0
    fills = [{'date': days[1], 'ticker': 'A00001', 'type': 'BUY', 'dq': 1, 'dcash': -new, 'order': 1, 'reason': funds.REASON_NEW}]
    if add_share > 0: fills.append({'date': days[2], 'ticker': 'A00001', 'type': 'BUY', 'dq': 1, 'dcash': -add, 'order': 2, 'reason': funds.REASON_ADD})
    for _ in range(profit): fills.append({'date': days[3], 'ticker': 'A00001', 'type': 'SELL', 'dq': -1, 'dcash': 900.0, 'order': 2, 'reason': funds.REASON_PROFIT})
    return curve, state, pd.DataFrame(fills), log


@test
def fund_metrics_match_hand_calculation():
    curve, state, fills, log = synth_x(n=10, invest=0.95, short_days=2, add_share=0.3, profit=2)
    m = funds.fund_metrics(curve, state, fills, log, ['A00001', 'B00002', 'C00003'])
    assert np.isclose(m['avg_invested'], 0.95) and np.isclose(m['first_day_invested'], 0.95) and np.isclose(m['min_cash_share'], 0.05)
    assert m['short_days_ratio'] == 0.2 and m['n_days'] == 10                         # 시작 전날 제외 10일 중 2일
    assert np.isclose(m['add_buy_share'], 0.3) and m['n_profit_sells'] == 2 and m['n_bought'] == 1 and m['n_never_bought'] == 2
    c2, s2, f2, l2 = synth_x(add_share=0, profit=0)
    m2 = funds.fund_metrics(c2, s2, f2.iloc[0:0], l2, ['A00001'])
    assert np.isnan(m2['add_buy_share']) and m2['n_bought'] == 0 and m2['n_never_bought'] == 1
    try: funds.fund_metrics(curve, state, fills, log.iloc[:-1], ['A00001']); raise AssertionError('missing log day accepted')
    except KeyError: pass


@test
def criteria_thresholds_are_inclusive_and_split_hold_label():
    ok = {'avg_invested': 0.90, 'short_days_ratio': 0.20, 'add_buy_share': 0.20, 'n_profit_sells': 1}
    assert funds.criteria(ok)['all_ok'] and funds.criteria(ok)['label'] == ''
    for key, bad in (('avg_invested', 0.8999999), ('short_days_ratio', 0.2000001), ('add_buy_share', 0.1999999), ('n_profit_sells', 0), ('avg_invested', float('nan')), ('add_buy_share', None)):
        c = funds.criteria({**ok, key: bad}); assert not c['all_ok'], (key, bad)
    assert funds.criteria({**ok, 'n_profit_sells': 0})['label'] == funds.SPLIT_HOLD
    assert funds.criteria({**ok, 'add_buy_share': 0.1})['label'] == funds.SPLIT_HOLD
    assert funds.criteria({**ok, 'avg_invested': 0.5})['label'] == '기준 미달'


@test
def exact_investment_threshold_passes_from_daily_ledger():
    curve, state, fills, log = synth_x(n=465)
    curve[['value_engine', 'value']] = 10_000_000
    state['cash'] = [10_000_000, *([1_500_000] * 232), *([500_000] * 232), 1_000_000]
    m = funds.fund_metrics(curve, state, fills, log, ['A00001'])
    assert funds.criteria(m)['invest_ok']
    state['cash'] += 1
    m = funds.fund_metrics(curve, state, fills, log, ['A00001'])
    assert not funds.criteria(m)['invest_ok']


@test
def boundary_needs_three_consecutive_grid_values():
    R = sz.RATIOS
    p = lambda bits: list(zip(R, bits))
    assert funds.boundary_runs(p([0, 1, 1, 1, 0, 0, 0])) == [[0.03, 0.04, 0.05]]
    assert funds.boundary_runs(p([1, 1, 0, 1, 1, 0, 1])) == []
    assert funds.boundary_runs(p([1, 1, 1, 0, 1, 1, 1])) == [[0.022, 0.03, 0.04], [0.06, 0.08, 0.10]]
    assert funds.boundary_runs(p([0, 0, 0, 0, 1, 1, 1])) == [[0.06, 0.08, 0.10]]
    ok = {(r, y): True for r in R for y in sb.YEARS}; ok[(0.05, 2018)] = False; del ok[(0.06, 2020)]
    passes = dict(funds.ratio_passes(ok, R, sb.YEARS))
    assert passes[0.04] and not passes[0.05] and not passes[0.06]


# ---------------- 판정 규칙 ----------------
def v(a=1.0, b=1.0, ok=True):
    return {'a': a, 'b': b, 'ok': ok}


def ratio_data(base_by_y, var_by_y=None):
    var_by_y = var_by_y or {y: [base_by_y[y]] * 18 for y in base_by_y}
    return {'base': base_by_y, 'variants': var_by_y, 'summary': rules.ratio_summary(base_by_y, var_by_y)}


@test
def table_keeps_below_criteria_ties_and_missing_out_of_cells():
    base = {2014: v(1, 1), 2016: v(-1, 1), 2018: v(1, -1, ok=False), 2020: v(0.0, 1), 2022: v(None, 1), 2024: v(-1, -1)}
    t = rules.ratio_summary(base, {y: [base[y]] * 18 for y in base})['table']
    assert t['cells'] == {'neg/neg': [2024], 'neg/pos': [2016], 'pos/neg': [], 'pos/pos': [2014]}
    assert t['below_criteria'] == [2018] and t['tie'] == [2020] and t['missing'] == [2022]


@test
def assumption_dependence_from_sign_or_criteria_flip():
    stable = [v(1, 1)] * 18
    assert rules.dependence(stable, 'a') == 'stable'
    assert rules.dependence(stable[:-1] + [v(-1, 1)], 'a') == 'dependent' and rules.dependence(stable[:-1] + [v(-1, 1)], 'b') == 'stable'
    assert rules.dependence(stable[:-1] + [v(0.0, 1)], 'a') == 'dependent'
    assert rules.dependence(stable[:-1] + [v(1, 1, ok=False)], 'a') == 'dependent' and rules.dependence(stable[:-1] + [v(1, 1, ok=False)], 'b') == 'dependent'
    assert rules.dependence(stable[:-1], 'a') == 'undetermined' and rules.dependence(stable[:-1] + [v(None, 1)], 'a') == 'undetermined'


@test
def sentences_follow_the_fixed_rules():
    ys = sb.YEARS
    win = ratio_data({y: v(0.5, -0.1) for y in ys}); lose = ratio_data({y: v(-0.5, 0.1) for y in ys})
    assert rules.sentence([], {0.03: win})['kind'] == 'no_boundary' and rules.sentence([], {0.03: win})['text'] == rules.S_NO_BOUNDARY
    assert rules.sentence([[0.03, 0.04, 0.05]], {r: win for r in (0.03, 0.04, 0.05)})['text'] == rules.S_WIN + '.'
    assert rules.sentence([[0.03, 0.04, 0.05]], {r: lose for r in (0.03, 0.04, 0.05)})['text'] == rules.S_LOSE + '.'
    one_var_loses = ratio_data({y: v(0.5, 1) for y in ys}, {y: [v(0.5, 1)] * 17 + [v(-0.1, 1)] for y in ys})
    s = rules.sentence([[0.03, 0.04, 0.05]], {0.03: win, 0.04: one_var_loses, 0.05: win})
    assert s['kind'] == 'counts' and rules.S_WIN not in s['text'] and '주문율 4%' in s['text']
    one_var_fails = ratio_data({y: v(0.5, 1) for y in ys}, {y: [v(0.5, 1)] * 17 + [v(0.5, 1, ok=False)] for y in ys})
    s = rules.sentence([[0.03, 0.04, 0.05]], {0.03: win, 0.04: one_var_fails, 0.05: win})
    assert s['kind'] == 'counts' and '기준 미달' in s['text']
    # 구간이 둘이면 함께 본다: 둘째 구간에서 지면 '이겼다'가 아니다
    s = rules.sentence([[0.022, 0.03, 0.04], [0.06, 0.08, 0.1]], {**{r: win for r in (0.022, 0.03, 0.04)}, **{r: lose for r in (0.06, 0.08, 0.1)}})
    assert s['kind'] == 'counts' and s['ratios'] == [0.022, 0.03, 0.04, 0.06, 0.08, 0.1]
    s = rules.sentence([[0.022, 0.03, 0.04], [0.06, 0.08, 0.1]], {r: win for r in sz.RATIOS if r != 0.05})
    assert s['kind'] == 'win'
    missing_basket = ratio_data({y: v(0.5, 1) for y in ys[:5]})
    assert rules.sentence([[0.03, 0.04, 0.05]], {0.03: missing_basket, 0.04: win, 0.05: win})['kind'] == 'counts'


# ---------------- 주문액 기록 검사 ----------------
class J:
    def __init__(self, **kw): self.__dict__.update(kw)


def order_case(ratio=0.05):
    """엔진 규칙대로 만든 합성 기록: 엔진 시작일과 매달 첫 거래일에 basis = 전일 평가액(+ 그날 입금 배당)으로 정한다."""
    dates = pd.bdate_range('2020-01-29', '2020-03-06')
    job = J(prefix='t', ratio=str(ratio), start=str(dates[1].date()), end=str(dates[-1].date()))
    value = pd.Series(sb.CASH0 + np.arange(len(dates)) * 1000.0, index=dates)
    curve = pd.DataFrame({'value_engine': value, 'receivable': 0.0, 'value': value})
    pay_day = [d for d in dates if d.month == 3][0]
    divs = pd.DataFrame([{'fy': 2019, 'event': 'paid', 'date': str(pay_day.date()), 'amount': 5000.0}])
    rows, ipo = [], None
    for i, d in enumerate(dates):
        new = i == 0 or d.month != dates[i - 1].month
        if new:
            basis = sb.CASH0 if i == 0 else float(np.float32(value.iloc[i - 1] + (5000.0 if d == pay_day else 0.0)))
            ipo = szc.f32_order(basis, ratio)
        rows.append({'date': d, 'recomputed': new, 'basis': basis if new else np.nan, 'investment_per_order': ipo})
    log = pd.DataFrame(rows)
    price = 9990.0; q = int(ipo // price); amt = q * price + np.floor(q * price * 0.00015)
    fills = pd.DataFrame([{'date': dates[3], 'ticker': 'A00001', 'type': 'BUY', 'dq': q, 'dcash': -amt, 'order': 1, 'reason': funds.REASON_NEW}])
    prices = fills[['date', 'ticker', 'dq']].assign(buy_price=price)
    return job, {'curve': curve, 'divs': divs, 'fills': fills, 'buyprices': prices}, log, dates


@test
def order_log_check_passes_engine_rule_and_catches_each_fault():
    job, run, log, dates = order_case()
    assert szc.order_log_errors(job, run, log, dates) == [], szc.order_log_errors(job, run, log, dates)
    bad = log.copy(); bad.loc[bad.recomputed, 'basis'] *= 1.01; bad['investment_per_order'] = [szc.f32_order(b, 0.05) if r else np.nan for b, r in zip(bad.basis, bad.recomputed)]
    bad['investment_per_order'] = bad.investment_per_order.ffill()
    assert any('prior close value' in e for e in szc.order_log_errors(job, run, bad, dates))
    wrong_ratio = J(**{**job.__dict__, 'ratio': '0.06'})
    assert any('float32(basis) x ratio' in e for e in szc.order_log_errors(wrong_ratio, run, log, dates))
    mid = log.copy(); mid.loc[5, 'investment_per_order'] += 1
    assert any('mid-month' in e for e in szc.order_log_errors(job, run, mid, dates))
    skip = log.copy(); i = skip.index[skip.recomputed][1]; skip.loc[i, 'recomputed'] = False
    assert any('month starts' in e for e in szc.order_log_errors(job, run, skip, dates))
    nodiv = {**run, 'divs': run['divs'].iloc[0:0]}
    assert any('same-day dividend' in e for e in szc.order_log_errors(job, nodiv, log, dates))
    assert any('dates differ' in e for e in szc.order_log_errors(job, run, log.iloc[:-1], dates))
    for shift in (-1, 1):
        fills = run['fills'].assign(dq=run['fills'].dq + shift)
        prices = run['buyprices'].assign(dq=fills.dq)
        drift = {**run, 'fills': fills, 'buyprices': prices}
        assert any('not sized' in e for e in szc.order_log_errors(job, drift, log, dates))
    assert any('does not match' in e for e in szc.order_log_errors(job, {**run, 'fills': fills}, log, dates))


@test
def buy_one_share_short_at_exact_budget_is_rejected():
    job, run, _, dates = order_case()
    date = dates[3]
    ipo = pd.Series({date: 1_000_000.0})
    for q in (49, 50):
        gross = q * 20_000
        fills = run['fills'].assign(dq=q, dcash=-(gross + np.floor(gross * 0.00015)))
        prices = run['buyprices'].assign(dq=q, buy_price=20_000)
        errs = szc.fill_size_errors(job, {'fills': fills, 'buyprices': prices}, ipo)
        assert bool(errs) is (q == 49), errs


# ---------------- r1 대조 ----------------
@test
def r1_comparison_is_byte_exact_except_meta_argv():
    import same_basket_size_run as szr
    d = tempfile.mkdtemp()
    try:
        a, b = os.path.join(d, 'new'), os.path.join(d, 'old')
        for pre in (a, b):
            for s in szr.CMP_FILES: open(f'{pre}_{s}.csv', 'w').write('date,x\n2020-01-02,1\n2020-01-03,2\n')
            json.dump({'end': '2020-01-03', 'initial_cash': 1e7}, open(pre + '_sbmeta.json', 'w'))
            json.dump({'success': True, 'argv': [pre]}, open(pre + '_meta.json', 'w'))
        assert szr.compare_r1(a, b, '2020-01-03')['passed']
        open(a + '_fills.csv', 'w').write('date,x\n2020-01-02,1\n2020-01-03,3\n')
        r = szr.compare_r1(a, b, '2020-01-03'); assert not r['passed'] and r['files']['fills'] is False and r['mode'] == 'bytes'
        assert szr.compare_r1(a, b, '2020-01-02')['passed'] and szr.compare_r1(a, b, '2020-01-02')['mode'] == 'truncated'
        json.dump({'success': False, 'argv': [a]}, open(a + '_meta.json', 'w')); open(a + '_fills.csv', 'w').write(open(b + '_fills.csv').read())
        assert not szr.compare_r1(a, b, '2020-01-03')['passed']
    finally: shutil.rmtree(d)


# ---------------- 조립 ----------------
def fake_outs(jobs, finals, crit):
    """finals(ratio, basket, k, div, delist) → (X, H, K 최종값), crit(ratio, basket, k) → 기준 통과 여부."""
    outs = {}
    for j in jobs.itertuples():
        x, h, k = finals(float(j.ratio), j.basket, j.k, j.div, j.delist)
        good = crit(float(j.ratio), j.basket, j.k)
        curve, state, fills, log = synth_x(n=5, invest=0.95 if good else 0.5, short_days=0)
        curve = curve.copy(); curve.iloc[-1] = [x, 0.0, x]
        hc = pd.DataFrame({'value': [sb.CASH0] + [h] * 5}, index=curve.index); kc = pd.Series([sb.CASH0] + [k] * 5, index=curve.index)
        outs[j.prefix] = {'x': {'curve': curve, 'state': state, 'fills': fills}, 'log': log, 'h': {'curve': hc}, 'k': {'curve': kc}}
    return outs


def run_assembly(finals, crit):
    import same_basket_size_verdict as ver
    jobs = sz.build_jobs(calendar())
    tables, summary = ver.assemble(jobs, fake_outs(jobs, finals, crit), {y: ['A00001', 'B00002'] for y in sb.YEARS})
    return tables, summary, ver.verdict_markdown(summary, tables)


@test
def assembly_finds_boundary_and_writes_the_fixed_sentence():
    in_band = lambda r: r in (0.04, 0.05, 0.06)
    tables, summary, md = run_assembly(lambda r, y, k, d, z: (1.3e7, 1.2e7, 1.1e7), lambda r, y, k: in_band(r))
    assert summary['boundary_runs'] == [['4%', '5%', '6%']] and summary['sentence']['kind'] == 'win' and rules.S_WIN in md
    assert len(tables['summary_jobs']) == 756 and len(tables['baskets_by_ratio']) == 42 and len(tables['ratios']) == 7
    r = tables['ratios'].set_index('ratio_text')
    assert r.loc['5%', 'cell_pos_pos'] == 6 and r.loc['2.2%', 'below_criteria'] == 6 and r.loc['2.2%', 'cell_pos_pos'] == 0
    tables, summary, md = run_assembly(lambda r, y, k, d, z: (1.1e7, 1.2e7, 1.0e7), lambda r, y, k: in_band(r))
    assert summary['sentence']['kind'] == 'lose' and rules.S_LOSE in md and rules.S_WIN not in md
    # 경계 구간 안 한 주문율에서 k=20 가정만 기준 미달 → 문장은 개수, 그 주문율·바구니는 가정 의존
    tables, summary, md = run_assembly(lambda r, y, k, d, z: (1.3e7, 1.2e7, 1.1e7), lambda r, y, k: in_band(r) and not (r == 0.05 and y == 2016 and k == 20))
    assert summary['sentence']['kind'] == 'counts' and rules.S_WIN not in md
    b = tables['baskets_by_ratio']; row = b[(b.ratio == 0.05) & (b.basket == 2016)].iloc[0]
    assert row.a_dependence == 'dependent' and summary['boundary_runs'] == [['4%', '5%', '6%']]
    tables, summary, md = run_assembly(lambda r, y, k, d, z: (1.3e7, 1.2e7, 1.1e7), lambda r, y, k: r in (0.04, 0.05))
    assert summary['boundary_runs'] == [] and summary['sentence']['text'] == rules.S_NO_BOUNDARY and rules.S_WIN not in md


# ---------------- 조립 거부(실제 실행) ----------------
def run_verdict(out, mode, *extra):
    return subprocess.run([sys.executable, '-B', os.path.join(HERE, 'same_basket_size_verdict.py'), out, '--mode', mode, *extra],
                          capture_output=True, text=True, cwd=os.getcwd())


def verdict_dir(out, jobs, status):
    os.makedirs(os.path.join(out, 'runs'), exist_ok=True)
    shutil.copytree(sb.BASKET_DIR, os.path.join(out, 'baskets'), dirs_exist_ok=True)
    jobs.to_csv(os.path.join(out, 'jobs.tsv'), sep='\t', index=False)
    with open(os.path.join(out, 'job_status.tsv'), 'w') as f:
        f.write('id\tprefix\texit\tseconds\tfinished_utc\n')
        for j, code in status: f.write(f'{j.id}\t{j.prefix}\t{code}\t1\tx\n')
    os.makedirs(os.path.join(out, 'verdict'), exist_ok=True)
    open(os.path.join(out, 'verdict', 'VERDICT.md'), 'w').write('old'); open(os.path.join(out, 'verdict', 'verdict.json'), 'w').write('{}')


def refused(out, mode, word):
    r = run_verdict(out, mode)
    v = os.path.join(out, 'verdict'); c = json.load(open(os.path.join(v, 'checks.json')))
    assert r.returncode == 2 and os.listdir(v) == ['checks.json'] and not c['passed'], (r.returncode, os.listdir(v), c)
    assert any(word in e for e in c['errors']), c['errors']


def assembler_refuses_end_to_end():
    import stage2_common as sc
    dates = sc.trading_dates(sc.connect())
    smoke = sz.build_smoke_jobs(dates, 2014, '2014-02-28', ratios=(0.022, 0.03))
    d = tempfile.mkdtemp()
    try:
        cases = [('failed', [(j, 0) for j in smoke.itertuples()][:-1] + [(list(smoke.itertuples())[-1], 1)], 'job failed'),
                 ('unrecorded', [(j, 0) for j in smoke.itertuples()][1:], 'job not recorded'),
                 ('no_r1cmp', [(j, 0) for j in smoke.itertuples()], 'r1 comparison not recorded')]
        for name, status, word in cases:
            out = os.path.join(d, name); verdict_dir(out, smoke, status); refused(out, 'smoke', word)
        out = os.path.join(d, 'r1cmp_failed'); verdict_dir(out, smoke, [(j, 0) for j in smoke.itertuples()])
        for j in smoke[smoke.kind == 'r1x'].itertuples():
            json.dump({'passed': j.id != 1, 'mode': 'truncated', 'r1_prefix': j.r1prefix, 'files': {}}, open(os.path.join(out, 'runs', j.prefix + '_r1cmp.json'), 'w'))
        refused(out, 'smoke', 'differs from r1 X')
        out = os.path.join(d, 'no_outputs'); verdict_dir(out, smoke, [(j, 0) for j in smoke.itertuples()])
        for j in smoke[smoke.kind == 'r1x'].itertuples():
            json.dump({'passed': True, 'mode': 'truncated', 'r1_prefix': j.r1prefix, 'files': {}}, open(os.path.join(out, 'runs', j.prefix + '_r1cmp.json'), 'w'))
        refused(out, 'smoke', 'cannot read outputs')
        full = sz.build_jobs(dates)
        out = os.path.join(d, 'formal_short'); verdict_dir(out, full.iloc[:-1], [(j, 0) for j in full.iloc[:-1].itertuples()]); refused(out, 'formal', 'job rows differ')
        out = os.path.join(d, 'formal_bytes'); verdict_dir(out, full, [(j, 0) for j in full.itertuples()])
        for j in full[full.kind == 'r1x'].itertuples():
            json.dump({'passed': True, 'mode': 'truncated', 'r1_prefix': j.r1prefix, 'files': {}}, open(os.path.join(out, 'runs', j.prefix + '_r1cmp.json'), 'w'))
        refused(out, 'formal', 'not a full byte comparison')
        out = os.path.join(d, 'fail_flag'); verdict_dir(out, smoke, [])
        r = run_verdict(out, 'smoke', '--fail', 'inputs differ')
        c = json.load(open(os.path.join(out, 'verdict', 'checks.json')))
        assert r.returncode == 2 and os.listdir(os.path.join(out, 'verdict')) == ['checks.json'] and c['errors'] == ['run_all: inputs differ']
    finally: shutil.rmtree(d)


def main():
    tests = list(TESTS) + ([] if '--no-db' in sys.argv else [assembler_refuses_end_to_end])
    failed = []
    for t in tests:
        try: t(); print(f'ok   {t.__name__}')
        except Exception as exc: failed.append(t.__name__); print(f'FAIL {t.__name__}: {type(exc).__name__}: {exc}')
    print(f'{len(tests) - len(failed)} tests passed' if not failed else f'{len(failed)} of {len(tests)} tests FAILED')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
