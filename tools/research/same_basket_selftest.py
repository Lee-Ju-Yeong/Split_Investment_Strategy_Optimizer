"""같은 종목 대조(same-basket-r1) 자체 검사: 합성 자료로 H 장부·K 계좌·판정 규칙·가정 의존 집계·조립 거부 조건을 확인한다.

usage: python -B tools/research/same_basket_selftest.py [--no-db]
3월 실행 체크아웃을 작업 디렉터리로 실행한다(엔진의 호가 올림과 대조하고, 감싸기 함수를 엔진 클래스 위에서 시험한다).
--no-db가 없으면 조립기를 임시 폴더에서 실제로 돌려 거부(종료 코드 2, 표·문장 없음)를 확인한다(DB는 거래일 목록만 읽는다).
실제 성과 자료는 읽지 않는다. 모든 검사가 통과하면 'N tests passed'를 출력하고 0으로 끝난다.
"""
import io, json, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import same_basket_common as sb
import same_basket_rules as rules
import same_basket_ledger as ledger
import same_basket_checks as chk
import same_basket_aux as aux

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = []


def test(fn):
    TESTS.append(fn); return fn


def raises(fn, exc=Exception):
    try: fn()
    except exc: return True
    return False


# ---------------- 합성 자료 ----------------
def calendar():
    return pd.bdate_range('2013-01-02', '2026-01-09')


def fake_baskets():
    return {y: [f'{y % 100:02d}{i:04d}' for i in range(sb.N_BASKET)] for y in sb.YEARS}


def hold_input(opens, closes, events=(), delist=None, tickers=('AAA001',)):
    """날짜 0은 시작 전날. opens·closes: [일][종목]."""
    n = len(opens)
    return {'dates': [f'2020-01-{d + 1:02d}' for d in range(n)], 'tickers': list(tickers), 'open': np.array(opens, dtype=float),
            'close': np.array(closes, dtype=float), 'start_i': 1, 'events': list(events), 'delist_i': list(delist or [None] * len(tickers))}


def event(fy, rights_i, pay_i, nodiv=(), **per):
    """per: 종목=(수정계수, 원주 종가, DPS, 대체 여부) → 입력 상태 ok. nodiv: 배당 없음으로 분류할 종목."""
    info = {t: {'ratio': r, 'close': c, 'dps': d, 'fallback': f, 'status': 'ok'} for t, (r, c, d, f) in per.items()}
    info.update({t: {'ratio': 1.0, 'close': 1.0, 'dps': 0.0, 'fallback': False, 'status': 'no_dividend'} for t in nodiv})
    return {'fy': fy, 'record_date': f'rec{fy}', 'rights_i': rights_i, 'pay_i': pay_i, 'per_ticker': info}


def curve_of(state):
    return pd.DataFrame(state['curve']).set_index('date')


# ---------------- H 장부 ----------------
@test
def hold_initial_buy_matches_hand_calculation():
    # 시가 10,037 → 호가 10원 올림 10,040. 50만 원 한도: 49주 = 491,960 + 수수료 floor(73.79) 73 = 492,033 (50주는 502,075로 초과)
    s = ledger.run_hold(hold_input([[np.nan], [10037], [10100]], [[10000], [10050], [10200]]), 'none', 'freeze')
    b = s['buys'][0]
    assert (b['price'], b['quantity'], b['gross'], b['commission'], b['cost']) == (10040, 49, 491960, 73, 492033), b
    c = curve_of(s)
    assert c.value.iloc[0] == sb.CASH0 and c.cash.iloc[0] == sb.CASH0
    assert c.cash.iloc[1] == sb.CASH0 - 492033 and c.stock_value.iloc[1] == 49 * 10050 and c.value.iloc[2] == sb.CASH0 - 492033 + 49 * 10200


@test
def hold_buys_the_most_shares_that_fit_with_fee():
    for budget, price in ((500000, 10040), (500000, 499925), (500000, 500000), (500000, 1), (9800, 4900), (4900, 4900), (0, 100)):
        n = ledger.max_shares(budget, price)
        assert ledger.buy_cost(n, price) <= budget < ledger.buy_cost(n + 1, price), (budget, price, n)
    assert ledger.max_shares(500000, 499925) == 1 and ledger.max_shares(500000, 500000) == 0     # 500,000원 한 주는 수수료 75원 때문에 못 산다


@test
def hold_untradable_slot_stays_cash_forever():
    opens = [[np.nan, np.nan], [10000, np.nan], [10000, 5000], [10000, 0]]
    s = ledger.run_hold(hold_input(opens, [[1, 1], [10000, 5000], [10000, 5000], [10000, 5000]], tickers=('AAA001', 'BBB002')), 'none', 'freeze')
    assert s['skipped'] == ['BBB002'] and [b['ticker'] for b in s['buys']] == ['AAA001'] and s['qty'][1] == 0
    zero_open = ledger.run_hold(hold_input([[np.nan], [0], [100]], [[1], [100], [100]]), 'none', 'freeze')
    assert zero_open['skipped'] == ['AAA001'] and not zero_open['buys'] and curve_of(zero_open).value.iloc[-1] == sb.CASH0


@test
def hold_dividend_uses_raw_quantity_and_reinvests_on_pay_day():
    # 49주 × 수정계수 2.0 = 원주 98주 × DPS 100 = 9,800원. 권리 확정 다음 날(3일째)부터 미수금, 5일째 지급.
    # 지급일 시가 4,900 → 1주 4,900원(수수료 0), 남은 4,900원은 현금. 주식분할(수정계수)로 수량은 바뀌지 않는다.
    opens = [[np.nan], [10037], [10000], [10000], [10000], [4900], [4900]]
    closes = [[10000]] * 5 + [[5000], [5000]]
    ev = event(2019, 3, 5, AAA001=(2.0, 5000.0, 100.0, False))
    s = ledger.run_hold(hold_input(opens, closes, [ev]), 'approx', 'freeze'); c = curve_of(s)
    assert s['divs'][0]['raw_qty'] == 98 and s['divs'][0]['amount'] == 9800
    assert list(c.receivable) == [0, 0, 0, 9800, 9800, 0, 0] and c.value.iloc[3] - c.value.iloc[2] == 9800
    re = [b for b in s['buys'] if b['kind'] == 'reinvest'][0]
    assert (re['date'], re['quantity'], re['cost'], re['budget']) == ('2020-01-06', 1, 4900, 9800) and s['qty'][0] == 50
    assert c.cash.iloc[5] == sb.CASH0 - 492033 + 9800 - 4900 and c.cash.iloc[6] == c.cash.iloc[5] and not s['pending']
    one = ledger.run_hold(hold_input(opens, closes, [event(2019, 3, 5, AAA001=(1.0, 5000.0, 100.0, False))]), 'approx', 'freeze')
    assert one['divs'][0]['amount'] == 4900 and one['qty'][0] == 50          # 수정계수만 다르면 배당만 달라지고 최초 수량은 같다


@test
def hold_reinvests_on_next_tradable_open_and_never_reuses_leftover():
    opens = [[np.nan], [10037], [10000], [10000], [np.nan], [np.nan], [4000], [4000]]
    closes = [[10000]] * 8
    ev = [event(2019, 3, 4, AAA001=(1.0, 10000.0, 100.0, False)), event(2020, 6, 7, AAA001=(1.0, 10000.0, 100.0, False))]
    s = ledger.run_hold(hold_input(opens, closes, ev), 'approx', 'freeze')
    re = [b for b in s['buys'] if b['kind'] == 'reinvest']
    assert (re[0]['date'], re[0]['budget'], re[0]['quantity']) == ('2020-01-07', 4900, 1)       # 지급일(5일)·다음 날 거래 불가 → 7일 시가
    # 두 번째 권리 수량은 7일 재투자 전(전일 종가) 보유 49주다. 예산은 그 배당 4,900원뿐이고 앞의 잔돈 900원은 쓰지 않는다
    assert (re[1]['date'], re[1]['budget'], re[1]['quantity']) == ('2020-01-08', 4900, 1) and s['divs'][1]['adj_qty'] == 49
    assert curve_of(s).cash.iloc[-1] == sb.CASH0 - 492033 + 4900 - 4000 + 4900 - 4000 and s['qty'][0] == 51


@test
def hold_dividend_assumptions_and_filters():
    opens = [[np.nan], [10037], [10000], [5000], [5000], [5000]]; closes = [[10000]] * 6
    ev = event(2019, 3, 5, AAA001=(1.0, 10000.0, 200.0, True))
    none = ledger.run_hold(hold_input(opens, closes, [ev]), 'none', 'freeze')
    assert not none['divs'] and not none['events'] and len(none['buys']) == 1
    upper = ledger.run_hold(hold_input(opens, closes, [ev]), 'upper', 'freeze'); c = curve_of(upper)
    assert [e['event'] for e in upper['events']] == ['receivable', 'paid'] and (c.receivable == 0).all()
    assert [b['date'] for b in upper['buys'] if b['kind'] == 'reinvest'] == ['2020-01-04'] and upper['qty'][0] == 50    # 9,800원으로 5,000원 1주
    approx = ledger.run_hold(hold_input(opens, closes, [event(2019, 3, None, AAA001=(1.0, 10000.0, 200.0, False))]), 'approx', 'freeze')
    assert curve_of(approx).receivable.iloc[-1] == 9800 and approx['qty'][0] == 49                # 지급일이 자료 밖이면 끝까지 미수금
    high = ledger.run_hold(hold_input(opens, closes, [event(2019, 3, 5, AAA001=(1.0, 100.0, 31.0, False))]), 'approx', 'freeze')
    assert not high['divs'] and high['events'][0]['amount'] == 0                                  # 배당률 31%는 제외
    assert raises(lambda: ledger.run_hold(hold_input(opens, closes, [event(2019, 3, 5, AAA001=(np.nan, 10000.0, 200.0, False))]), 'approx', 'freeze'), ValueError)
    early = ledger.run_hold(hold_input(opens, closes, [event(2019, 1, 2, AAA001=(1.0, 10000.0, 200.0, False))]), 'approx', 'freeze')
    assert not early['divs']                                                                      # 시작일의 권리 수량은 0(전일 보유 없음)


@test
def hold_delisting_freeze_and_zero():
    opens = [[np.nan], [10037], [10000], [np.nan], [np.nan]]; closes = [[10000], [10000], [9000], [9000], [9000]]   # 3일째부터 거래 없음(마지막 종가 유지)
    freeze = curve_of(ledger.run_hold(hold_input(opens, closes, delist=[3]), 'none', 'freeze'))
    zero = curve_of(ledger.run_hold(hold_input(opens, closes, delist=[3]), 'none', 'zero'))
    assert list(freeze.stock_value) == [0, 490000, 441000, 441000, 441000] and list(zero.stock_value) == [0, 490000, 441000, 0, 0]
    assert zero.value.iloc[-1] == sb.CASH0 - 492033 and (freeze.cash == zero.cash).all()
    alive = curve_of(ledger.run_hold(hold_input(opens, closes, delist=[None]), 'none', 'zero'))
    assert alive.equals(freeze)                                                                   # 폐지가 없으면 두 가정이 같다


@test
def dividend_events_map_rights_and_pay_days():
    import stage2_common as sc
    cal = sc.Calendar(calendar()); dates = cal.dates[(cal.dates >= '2013-12-31') & (cal.dates <= '2015-12-31')]
    keep = (sc.dps_table, sc.adj_ratio_on); asked = []
    def dps_table(conn, tickers, years):
        asked.append(tuple(years))
        return pd.DataFrame([('AAA001', years[1], 150.0, False), ('BBB002', years[1], 70.0, True), ('AAA001', years[0], 1.0, False)], columns=['ticker', 'fy', 'dps', 'dps_fallback'])
    try:
        sc.dps_table = dps_table
        sc.adj_ratio_on = lambda conn, tickers, day: {'AAA001': (2.0, 5000.0), 'CCC003': (1.0, 100.0)}
        ev = ledger.dividend_events(None, cal, ['AAA001', 'BBB002', 'CCC003'], dates, 1)
    finally: sc.dps_table, sc.adj_ratio_on = keep
    assert [e['fy'] for e in ev] == [2014, 2015] and asked == [(2013, 2014), (2014, 2015)]      # FY2013의 권리 확정 다음 날은 시작 전날이라 빠진다
    e14, e15 = ev
    assert e14['record_date'] == '2014-12-30' and str(dates[e14['rights_i']].date()) == '2014-12-31' and str(dates[e14['pay_i']].date()) == '2015-05-18'
    assert {t: v['status'] for t, v in e14['per_ticker'].items()} == {'AAA001': 'ok', 'BBB002': 'ratio_missing', 'CCC003': 'no_dividend'}
    assert (e14['per_ticker']['AAA001']['ratio'], e14['per_ticker']['AAA001']['dps']) == (2.0, 150.0)
    assert str(dates[e15['rights_i']].date()) == '2015-12-31' and e15['pay_i'] is None         # 지급일이 자료 밖이면 None(끝까지 미수금)
    ev = [{**e, 'per_ticker': {t: v for t, v in e['per_ticker'].items()}} for e in ev]
    for e in ev: e['per_ticker']['BBB002'] = {**e['per_ticker']['BBB002'], 'status': 'no_dividend'}     # 이 시험은 계산 경로만 본다(환산 불가는 r2 시험)
    n = len(dates); inp = {'dates': [str(d.date()) for d in dates], 'tickers': ['AAA001', 'BBB002', 'CCC003'], 'open': np.full((n, 3), 10037.0),
                           'close': np.full((n, 3), 10000.0), 'start_i': 1, 'events': ev, 'delist_i': [None] * 3}
    s = ledger.run_hold(inp, 'approx', 'freeze'); c = curve_of(s)
    assert [d['amount'] for d in s['divs']] == [49 * 2.0 * 150.0, 50 * 2.0 * 150.0] and c.receivable.iloc[-1] == 15000.0
    assert [b['date'] for b in s['buys'] if b['kind'] == 'reinvest'] == ['2015-05-18'] and s['qty'] == [50, 49, 49]


@test
def tick_rule_matches_engine():
    from src.backtest.cpu.execution import BasicExecutionHandler
    ex = BasicExecutionHandler()
    prices = [1, 999.5, 1999, 1999.2, 2000, 2001, 4999, 5000, 5003, 19999, 20000, 20010, 49999, 50000, 50001, 199999, 200000, 200050, 499999, 500000, 500400, 1234567]
    prices += list(np.random.default_rng(7).uniform(1, 900000, 3000))
    for p in prices: assert ledger.tick_up(p) == ex._adjust_price_up(np.float32(p)), p
    assert ledger.tick_up(10037) == 10040 and ledger.tick_up(10040) == 10040


@test
def hold_files_round_trip_and_checks_catch_tampering():
    days = calendar(); job = first(fake_jobs())
    d = days[(days >= sb.prev_day(days, job.start)) & (days <= pd.Timestamp(job.end))]
    tickers = fake_baskets()[2014]; n = len(d)
    opens = np.full((n, 20), 10037.0); closes = np.full((n, 20), 10000.0); opens[:, 3] = np.nan
    inp = {'dates': [str(x.date()) for x in d], 'tickers': tickers, 'open': opens, 'close': closes, 'start_i': 1, 'delist_i': [None] * 20,
           'events': [event(2014, 250, 300, nodiv=tickers[1:], **{tickers[0]: (1.0, 10000.0, 300.0, False)})]}
    tmp = tempfile.mkdtemp(); prefix = os.path.join(tmp, job.prefix)
    try:
        meta = {'price_basis': 'adjusted', 'delisted': {}, 'account_start': job.start, 'end': job.end, 'div': 'approx', 'delist': 'freeze', 'basket_sha256': 'h'}
        ledger.write_hold(prefix, ledger.run_hold(inp, 'approx', 'freeze'), meta)
        h = chk.read_h(prefix)
        assert chk.h_errors(job, h, days, tickers, 'h') == [], chk.h_errors(job, h, days, tickers, 'h')
        assert h['meta']['skipped_at_start'] == [tickers[3]] and h['meta']['n_reinvest_buys'] == 1
        bad = {**h, 'buys': h['buys'].assign(quantity=lambda b: b.quantity - (b.index == 0))}
        assert any('price/quantity/fee' in e for e in chk.h_errors(job, bad, days, tickers, 'h'))
        bad = {**h, 'curve': h['curve'].assign(cash=lambda c: c.cash + (c.index == c.index[5]) * 10)}
        assert any('reconcile' in e or 'value !=' in e for e in chk.h_errors(job, bad, days, tickers, 'h'))
        assert any('H meta div' in e for e in chk.h_errors(job._replace(div='none'), h, days, tickers, 'h'))
        assert any('basket' in e for e in chk.h_errors(job, h, days, tickers[:19] + ['ZZZ999'], 'h'))
    finally: shutil.rmtree(tmp)


# ---------------- K 계좌 ----------------
@test
def kodex_formula_and_bad_inputs():
    days = pd.bdate_range('2020-01-02', periods=3)
    etf = pd.DataFrame({'open': [25010.0, 25100, 25200], 'close': [25050.0, 25300, 24900]}, index=days)
    v, n = ledger.kodex_curve(etf, days)                    # 10,000,000 / (25,010 × 1.00015) = 399.78 → 399주, 수수료 floor(1,496.8) = 1,496
    assert n == 399 and v.iloc[0] == sb.CASH0 - (399 * 25010 + 1496) + 399 * 25050 and v.iloc[2] == sb.CASH0 - (399 * 25010 + 1496) + 399 * 24900
    for bad in (etf.assign(close=[25050, np.nan, 24900]), etf.assign(open=[0, 1, 1]), etf.iloc[[0, 2]], pd.concat([etf, etf.iloc[:1]])):
        assert raises(lambda: ledger.kodex_curve(bad, days), ValueError)
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'etf.csv')
        open(path, 'w').write('date,open,high,low,close,volume\n2025-12-29,1,1,1,1,1\n2025-12-30,2,2,2,2,2\n2026-01-02,not-a-number,3,3,3,3\n')
        assert list(ledger.read_etf_capped(path).index.strftime('%Y-%m-%d')) == ['2025-12-29', '2025-12-30']
    finally: shutil.rmtree(tmp)
    job = first(fake_jobs()); cal = calendar()
    d = cal[(cal >= sb.prev_day(cal, job.start)) & (cal <= pd.Timestamp(job.end))]
    full = pd.DataFrame({'open': 25010.0, 'close': np.linspace(25000, 26000, len(cal))}, index=cal)
    good, _ = ledger.kodex_curve(full, d[1:]); k = {'curve': pd.concat([pd.Series([sb.CASH0], index=d[:1]), good])}
    assert chk.k_errors(job, k, cal, full, archived=good) == []
    assert any('archived' in e for e in chk.k_errors(job, k, cal, full, archived=good + 1))
    assert any('formula' in e for e in chk.k_errors(job, {'curve': k['curve'] * 1.0001}, cal, full))


# ---------------- 달력·작업표 ----------------
def fake_jobs():
    return sb.build_jobs(calendar(), fake_baskets())


def first(df):
    """조립기가 쓰는 것과 같은 행 형태(itertuples)."""
    return next(df.itertuples())


@test
def calendar_and_job_table():
    cal = calendar()
    assert sb.open_day(cal, 2014) == pd.Timestamp('2014-01-01') and sb.prev_day(cal, '2014-01-01') == pd.Timestamp('2013-12-31')
    assert sb.start_day(cal, 2014, 5) == pd.Timestamp('2014-01-08') and sb.start_day(cal, 2014, 20) == pd.Timestamp('2014-01-29')
    assert sb.main_end(cal, 2014) == pd.Timestamp('2015-12-31') and sb.main_end(cal, 2024) == pd.Timestamp('2025-12-30')
    assert raises(lambda: sb.prev_day(cal, '2014-01-04'), ValueError)
    jobs = fake_jobs()
    assert jobs.groupby('kind').size().to_dict() == sb.N_JOBS and jobs.prefix.is_unique and list(jobs.id) == list(range(1, 355))
    first = jobs[jobs.basket == 2014].iloc[0]
    assert (first.kind, first.k, first['div'], first.delist) == ('main',) + sb.MAIN
    assert set(jobs[jobs.kind == 'ext'].end) == {'2025-12-30'} and (jobs[jobs.kind != 'ext'].end <= '2025-12-30').all()
    a = jobs[jobs.kind == 'aux']
    assert set(a.ratio) == set(sb.AUX_RATIOS) and (a.groupby(['basket', 'ratio']).ticker.nunique() == 20).all() and set(a.k) == {0}
    assert rules.check_job_structure(jobs.to_dict('records'), sb.N_JOBS) == [] and sb.compare_jobs(jobs, fake_jobs()) == []
    tampered = jobs.copy(); tampered.loc[5, 'end'] = '2015-12-30'
    assert sb.compare_jobs(tampered, jobs) and sb.compare_jobs(jobs.iloc[:-1], jobs) and sb.compare_jobs(jobs.iloc[::-1], jobs)
    assert rules.check_job_structure(jobs.iloc[1:].to_dict('records'), sb.N_JOBS) == ['main: 107 jobs (need 108)']
    smoke = sb.build_smoke_jobs(cal, fake_baskets(), 2014, '2014-01-29', '2014-02-28')
    assert smoke.groupby('kind').size().to_dict() == {'aux': 40, 'ext': 1, 'main': 6} and set(smoke[smoke.kind != 'ext'].end) == {'2014-01-29'}


# ---------------- 판정 규칙 ----------------
def basket_values(a=-1.0, b=-1.0, a_all=None, b_all=None):
    return {'a': a, 'b': b, 'a_all': a_all if a_all is not None else [a] * 18, 'b_all': b_all if b_all is not None else [b] * 18}


@test
def sign_and_dependence():
    assert [rules.sign(x) for x in (-1e-12, 0.0, 3, None, float('nan'), float('inf'))] == ['neg', 'tie', 'pos', 'missing', 'missing', 'missing']
    assert rules.dependence([-1.0] * 18) == 'stable' and rules.dependence([2.0] * 18) == 'stable'
    assert rules.dependence([-1.0] * 17 + [0.5]) == 'dependent' and rules.dependence([-1.0] * 17 + [0.0]) == 'dependent'      # 0도 부호가 바뀐 것으로 본다
    assert rules.dependence([-1.0] * 17 + [None]) == 'undetermined' and rules.dependence([-1.0] * 17) == 'undetermined'
    assert rules.dependence([0.0] * 18) == 'stable'


@test
def table_counts_ties_and_missing_separately():
    v = {2014: basket_values(-1, -2), 2016: basket_values(-1, 2), 2018: basket_values(1, -2), 2020: basket_values(1, 2),
         2022: basket_values(0.0, 1), 2024: basket_values(None, 1, a_all=[None] * 18)}
    s = rules.summarize(v, {})
    assert s['table']['cells'] == {'neg/neg': [2014], 'neg/pos': [2016], 'pos/neg': [2018], 'pos/pos': [2020]}
    assert s['table']['tie'] == [2022] and s['table']['missing'] == [2024]
    assert s['a_counts']['undetermined'] == [2024] and s['labels'][2022]['a'] == {'sign': 'tie', 'dependence': 'stable'}
    assert s['sentence']['kind'] == 'counts' and '2개' in s['sentence']['text']


@test
def sentences_follow_the_fixed_rules():
    years = sb.YEARS
    all_neg = rules.summarize({y: basket_values(-1, 1) for y in years}, {})
    assert all_neg['sentence'] == {'kind': 'all_neg', 'text': rules.S_ALL_NEG + '.'}
    all_pos = rules.summarize({y: basket_values(1, -1 if y in (2016, 2022) else 1) for y in years}, {})
    assert all_pos['sentence']['kind'] == 'all_pos' and all_pos['sentence']['x_below_k'] == [2016, 2022] and rules.S_POS_NOTE in all_pos['sentence']['text']
    assert rules.S_POS_NOTE not in rules.summarize({y: basket_values(1, 1) for y in years}, {})['sentence']['text']
    five = {y: basket_values(-1, 1) for y in years}; five[2020] = basket_values(2, 1)
    assert rules.summarize(five, {})['sentence']['kind'] == 'mostly_neg' and '대체로' in rules.summarize(five, {})['sentence']['text']
    dep = dict(five); dep[2014] = basket_values(-1, 1, a_all=[-1.0] * 17 + [0.3])
    assert rules.summarize(dep, {})['sentence']['kind'] == 'counts' and '대체로' not in rules.summarize(dep, {})['sentence']['text']
    dep_all_neg = {y: basket_values(-1, 1) for y in years}; dep_all_neg[2018] = basket_values(-1, 1, a_all=[-1.0] * 17 + [0.0])
    assert rules.summarize(dep_all_neg, {})['sentence']['kind'] == 'counts'            # 여섯 모두 음수여도 가정 의존이 있으면 '모두'를 쓰지 않는다
    four = dict(five); four[2022] = basket_values(2, 1)
    assert rules.summarize(four, {})['sentence']['kind'] == 'counts'
    b_dep = {y: basket_values(-1, 1, b_all=[1.0] * 17 + [-1.0]) for y in years}        # b의 가정 의존은 a 문장을 막지 않고 따로 센다
    s = rules.summarize(b_dep, {}); assert s['sentence']['kind'] == 'all_neg' and s['b_counts']['dependent'] == list(years)
    assert rules.summarize({y: basket_values(-1, 1) for y in years[:5]}, {})['sentence']['kind'] == 'counts'   # 다섯 바구니뿐이면 '모두'를 쓰지 않는다


@test
def aux_direction_label():
    v = {2014: basket_values(-1, 1), 2016: basket_values(-1, 1), 2018: basket_values(1, 1)}
    f = rules.aux_flags(rules.summarize(v, {})['labels'], {2014: {'r44': -0.5, 'r10': -2.0}, 2016: {'r44': 0.2, 'r10': -1.0}, 2018: {'r44': 1.0, 'r10': 0.0}})
    assert f[2014]['label'] == '' and f[2016] == {'main_sign': 'neg', 'aux_signs': {'r10': 'neg', 'r44': 'pos'}, 'differs': ['r44'], 'ties': [], 'label': rules.AUX_LABEL}
    assert f[2018]['differs'] == [] and f[2018]['ties'] == ['r10'] and f[2018]['label'] == ''          # r2: 동률은 방향 변화로 보지 않는다


@test
def refusal_conditions():
    need = ['a', 'b', 'c']
    assert rules.refusal_reasons({'a': [0], 'b': [0], 'c': [0]}, need) == []
    assert rules.refusal_reasons({'a': [0], 'b': [1, 0], 'c': [0]}, need) == []                       # 재개 뒤 성공
    assert rules.refusal_reasons({'a': [0], 'b': [0, 1], 'c': [0]}, need) == ['job failed: b exit=1']
    assert rules.refusal_reasons({'a': [0], 'c': [3]}, need) == ['job not recorded: b', 'job failed: c exit=3']
    assert rules.refusal_reasons({'a': [0], 'b': [0], 'c': [0], 'z': [0]}, need) == ['status for unknown job: z']


# ---------------- 조립(합성 산출물) ----------------
def fake_x(days, final, cash0=sb.CASH0, n_eligible=3.0):
    v = np.linspace(cash0, final, len(days)); v[0] = cash0
    curve = pd.DataFrame({'value_engine': v, 'receivable': 0.0, 'value': v}, index=days)
    state = pd.DataFrame({'engine_value': v, 'cash': v * 0.4, 'n_tickers': 5, 'n_lots': 7, 'n_full': 0, 'max_lots': 2}, index=days)
    fills = pd.DataFrame({'date': [days[1], days[70], days[70], days[90]], 'ticker': ['A', 'A', 'A', 'B'], 'type': ['BUY', 'SELL', 'SELL', 'SELL'],
                          'dq': [1, -1, -1, -1], 'dcash': [0.0] * 4, 'order': [1, 1, 2, 1], 'reason': ['신규 진입', chk_reason('inactive'), chk_reason('inactive'), chk_reason('stop')]})
    el = pd.DataFrame({'signal_date': '', 'tier_source': 'TIER_1', 'n_candidates': 40.0, 'n_eligible': n_eligible, 'n_held': 5, 'n_entry_signals': 0}, index=days)
    el.iloc[5:8, el.columns.get_loc('n_eligible')] = 0.0
    return {'curve': curve, 'state': state, 'fills': fills, 'eligible': el,
            'sbmeta': {'delisted_zeroed': {}, 'allowed': ['A', 'B', 'C', 'D'], 'start_unaffordable': ['C'], 'ever_candidate': ['A', 'B', 'D']}}


def chk_reason(kind):
    import same_basket_verdict as vd
    return vd.REASON_INACTIVE if kind == 'inactive' else vd.REASON_STOP + ' (평균가 기준)'


def fake_hk(days, final_h, final_k):
    v = np.linspace(sb.CASH0, final_h, len(days))
    h = {'curve': pd.DataFrame({'value': v, 'cash': v * 0.02, 'receivable': 0.0, 'stock_value': v * 0.98, 'n_held': 20}, index=days),
         'meta': {'skipped_at_start': [], 'delisted': {}}}
    return h, {'curve': pd.Series(np.linspace(sb.CASH0, final_k, len(days)), index=days)}


def fake_outs(jobs, cal, finals, flips=()):
    """finals: {바구니: (X, H, K 종료 평가액)}. flips: X를 다른 값으로 바꿀 (바구니, k, 배당, 상폐, X 값). 보조 계좌는 종목당 X/20."""
    outs = {}
    for job in jobs.itertuples():
        days = cal[(cal >= sb.prev_day(cal, job.start)) & (cal <= pd.Timestamp(job.end))]
        x, h, k = finals[job.basket]
        x = next((f[4] for f in flips if f[:4] == (job.basket, job.k, job.div, job.delist) and job.kind == 'main'), x)
        if job.kind == 'aux':
            share = (x if job.ratio == 'r44' else 2 * h - x) / 20          # r10은 H를 기준으로 방향을 뒤집는다
            outs[job.prefix] = {'x': fake_x(days, share, cash0=sb.SLOT_CASH)}
        else:
            hh, kk = fake_hk(days, h, k); outs[job.prefix] = {'x': fake_x(days, x), 'h': hh, 'k': kk}
    return outs


@test
def assembly_counts_assumption_dependence_and_publishes():
    import same_basket_verdict as vd
    cal, jobs = calendar(), fake_jobs()
    finals = {2014: (11e6, 12e6, 13e6), 2016: (11e6, 12e6, 10e6), 2018: (13e6, 12e6, 14e6), 2020: (13e6, 12e6, 11e6), 2022: (11e6, 12e6, 13e6), 2024: (11e6, 12e6, 13e6)}
    flips = [(2016, 20, 'none', 'zero', 12.5e6), (2022, 5, 'upper', 'freeze', 13.5e6)]     # 2016: a만 뒤집힘, 2022: a와 b 모두 뒤집힘
    tables, s = vd.assemble(jobs, fake_outs(jobs, cal, finals, flips), cal)
    assert s['table']['cells'] == {'neg/neg': [2014, 2022, 2024], 'neg/pos': [2016], 'pos/neg': [2018], 'pos/pos': [2020]}
    assert s['a_counts']['dependent'] == [2016, 2022] and s['b_counts']['dependent'] == [2022] and s['sentence']['kind'] == 'counts'
    assert s['labels'][2014]['a'] == {'sign': 'neg', 'dependence': 'stable'}
    assert all(s['aux'][y]['differs'] == ['r10'] and s['aux'][y]['label'] == rules.AUX_LABEL for y in sb.YEARS)
    rows = tables['summary_jobs']
    assert len(rows) == 114 and len(tables['aux']) == 12 and len(tables['summary_baskets']) == 6 and len(tables['extension']) == 11
    r = rows[(rows.basket == 2014) & (rows.kind == 'main')].iloc[0]
    want = ((11e6 / 1e7) ** (365.25 / (pd.Timestamp('2015-12-31') - pd.Timestamp('2013-12-31')).days) - 1) * 100
    assert abs(r.x_cagr - want) < 1e-9 and r.x_pnl == 1e6 and (r.x_inactivity_exits, r.x_stop_losses, r.x_fills) == (1, 1, 4)
    assert (r.x_never_bought, r.x_never_bought_order_below_one_share, r.x_never_bought_never_tier1_candidate, r.x_never_bought_other) == (3, 1, 0, 2)
    assert abs(r.a - (r.x_cagr - r.h_cagr)) < 1e-12 and abs(r.b - (r.x_cagr - r.k_cagr)) < 1e-12 and abs(r.x_avg_invested - 0.6) < 1e-9
    b = tables['summary_baskets'].set_index('basket')
    assert b.loc[2014, 'days_no_eligible'] == 3 and b.loc[2016, 'a_dependence'] == 'dependent' and b.loc[2016, 'aux_label'] == rules.AUX_LABEL
    e = tables['extension']; assert set(e[e.basket == 2024].point) == {'end'} and set(e[e.basket == 2014].point) == {'end_2023', 'end'}
    assert set(tables['eligibility_by_year'][tables['eligibility_by_year'].basket == 2022].year) == {2022, 2023, 2024, 2025}
    clean = {y: (11e6, 12e6, 13e6) for y in sb.YEARS}
    assert vd.assemble(jobs, fake_outs(jobs, cal, clean), cal)[1]['sentence']['kind'] == 'all_neg'
    tmp = tempfile.mkdtemp()
    try:
        vd.publish(tmp, 'smoke', tables, s)
        assert sorted(os.listdir(tmp)) == ['smoke_tables', 'verdict_smoke.json'] and json.load(open(os.path.join(tmp, 'verdict_smoke.json')))['official'] is False
        vd.publish(tmp, 'formal', tables, s)
        md = open(os.path.join(tmp, 'VERDICT.md')).read(); v = json.load(open(os.path.join(tmp, 'verdict.json')))
        assert s['sentence']['text'] in md and rules.NOTE_OVERLAP in md and v['official'] is True and v['table']['cells']['neg/pos'] == [2016]
    finally: shutil.rmtree(tmp)


@test
def cross_checks_catch_differences():
    cal, jobs = calendar(), fake_jobs()
    main = first(jobs[(jobs.kind == 'main') & (jobs.basket == 2014)]); ext = first(jobs[(jobs.kind == 'ext') & (jobs.basket == 2014)])
    days = lambda j: cal[(cal >= sb.prev_day(cal, j.start)) & (cal <= pd.Timestamp(j.end))]
    def out_for(j, slope):
        d = days(j); v = sb.CASH0 + slope * np.arange(len(d))
        x = fake_x(d, 0); x['curve'] = pd.DataFrame({'value_engine': v, 'receivable': 0.0, 'value': v}, index=d)
        x['state'] = pd.DataFrame({'engine_value': v, 'cash': v, 'n_tickers': 0}, index=d)
        x['fills'] = pd.DataFrame({'date': [d[1]], 'ticker': ['A'], 'type': ['BUY'], 'dq': [1], 'dcash': [-1.0]})
        h, k = fake_hk(d, 0, 0); h['curve'] = pd.DataFrame({'value': v}, index=d); k['curve'] = pd.Series(v, index=d)
        return {'x': x, 'h': h, 'k': k}
    assert chk.cut_errors(main, out_for(main, 10), out_for(ext, 10)) == []
    assert len(chk.cut_errors(main, out_for(main, 10), out_for(ext, 11))) == 4
    late = out_for(ext, 10); late['x']['fills'] = late['x']['fills'].assign(dq=2)
    assert chk.cut_errors(main, out_for(main, 10), late) == [f'{main.prefix}: X fills of the extension run differ from the 2-year run before the cut']
    assert chk.no_delisting_errors(main, out_for(main, 10), out_for(main, 10)) == [] and chk.no_delisting_errors(main, out_for(main, 10), out_for(main, 9))
    gone = out_for(main, 9); gone['h']['meta']['delisted'] = {'A': '2015-01-02'}
    assert chk.no_delisting_errors(main, out_for(main, 10), gone) == []
    g = jobs[(jobs.kind == 'aux') & (jobs.basket == 2014) & (jobs.ratio == 'r44')]
    accounts = [{'x': fake_x(days(j), 6e5, cash0=sb.SLOT_CASH)} for j in g.itertuples()]
    for o in accounts: o['x']['state'] = o['x']['state'].assign(cash=lambda s: s.engine_value)
    assert aux.group_errors('g', list(g.itertuples()), accounts, fake_baskets()[2014]) == []
    assert aux.group_errors('g', list(g.itertuples())[:19], accounts[:19], fake_baskets()[2014])       # 계좌 하나가 빠지면 원금 합이 맞지 않는다
    total, _ = aux.aggregate(accounts); assert total.iloc[0] == sb.CASH0 and abs(total.iloc[-1] - 12e6) < 1e-6


# ---------------- X 기술 검사 ----------------
def real_x(job, cal, tickers):
    """검사를 통과하는 합성 X 산출물: 시작일에 한 종목 매수, 배당 한 번 입금."""
    d = cal[(cal >= sb.prev_day(cal, job.start)) & (cal <= pd.Timestamp(job.end))]; n = len(d)
    cash = np.full(n, sb.CASH0); cash[1:] -= 200030.0; cash[100:] += 5000.0
    stock = np.zeros(n); stock[1:] = 200000.0
    recv = np.zeros(n); recv[50:100] = 5000.0
    curve = pd.DataFrame({'value_engine': cash + stock, 'receivable': recv, 'value': cash + stock + recv}, index=d)
    state = pd.DataFrame({'engine_value': cash + stock, 'cash': cash, 'n_tickers': (stock > 0).astype(int), 'n_lots': (stock > 0).astype(int), 'n_full': 0, 'max_lots': 1}, index=d)
    fills = pd.DataFrame({'date': [d[1]], 'ticker': [tickers[0]], 'type': ['BUY'], 'dq': [20], 'dcash': [-200030.0], 'order': [1], 'reason': ['신규 진입']})
    divs = pd.DataFrame({'fy': [2014, 2014], 'event': ['receivable', 'paid'], 'date': [str(d[50].date()), str(d[100].date())], 'amount': [5000.0, 5000.0]})
    el = pd.DataFrame({'signal_date': [''] + [str(x.date()) for x in d[:-1]], 'tier_source': ['NO_SIGNAL_DATE', 'TIER_1'] + ['NOT_EVALUATED'] * (n - 2),
                       'n_candidates': np.nan, 'n_eligible': np.nan, 'n_held': 1, 'n_entry_signals': [0, 1] + [0] * (n - 2)}, index=d)
    meta = {'success': True, 'degraded_run': False, 'universe_mode': 'strict_pit', 'env_universe_mode': 'strict_pit', 'candidate_lookup_error_count': 0, 'source_lookup_error_days': 0}
    sbmeta = {'account_start': job.start, 'engine_start': str(d[0].date()), 'end': job.end, 'warmup': True, 'restrict': True, 'eligible_all': False, 'delist': job.delist,
              'basket_sha256': 'h', 'only': None, 'swr_args': chk.SWR_ARGS(job.div), 'fills_before_start': 0, 'buys_outside_allowed': [], 'signal_ready': True,
              'initial_cash': sb.CASH0, 'order_investment_ratio': chk.MAIN_RATIO, 'delisted_zeroed': {}, 'basket_delisted': {}, 'ever_candidate': list(tickers),
              'start_unaffordable': [], 'start_no_open': [], 'allowed': list(tickers)}
    return {'curve': curve, 'state': state, 'fills': fills, 'divs': divs, 'eligible': el, 'meta': meta, 'sbmeta': sbmeta,
            'endpos': pd.DataFrame({'ticker': [tickers[0]], 'quantity': [20], 'lots': [1], 'price_used': [10000.0]})}


@test
def x_checks_pass_clean_run_and_catch_each_fault():
    cal, jobs, tickers = calendar(), fake_jobs(), fake_baskets()[2014]
    job = first(jobs); check = lambda run, j=job: chk.x_errors(j, run, cal, tickers, 'h')
    good = real_x(job, cal, tickers)
    assert check(good) == [], check(good)
    def changed(part, fn):
        run = real_x(job, cal, tickers); run[part] = fn(run[part]); return run
    def has(run, word): return any(word in e for e in check(run))
    assert has(changed('fills', lambda f: f.assign(ticker='ZZZ999')), 'outside the basket')                                  # 10절 2
    assert has(changed('eligible', lambda e: e.assign(signal_date=[''] + ['2000-01-01'] * (len(e) - 1))), 'prior trading day')   # 10절 3
    assert has(changed('eligible', lambda e: e.assign(tier_source=['NO_SIGNAL_DATE'] + ['NOT_EVALUATED'] * (len(e) - 1))), 'not prepared')
    assert has(changed('fills', lambda f: f.assign(date=cal[cal < pd.Timestamp(job.start)][-1])), 'fill before start')
    assert has(changed('state', lambda s: s.assign(cash=s.cash + (s.index == s.index[30]) * 7.0)), 'cash does not reconcile')
    assert has(changed('endpos', lambda p: p.assign(quantity=19)), 'end positions')
    assert has(changed('curve', lambda c: c.iloc[:-1]), 'dates differ')
    assert has(changed('curve', lambda c: c.assign(value=c.value.where(c.index != c.index[9]))), 'missing or infinite')
    assert has(changed('meta', lambda m: {**m, 'universe_mode': 'optimistic_survivor'}), 'strict_pit')
    assert has(changed('meta', lambda m: {**m, 'degraded_run': True}), 'degraded')
    for key, val in (('delist', 'zero'), ('swr_args', ['--tax', 'fixed', '--div', 'approx', '--hold', 'none']), ('signal_ready', False), ('initial_cash', 5e5),
                     ('basket_sha256', 'x'), ('restrict', False), ('engine_start', job.start)):
        assert has(changed('sbmeta', lambda m: {**m, key: val}), f'sbmeta {key}') or has(changed('sbmeta', lambda m: {**m, key: val}), key), key
    assert has(changed('sbmeta', lambda m: {**m, 'basket_delisted': {tickers[0]: str(cal[cal < pd.Timestamp(job.start)][-1].date())}}), 'after delisting')
    assert has(changed('sbmeta', lambda m: {k: v for k, v in m.items() if k != 'ever_candidate'}), 'ever_candidate missing')
    ext = first(jobs[(jobs.kind == 'ext') & (jobs.basket == 2014)])
    run = real_x(ext, cal, tickers); run['sbmeta']['eligible_all'] = True
    assert any('unevaluated days' in e for e in chk.x_errors(ext, run, cal, tickers, 'h'))                                  # 연장 계좌는 매일 기록해야 한다


# ---------------- 감싸기(엔진 클래스 위에서) ----------------
@test
def wrappers_filter_candidates_and_zero_delisted_prices():
    import same_basket_run as sr
    from src.data_handler import DataHandler
    from src.backtest.cpu.portfolio import Portfolio, Position
    from src.analysis import walk_forward_analyzer as wfa
    keep = (DataHandler.get_candidates_with_tier_fallback_pit_gated, Portfolio.get_total_value, wfa.run_cpu_single_backtest)
    try:
        source = {'v': 'TIER_1_SNAPSHOT_ASOF'}
        DataHandler.get_candidates_with_tier_fallback_pit_gated = lambda self, date, **kw: (['A', 'B', 'C'], source['v'])
        dh = object.__new__(DataHandler); seen = {}
        lookup0 = sr.install_basket_filter({'B', 'Z'}, True, seen)
        assert dh.get_candidates_with_tier_fallback_pit_gated('2020-01-02') == (['B'], 'TIER_1_SNAPSHOT_ASOF')            # 교차가 순위식보다 먼저다
        assert seen['2020-01-02'] == {'tier_source': 'TIER_1_SNAPSHOT_ASOF', 'n_candidates': 3, 'n_eligible': 1} and lookup0(dh, '2020-01-02')[0] == ['A', 'B', 'C']
        source['v'] = 'TIER_2_FALLBACK_SNAPSHOT_ASOF'
        dh.get_candidates_with_tier_fallback_pit_gated('2020-01-03'); assert seen['2020-01-03']['n_eligible'] == 0
        DataHandler.get_candidates_with_tier_fallback_pit_gated = keep[0]
        DataHandler.get_candidates_with_tier_fallback_pit_gated = lambda self, date, **kw: (['A', 'B', 'C'], 'TIER_1')
        sr.install_basket_filter({'B'}, False, {})
        assert dh.get_candidates_with_tier_fallback_pit_gated('2020-01-02')[0] == ['A', 'B', 'C']                          # 제한 끄기
        pf = Portfolio(1000.0, '2020-01-01', '2020-12-31')
        for t, q in (('A', 3), ('B', 5)):
            p = Position(100.0, q, 1, 0.06, 0.16); pf.positions[t] = [p]
        class Prices:
            def get_latest_prices(self, date, tickers, s, e): return {t: 100.0 for t in tickers}
        zeroed = {}
        sr.install_delist_zero({'B': pd.Timestamp('2020-02-28')}, zeroed)                                                   # 폐지일 = 마지막 상장일
        assert pf.get_total_value(pd.Timestamp('2020-02-28'), Prices()) == 1800.0 and not zeroed
        assert pf.get_total_value(pd.Timestamp('2020-03-02'), Prices()) == 1300.0 and zeroed == {'B': '2020-03-02'}        # 폐지일부터 0원(주문 예산도 이 값을 쓴다)
        got = {}
        wfa.run_cpu_single_backtest = lambda base, **kw: got.update(kw) or ('curve', {})
        used = {}
        sr.install_account_override(5e5, 0.44, used)
        wfa.run_cpu_single_backtest({}, start_date='a', end_date='b', params_dict={'order_investment_ratio': 0.022, 'max_stocks': 20}, initial_cash=1e7)
        assert got['initial_cash'] == 5e5 and got['params_dict'] == {'order_investment_ratio': 0.44, 'max_stocks': 20} and used['order_investment_ratio'] == 0.44
        rows = [{'date': '2020-01-02', 'signal_date': '', 'tier_source': 'NO_SIGNAL_DATE'}, {'date': '2020-01-03', 'signal_date': '2020-01-02', 'tier_source': 'TIER_1'}]
        assert sr.signal_ready(rows, pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-02'))
        assert not sr.signal_ready(rows, pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-01')) and not sr.signal_ready(rows[:1], pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-02'))
        assert not sr.signal_ready([rows[0], {**rows[1], 'tier_source': 'NOT_EVALUATED'}], pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-02'))
    finally:
        DataHandler.get_candidates_with_tier_fallback_pit_gated, Portfolio.get_total_value, wfa.run_cpu_single_backtest = keep



# ================= r2 (review-reconciled-impl.md 고칠 것 1~6, 덧붙임 1) =================
R2_TESTS = []


def r2(fn):
    R2_TESTS.append(fn); TESTS.append(fn); return fn


@r2
def r2_fix1_fee_uses_integer_formula():
    # 30만 원 매수 수수료: 300,000 × 3 // 20,000 = 45원(실수 곱셈은 44원). 2만 원짜리 1주에 필요한 돈은 20,003원
    assert ledger.buy_cost(1, 300000) == 300045, ledger.buy_cost(1, 300000)
    assert ledger.max_shares(20002, 20000) == 0 and ledger.max_shares(20003, 20000) == 1
    assert ledger.max_shares(300045, 300000) == 1 and ledger.max_shares(300044, 300000) == 0
    for gross in (1, 6666, 6667, 13333, 13334, 299999, 300000, 491960, 1234567):
        assert ledger.buy_cost(1, gross) - gross == gross * 3 // 20000, gross
    import same_basket_smoke_check as smk
    assert smk.hand_buy(20000, 20002) == (20000, 0, 0) and smk.hand_buy(20000, 20003) == (20000, 1, 3)
    assert smk.hand_buy(299999, 500000) == (300000, 1, 45) and smk.hand_buy(10037, 500000) == (10040, 49, 73)


def fake_dividend_sources(dps_rows, ratios):
    import stage2_common as sc
    keep = (sc.dps_table, sc.adj_ratio_on)
    sc.dps_table = lambda conn, tickers, years: pd.DataFrame([r for r in dps_rows if r[1] in years], columns=['ticker', 'fy', 'dps', 'dps_fallback'])
    sc.adj_ratio_on = lambda conn, tickers, day: {t: v for t, v in ratios.items() if t in list(tickers)}
    return sc, keep


@r2
def r2_fix2_dividend_inputs_are_classified_and_missing_ratio_stops():
    cal = __import__('stage2_common').Calendar(calendar()); dates = cal.dates[(cal.dates >= '2013-12-31') & (cal.dates <= '2014-12-31')]
    rows = [('AAA001', 2014, 150.0, False), ('BBB002', 2014, 70.0, False), ('CCC003', 2014, 0.0, False), ('EEE005', 2014, 40.0, False)]
    sc, keep = fake_dividend_sources(rows, {'AAA001': (2.0, 5000.0), 'CCC003': (1.0, 100.0), 'DDD004': (1.0, 100.0)})
    try: ev = ledger.dividend_events(None, cal, ['AAA001', 'BBB002', 'CCC003', 'DDD004', 'EEE005'], dates, 1, delisted={'EEE005': pd.Timestamp('2014-06-02')})
    finally: sc.dps_table, sc.adj_ratio_on = keep
    status = {t: v['status'] for t, v in ev[0]['per_ticker'].items()}
    assert status == {'AAA001': 'ok', 'BBB002': 'ratio_missing', 'CCC003': 'no_dividend', 'DDD004': 'no_dividend', 'EEE005': 'delisted'}, status
    n = len(dates); inp = {'dates': [str(d.date()) for d in dates], 'tickers': ['AAA001', 'BBB002', 'CCC003', 'DDD004', 'EEE005'], 'open': np.full((n, 5), 10037.0),
                           'close': np.full((n, 5), 10000.0), 'start_i': 1, 'events': ev, 'delist_i': [None] * 4 + [int(dates.searchsorted(pd.Timestamp('2014-06-02')))]}
    assert raises(lambda: ledger.run_hold(inp, 'approx', 'freeze'), ValueError)            # 배당 있는 보유분을 원주로 환산할 수 없으면 중단
    assert not raises(lambda: ledger.run_hold(inp, 'none', 'freeze'))                       # 배당 없음 가정에서는 배당 입력을 쓰지 않는다
    ok = dict(inp, events=[{**ev[0], 'per_ticker': {**ev[0]['per_ticker'], 'BBB002': {**ev[0]['per_ticker']['BBB002'], 'status': 'no_dividend'}}}])
    s = ledger.run_hold(ok, 'approx', 'freeze')
    assert [d['ticker'] for d in s['divs']] == ['AAA001']
    assert {r['ticker']: r['reason'] for r in s['div_skips']} == {'BBB002': 'no_dividend', 'CCC003': 'no_dividend', 'DDD004': 'no_dividend', 'EEE005': 'delisted'}
    unclassified = dict(inp, events=[{**ev[0], 'per_ticker': {t: v for t, v in ev[0]['per_ticker'].items() if t != 'CCC003'}}])
    assert raises(lambda: ledger.run_hold(unclassified, 'approx', 'freeze'), ValueError)        # 분류되지 않은 보유 종목도 중단
    import same_basket_run as sr
    sc, keep = fake_dividend_sources(rows, {'AAA001': (2.0, 5000.0), 'CCC003': (1.0, 100.0)})
    try:
        log = []
        sr.install_dividend_guard({'EEE005': pd.Timestamp('2014-06-02')}, log)
        sc.dps_table(None, ['AAA001', 'CCC003', 'EEE005'], [2013, 2014]); assert sc.adj_ratio_on(None, ['AAA001', 'CCC003', 'EEE005'], pd.Timestamp('2014-12-30'))
        assert {r['ticker']: r['status'] for r in log} == {'AAA001': 'ok', 'CCC003': 'no_dividend', 'EEE005': 'delisted'}
        sc.dps_table(None, ['AAA001', 'BBB002'], [2013, 2014])
        assert raises(lambda: sc.adj_ratio_on(None, ['AAA001', 'BBB002'], pd.Timestamp('2014-12-30')), RuntimeError)   # X도 같은 경우 중단
    finally: sc.dps_table, sc.adj_ratio_on = keep


@r2
def r2_fix3_delisting_blocks_trading_and_freezes_last_close():
    opens = [[np.nan], [10037], [10000], [10000], [12000]]; closes = [[10000], [10000], [9000], [9500], [12000]]   # 3일째 폐지. 폐지 뒤에도 가격 행이 있는 반례
    ev = [event(2019, 2, 4, AAA001=(1.0, 10000.0, 100.0, False))]
    for div in ('approx', 'upper'):
        for z in ('freeze', 'zero'):
            s = ledger.run_hold(hold_input(opens, closes, ev, delist=[3]), div, z); c = curve_of(s)
            assert not [b for b in s['buys'] if b['kind'] == 'reinvest' and b['date'] >= '2020-01-04'], (div, z)        # 폐지일 뒤 재투자 없음
            want = [0, 490000, 441000] + ([441000, 441000] if z == 'freeze' else [0, 0])                                # 동결 = 폐지 전 마지막 종가
            assert list(c.stock_value) == want, (div, z, list(c.stock_value))
    import same_basket_run as sr
    import continuous_account_run as car
    from src.backtest.cpu.portfolio import Portfolio, Position
    keep = (Portfolio.get_total_value, Portfolio._resolve_price_map)
    try:
        pf = Portfolio(1000.0, '2020-01-01', '2020-12-31')
        for t, q in (('A', 3), ('B', 5)): pf.positions[t] = [Position(100.0, q, 1, 0.06, 0.16)]
        class Prices:
            def get_latest_prices(self, date, tickers, s, e): return {t: 100.0 for t in tickers}
            def get_latest_price(self, date, t, s, e): return 100.0
        zeroed = {}; sr.install_delist_zero({'B': pd.Timestamp('2020-02-28')}, zeroed)
        day = pd.Timestamp('2020-03-02')
        assert pf.get_total_value(day, Prices()) == 1300.0
        ep = {r['ticker']: r['price_used'] for r in car.end_positions(pf, day, Prices(), None)}
        assert ep == {'A': 100.0, 'B': 0.0}, ep                                                                             # 마지막 보유 기록도 0원
    finally: Portfolio.get_total_value, Portfolio._resolve_price_map = keep


def aux_accounts(jobs, cal, seed=3):
    rng = np.random.default_rng(seed); out = []
    for j in jobs.itertuples():
        d = cal[(cal >= sb.prev_day(cal, j.start)) & (cal <= pd.Timestamp(j.end))]
        v = sb.SLOT_CASH * np.cumprod(np.r_[1.0, 1 + rng.normal(0, 0.01, len(d) - 1)]); cash = v * rng.uniform(0.1, 0.9)
        cash[0] = sb.SLOT_CASH
        x = {'curve': pd.DataFrame({'value_engine': v, 'receivable': 0.0, 'value': v}, index=d), 'state': pd.DataFrame({'engine_value': v, 'cash': cash}, index=d)}
        out.append({'x': x})
    return out


@r2
def r2_fix4_aux_sum_is_checked_independently_and_missing_is_not_a_direction_change():
    cal, jobs = calendar(), fake_jobs()
    g = jobs[(jobs.kind == 'aux') & (jobs.basket == 2014) & (jobs.ratio == 'r44')]
    accounts = aux_accounts(g, cal); basket = fake_baskets()[2014]
    assert aux.group_errors('g', list(g.itertuples()), accounts, basket) == []
    keep = aux.aggregate
    try:
        aux.aggregate = lambda outs: keep([outs[0]] + list(outs[:-1]))           # 마지막 계좌 대신 첫 계좌를 두 번 더하는 잘못된 합산
        assert any('independent' in e for e in aux.group_errors('g', list(g.itertuples()), accounts, basket))
    finally: aux.aggregate = keep
    labels = rules.summarize({2014: basket_values(-1, 1), 2016: basket_values(-1, 1), 2018: basket_values(-1, 1)}, {})['labels']
    f = rules.aux_flags(labels, {2014: {'r44': None, 'r10': -1.0}, 2016: {'r10': -1.0}, 2018: {'r44': 0.0, 'r10': 2.0}})
    assert f[2014]['label'] == rules.AUX_MISSING and f[2014]['differs'] == [] and f[2016]['label'] == rules.AUX_MISSING
    assert f[2018]['differs'] == ['r10'] and f[2018]['label'] == rules.AUX_LABEL and f[2018]['ties'] == ['r44']


@r2
def r2_fix3b_delisting_day_is_last_listed_day():
    import stage2_common as sc
    days = pd.bdate_range('2020-01-01', periods=6)
    keep = sc.q
    try:
        sc.q = lambda conn, sql, params: pd.DataFrame({'stock_code': ['AAA001'], 'delisted_date': [days[3]]})
        di, dl = ledger.delist_positions(None, ['AAA001', 'BBB002'], days)
    finally: sc.q = keep
    assert di == [4, None] and dl == {'AAA001': str(days[3].date())}                     # 폐지일(3) 다음 거래일(4)부터 적용
    info = ledger.classify_dividend('AAA001', {'AAA001': (10.0, False)}, {'AAA001': (1.0, 100.0)}, days[3], {'AAA001': days[3]})
    assert info['status'] == 'ok'                                                          # 폐지일이 권리일이면 아직 상장
    assert ledger.classify_dividend('AAA001', {'AAA001': (10.0, False)}, {'AAA001': (1.0, 100.0)}, days[4], {'AAA001': days[3]})['status'] == 'delisted'


@r2
def r2_add1_never_bought_reasons():
    got = chk.never_bought(['A', 'B', 'C', 'D'], bought={'A'}, unaffordable={'B', 'C'}, ever_candidate={'B', 'D'})
    assert got == {'B': 'order_below_one_share', 'C': 'order_below_one_share', 'D': 'other'}, got
    assert chk.never_bought(['A', 'E'], bought={'A'}, unaffordable=set(), ever_candidate={'A'}) == {'E': 'never_tier1_candidate'}


# ---------------- 조립기 거부(실제 실행, DB는 달력만) ----------------
def run_verdict(out, mode):
    p = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'same_basket_verdict.py'), out, '--mode', mode], capture_output=True, text=True)
    vdir = os.path.join(out, 'verdict')
    files = sorted(os.listdir(out)) + (sorted(os.listdir(vdir)) if os.path.isdir(vdir) else [])
    path = os.path.join(vdir, 'checks.json')
    return p.returncode, files, json.load(open(path))['errors'] if os.path.exists(path) else None


def status_file(out, jobs, fail=None, drop=None):
    rows = [(j.id, j.prefix, 1 if j.prefix == fail else 0, 1, 't') for j in jobs.itertuples() if j.prefix != drop]
    pd.DataFrame(rows, columns=['id', 'prefix', 'exit', 'seconds', 'finished_utc']).to_csv(os.path.join(out, 'job_status.tsv'), sep='\t', index=False)


def verdict_refuses_end_to_end():
    import stage2_common as sc
    cal = sc.trading_dates(sc.connect()); baskets = sb.load_baskets()
    jobs = sb.build_jobs(cal, baskets); tmp = tempfile.mkdtemp(); published = {'verdict.json', 'VERDICT.md', 'summary_jobs.csv', 'summary_baskets.csv', 'smoke_tables'}
    try:
        shutil.copytree(sb.BASKET_DIR, os.path.join(tmp, 'baskets')); jobs.to_csv(os.path.join(tmp, 'jobs.tsv'), sep='\t', index=False)
        cases = {'failed job': dict(fail=jobs.prefix[7]), 'unrecorded job': dict(drop=jobs.prefix[200]), 'all ok but no reproduction record and no outputs': {}}
        for name, kw in cases.items():
            status_file(tmp, jobs, **kw); rc, files, errs = run_verdict(tmp, 'formal')
            assert rc == 2 and errs and not published & set(files), (name, rc, files)
        assert any('reproduction check' in e for e in errs) and not any('job' in e for e in errs)
        os.makedirs(os.path.join(tmp, 'repro')); json.dump({'passed': True}, open(os.path.join(tmp, 'repro', 'repro_check.json'), 'w'))
        rc, files, errs = run_verdict(tmp, 'formal')
        assert rc == 2 and len(errs) == len(jobs) and all('cannot read outputs' in e for e in errs) and not published & set(files)   # 산출물이 없으면 거부
        jobs.iloc[:-1].to_csv(os.path.join(tmp, 'jobs.tsv'), sep='\t', index=False); status_file(tmp, jobs.iloc[:-1])
        rc, files, errs = run_verdict(tmp, 'formal')
        assert rc == 2 and any('job rows differ' in e for e in errs) and any('aux: 239 jobs' in e for e in errs)                     # 작업표가 승인 구성과 다르면 거부
        os.remove(os.path.join(tmp, 'job_status.tsv')); rc, files, errs = run_verdict(tmp, 'smoke')
        assert rc == 2 and errs == ['job_status.tsv unreadable (FileNotFoundError)']
    finally: shutil.rmtree(tmp)


# ================= r3 (review-reconciled-impl-r2.md 고칠 것 1·2·4. 3은 일괄 실행기 시험으로 따로) =================
R3_TESTS = []


def r3(fn):
    R3_TESTS.append(fn); TESTS.append(fn); return fn


def swr_dividend_block():
    """split_window_runs.py의 실제 배당 권리·지급 코드(sell_wrap 안 'if d in after_rec:' 블록과 지급 루프)를 소스에서 그대로 꺼낸다."""
    src = open(os.path.join(HERE, 'split_window_runs.py')).read().splitlines()
    i = next(k for k, l in enumerate(src) if l.strip() == 'if d in after_rec:     # 전날 종가 보유 = 권리 확정 보유')
    j = next(k for k, l in enumerate(src) if l.strip().startswith('return orig_sell('))
    import textwrap
    return textwrap.dedent('\n'.join(src[i:j]))


def run_swr_dividend(held, dps_rows, ratios, delisted, record, pay_day):
    """실제 X 배당 코드를 same_basket_run의 배당 가드를 씌운 stage2_common 함수로 실행한다. 반환: (권리 금액, 지급 뒤 현금, divs, 상태 기록)."""
    import stage2_common as sc, same_basket_run as sr
    keep = (sc.dps_table, sc.adj_ratio_on)
    try:
        sc.dps_table = lambda conn, tickers, years: pd.DataFrame([r for r in dps_rows if r[1] in years], columns=['ticker', 'fy', 'dps', 'dps_fallback'])
        sc.adj_ratio_on = lambda conn, tickers, day: {t: v for t, v in ratios.items() if t in list(tickers)}
        log = []; sr.install_dividend_guard(delisted, log)
        class Pf:
            def __init__(self): self.cash = 0.0; self.positions = {t: [type('P', (), {'quantity': q})()] for t, q in held.items()}
            def update_cash(self, x): self.cash += x
        class Cal:
            def shift(self, d, n): return record if n == -1 else d
        class A: div = 'approx'
        rights = record + pd.Timedelta(days=1); pf = Pf(); divs = []; pending = {}
        ns = {'a': A(), 'sc': sc, 'conn': None, 'cal': Cal(), 'portfolio': pf, 'divs': divs, 'pending': pending,
              'after_rec': {rights: 2014}, 'pay': {2014: pay_day}, 'pd': pd}
        block = swr_dividend_block()
        for d in (rights, pay_day):
            ns['d'] = d; exec(block, ns)
        recv = [e for e in divs if e.get('event') == 'receivable']
        return recv[0]['amount'], pf.cash, divs, log
    finally: sc.dps_table, sc.adj_ratio_on = keep


@r3
def r3_fix1_delisting_on_record_date_keeps_dividend_and_blocks_reinvest():
    # H: 권리일(rec = 2일째) 당일 폐지. 폐지일은 마지막 상장일이라 권리일 분류는 ok, 배당은 인정하고 지급일(5일째) 재투자는 막는다
    opens = [[np.nan], [10037], [10000], [10000], [10000], [4000], [4000]]; closes = [[10000]] * 7
    ev = [event(2019, 3, 5, AAA001=(1.0, 10000.0, 100.0, False))]                 # rights_i = 3 = 권리일 다음 거래일
    for div in ('approx', 'upper'):
        s = ledger.run_hold(hold_input(opens, closes, ev, delist=[3]), div, 'freeze')     # delist_i 3 = 폐지일(2) 다음 거래일
        assert [d['amount'] for d in s['divs']] == [4900.0], (div, s['divs'], s['div_skips'])
        assert not [b for b in s['buys'] if b['kind'] == 'reinvest'], div
        assert s['pending'] and list(s['pending'].values()) == [4900.0], div            # 재투자하지 못한 배당은 현금에 남는다
        assert curve_of(s).cash.iloc[-1] == sb.CASH0 - 492033 + 4900
    # X: 같은 날 폐지된 보유분의 권리를 인정한다
    rec = pd.Timestamp('2014-12-29')
    amt, cash, divs, log = run_swr_dividend({'AAA001': 49}, [('AAA001', 2014, 100.0, False)], {'AAA001': (1.0, 10000.0)}, {'AAA001': rec}, rec, pd.Timestamp('2015-05-18'))
    assert amt == 4900.0 and cash == 4900.0 and log[0]['status'] == 'ok', (amt, cash, log)
    amt, cash, _, log = run_swr_dividend({'AAA001': 49}, [('AAA001', 2014, 100.0, False)], {'AAA001': (1.0, 10000.0)}, {'AAA001': rec - pd.Timedelta(days=1)}, rec, pd.Timestamp('2015-05-18'))
    assert amt == 0.0 and cash == 0.0 and log[0]['status'] == 'delisted'              # 전날 폐지면 권리 없음


@r3
def r3_fix2_no_dividend_contributes_zero_in_x_even_with_nan_inputs():
    rec, pay_day = pd.Timestamp('2014-12-29'), pd.Timestamp('2015-05-18')
    cases = {'ratio nan, dps 0': ([('A', 2014, 0.0, False)], {'A': (np.nan, 10000.0)}),
             'dps nan, ratio ok': ([('A', 2014, np.nan, False)], {'A': (1.0, 10000.0)}),
             'dps nan, ratio nan': ([('A', 2014, np.nan, False)], {'A': (np.nan, np.nan)}),
             'dps 0, ratio ok': ([('A', 2014, 0.0, False)], {'A': (1.0, 10000.0)}),
             'dps negative': ([('A', 2014, -5.0, False)], {'A': (1.0, 10000.0)})}
    for name, (rows, ratios) in cases.items():
        amt, cash, divs, log = run_swr_dividend({'A': 10, 'B': 20}, rows + [('B', 2014, 50.0, False)], {**ratios, 'B': (2.0, 1000.0)}, {}, rec, pay_day)
        assert log[[r['ticker'] for r in log].index('A')]['status'] == 'no_dividend', name
        assert amt == 20 * 2.0 * 50.0 and cash == amt and np.isfinite(cash), (name, amt, cash)         # 무배당 기여 0, 지급까지 유한
        assert all(np.isfinite(e['amount']) for e in divs), name


@r3
def r3_fix4_never_bought_reason_name():
    got = chk.never_bought(['A', 'E', 'F'], bought={'A'}, unaffordable=set(), ever_candidate={'A', 'F'})
    assert got == {'E': 'never_tier1_candidate', 'F': 'other'}, got
    assert 'never_tier1_candidate' in chk.NEVER_BOUGHT_REASONS and 'tier_exit' not in chk.NEVER_BOUGHT_REASONS
    import same_basket_verdict as vd
    assert vd.NEVER_BOUGHT_NOTE and '한 번도 1등급 후보에 없었음' in vd.NEVER_BOUGHT_NOTE



# ================= r4 (review-reconciled-impl-r3.md 고칠 것 3. 1·2는 일괄 실행기 시험 r4_runall_test.sh) =================
R4_TESTS = []


def r4(fn):
    R4_TESTS.append(fn); TESTS.append(fn); return fn


@r4
def r4_fix3_generated_missing_table_title_matches_definition():
    import same_basket_verdict as vd
    cal, jobs = calendar(), fake_jobs()
    finals = {y: (11e6, 12e6, 13e6) for y in sb.YEARS}
    tables, s = vd.assemble(jobs, fake_outs(jobs, cal, finals), cal)
    empty = tables['summary_baskets'].iloc[0:0]
    for bdf in (tables['summary_baskets'], empty):
        md = vd.verdict_markdown(s, bdf, tables['aux'], tables['extension'], tables['eligibility_by_year'])
        title = next(l for l in md.splitlines() if l.startswith('### X가 기간 안에'))
        assert '등급 이탈' not in md, title
        assert title == vd.MISSING_TITLE and vd.NEVER_BOUGHT_NOTE in md
        assert '한 번도 1등급 후보에 없었음' in md


def r2_fix6_failed_recheck_removes_old_verdict():
    """재검사가 실패하면 예전 판정·표가 남지 않고 최신 검사 기록만 남는다(DB는 달력만)."""
    import stage2_common as sc
    cal = sc.trading_dates(sc.connect()); jobs = sb.build_jobs(cal, sb.load_baskets()); tmp = tempfile.mkdtemp()
    try:
        shutil.copytree(sb.BASKET_DIR, os.path.join(tmp, 'baskets')); jobs.to_csv(os.path.join(tmp, 'jobs.tsv'), sep='\t', index=False)
        for rel in ('verdict.json', 'VERDICT.md', 'summary_jobs.csv', 'verdict/verdict.json', 'verdict/VERDICT.md', 'verdict/summary_jobs.csv'):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True); open(os.path.join(tmp, rel), 'w').write('old official verdict')
        status_file(tmp, jobs, fail=jobs.prefix[3])
        p = subprocess.run([sys.executable, '-B', os.path.join(HERE, 'same_basket_verdict.py'), tmp, '--mode', 'formal'], capture_output=True, text=True)
        assert p.returncode == 2
        stale = [rel for rel in ('verdict.json', 'VERDICT.md', 'summary_jobs.csv', 'verdict/verdict.json', 'verdict/VERDICT.md', 'verdict/summary_jobs.csv') if os.path.exists(os.path.join(tmp, rel))]
        assert stale == [], stale
        assert sorted(os.listdir(os.path.join(tmp, 'verdict'))) == ['checks.json'] and json.load(open(os.path.join(tmp, 'verdict', 'checks.json')))['passed'] is False
    finally: shutil.rmtree(tmp)


def main():
    db = [] if '--no-db' in sys.argv else [verdict_refuses_end_to_end, r2_fix6_failed_recheck_removes_old_verdict]
    tests = (R2_TESTS + ([] if '--no-db' in sys.argv else [r2_fix6_failed_recheck_removes_old_verdict])) if '--r2-only' in sys.argv else TESTS + db
    if '--r3-only' in sys.argv: tests = R3_TESTS
    if '--r4-only' in sys.argv: tests = R4_TESTS
    failed = []
    for fn in tests:
        try: fn(); print('ok', fn.__name__)
        except Exception as exc:
            failed.append(fn.__name__); print('FAIL', fn.__name__, type(exc).__name__, str(exc)[:200])
    print(f'{len(tests) - len(failed)} tests passed, {len(failed)} failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
