"""연구 (b) 합성 자료 자체 검사(DESIGN-v2.md 5절 + DESIGN-v2.1.md A·B + 구현 리뷰 F1~F10 반례). 실패하면 종료 코드 1.

usage: python -B tools/research/pool_vs_index_selftest.py [--db]
3월 실행 체크아웃을 작업 디렉터리로 실행(패널 모듈이 src를 가져온다).
--db: 실제 스냅숏의 재등장 사례(036220, 101970)와 과거 호가 대조(F7)를 DB SELECT로 확인한다.
반례 출처: astra-impl-check/counterexamples.txt, integration-check.txt, executions-and-status.txt (테스트 이름 앞에 F 번호).
"""
import json, os, sys, tempfile, types
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pool_vs_index_common as pc
import pool_vs_index_accounting as ac

RESULTS = []
NO_COST = ac.Costs(0.0, False)


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail)); print(('PASS ' if cond else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))


def market(days, n, adj_open, adj_close, raw_open=None, raw_close=None, volume=None, ratio=None, dates=None):
    dates = dates if dates is not None else pd.bdate_range(days[0], periods=days[1])
    f = lambda x, d=1.0: np.array(x, dtype=float) if x is not None else np.full((len(dates), n), d)
    ao, acl = f(adj_open), f(adj_close)
    ro = f(raw_open) if raw_open is not None else ao.copy(); rc = f(raw_close) if raw_close is not None else acl.copy()
    vol = f(volume, 100.0); rt = f(ratio, 1.0)
    pos = lambda a: pd.DataFrame(np.where(np.isfinite(a) & (a > 0), a, np.nan))
    return ac.Market(dates, [f'T{j}' for j in range(n)], ao, pos(acl).ffill().values, ro, pos(rc).ffill().values, vol, pos(rt).ffill().values,
                     pos(acl).isna().values, pd.DatetimeIndex(dates), pos(rt).values)


def events(rebal, **kw):
    return ac.Events(rebal, kw.get('post', {}), kw.get('pre', {}), kw.get('record', {}), kw.get('pay', {}), kw.get('dps', {}),
                     order=kw.get('order', {}), markets=kw.get('markets', {}), marks=kw.get('marks', ()), log_fills=kw.get('fills', False))


def dps(vals, observed):
    return (np.array(vals, dtype=float), np.array(observed, dtype=bool))


FLAT_TICK = lambda p, m, d: float(p)


# ---------------- 기존 회계 검사 ----------------

def t_split_conservation():
    ao = [[50.0]] * 6; acl = [[50.0]] * 6
    ro = [[100.0]] * 3 + [[50.0]] * 3; rt = [[0.5]] * 3 + [[1.0]] * 3
    mk = market(('2020-01-01', 6), 1, ao, acl, ro, ro, ratio=rt)
    v, _ = ac.run_fractional(mk, events({1: np.array([1.0])}), NO_COST, False, 1000.0)
    check('split: fractional value preserved', np.allclose(v.values, 1000.0), str(v.values))
    vi, lg = ac.run_integer(mk, events({1: np.array([1.0])}, order={1: [0]}, markets={1: np.array(['KOSPI'])}), NO_COST, False, 1000.0, FLAT_TICK)
    check('split: integer value preserved', np.allclose(vi.values, 1000.0) and lg['ca'] and lg['ca'][0]['factor'] == 2.0, str(vi.values))


def t_suspension_across_boundary():
    n_days = 8; ao = np.full((n_days, 2), 10.0); ro = ao.copy(); vol = np.full((n_days, 2), 100.0)
    ro[3:6, 0] = 0.0; vol[3:6, 0] = 0.0
    mk = market(('2021-12-27', n_days), 2, ao, ao, ro, ao, volume=vol)
    w = {1: np.array([0.5, 0.5]), 4: np.array([0.0, 1.0]), 6: np.array([0.0, 1.0])}
    ye = int(np.where(mk.dates.year == 2021)[0][-1])
    v, lg = ac.run_fractional(mk, events(w, marks=(ye,)), NO_COST, False, 1000.0)
    o = lg['orders']
    check('suspension: frozen qty kept on rebalance', o[1]['n_frozen'] == 1 and abs(o[1]['frozen_book'] - 500) < 1e-9, str(o[1]))
    m = lg['marks'][0]
    check('suspension: year-end mark has frozen qty/book/weight/disposable', m['n_frozen'] == 1 and m['frozen_disposable'] == 0.0 and abs(m['frozen_weight'] - 0.5) < 1e-12)
    check('suspension: mark detail per ticker', any(x['kind'] == 'frozen' and x['ticker'] == 'T0' for x in lg['mark_detail']))
    check('suspension: sold after resume', np.allclose(v.values, 1000.0) and lg['orders'][2]['sell_gross'] > 0)


def t_open_attribution():
    ao = np.array([[10.0], [10.0], [12.0], [12.0]]); acl = np.array([[10.0], [10.0], [10.0], [12.0]])
    mk = market(('2020-01-01', 4), 1, ao, acl)
    v_hold, _ = ac.run_fractional(mk, events({1: np.array([1.0])}), NO_COST, False, 1000.0)
    v_new, _ = ac.run_fractional(mk, events({2: np.array([1.0])}), NO_COST, False, 1000.0)
    check('open attribution: holder earns close-to-open', abs(v_hold.iloc[-1] - 1200) < 1e-9)
    check('open attribution: open buyer does not', abs(v_new.iloc[-1] - 1000) < 1e-9 and abs(v_new.iloc[2] - 1000 * 10 / 12) < 1e-9)


def t_dividend_after_observation():
    n = 10; ao = np.full((n, 2), 10.0); rt = np.full((n, 2), 0.5)
    mk = market(('2020-12-28', n), 2, ao, ao, ratio=rt)
    w = {1: np.array([1.0, 0.0]), 4: np.array([0.0, 1.0])}
    v, lg = ac.run_fractional(mk, events(w, record={2: 2020}, pay={7: 2020}, dps={2020: dps([2.0, 0.0], [True, True])}), NO_COST, False, 1000.0)
    check('dividend: not in value before pay date', np.allclose(v.values[:7], 1000.0))
    check('dividend: paid at pay-date close from raw rights (sold before pay keeps rights)', abs(v.iloc[7] - 1100.0) < 1e-9, str(v.values))
    v2, lg2 = ac.run_fractional(mk, events({1: np.array([1.0, 0.0])}, record={2: 2020}, dps={2020: dps([2.0, 0.0], [True, True])}), NO_COST, False, 1000.0)
    check('dividend: unpaid rights excluded at end', np.allclose(v2.values, 1000.0) and lg2['unpaid_rights_fy'] == [2020])


def t_cash_nonnegative_after_costs():
    ao = np.full((3, 3), 7.0); mk = market(('2020-01-01', 3), 3, ao, ao)
    costs = ac.Costs(0.00015, True, lambda d: 0.002)
    v, lg = ac.run_fractional(mk, events({1: np.array([0.4, 0.3, 0.3]), 2: np.array([0.0, 0.5, 0.5])}), costs, False, 1000.0)
    o = lg['orders']
    check('costs: cash >= 0 and buys scaled together', all(x['cash_after'] >= 0 for x in o) and o[0]['buy_scale'] < 1.0)
    check('costs: conservation (value lost = fees + tax)', all(abs(x['conservation_err']) < 1e-9 for x in o))


def t_frozen_budget_example():
    ao = np.full((3, 2), 10.0); ro = ao.copy(); ro[2, 0] = 0.0
    mk = market(('2020-01-01', 3), 2, ao, ao, ro)
    st = ac.State(q=np.array([4.0, 0.0]), cash=60.0)
    ac.rebalance(st, mk, 2, np.array([0.5, 0.5]), NO_COST)
    check('A1 example (v2.1): B buys 50, cash 10, A kept', abs(st.q[1] - 5.0) < 1e-12 and abs(st.cash - 10.0) < 1e-12 and st.q[0] == 4.0)


def t_zero_candidate_month():
    ao = np.full((3, 2), 10.0); mk = market(('2020-01-01', 3), 2, ao, ao)
    v, lg = ac.run_fractional(mk, events({1: np.array([0.5, 0.5]), 2: np.zeros(2)}), NO_COST, False, 1000.0)
    check('zero candidates: all tradable sold to cash', abs(lg['orders'][1]['cash_after'] - 1000) < 1e-9)


def t_delist_generations():
    ao = np.full((6, 1), 10.0); ro = ao.copy(); ro[2:4, 0] = 0.0
    mk = market(('2020-01-01', 6), 1, ao, ao, ro)
    w = {1: np.array([0.5]), 4: np.array([1.0])}
    for zero in (False, True):
        v, lg = ac.run_fractional(mk, events(w, post={2: np.array([0])}, marks=(5,)), NO_COST, zero, 1000.0)
        tag = 'zero' if zero else 'last'; dead = 0.0 if zero else 500.0
        check(f'delist[{tag}]: exit recorded, generation 0, post', len(lg['exits']) == 1 and lg['exits'][0]['generation'] == 0 and lg['exits'][0]['when'] == 'post')
        check(f'delist[{tag}]: value at exit close (no cash inflow)', abs(v.iloc[2] - (500.0 + dead)) < 1e-9)
        o = lg['orders'][1]
        check(f'delist[{tag}]: old generation in budget, not revived; new buy from cash only',
              abs(o['total_at_open'] - (500.0 + dead)) < 1e-9 and abs(o['buy_gross'] - 500.0) < 1e-9 and abs(v.iloc[-1] - (500.0 + dead)) < 1e-9)
        det = [x for x in lg['mark_detail'] if x['kind'] == 'dead']
        check(f'delist[{tag}]: mark detail has dead generation residual', len(det) == 1 and det[0]['generation'] == 0 and abs(det[0]['qty'] - 50) < 1e-9)


def t_cells_and_terciles():
    import pool_vs_index_sim as sim
    import pool_vs_index_data as dat
    u = pd.DataFrame({'stock_code': [f'{i:06d}' for i in range(10)], 'market': ['KOSPI'] * 7 + ['KOSDAQ'] * 3,
                      'market_cap': [7, 6, 5, 4, 3, 2, 1, 30, 20, 10]})
    t = dat.tercile(u)
    sizes = t[u.market == 'KOSPI'].value_counts().sort_index().tolist()
    check('tercile: 7 -> 3,2,2 (smallest cell first)', sizes == [3, 2, 2], str(sizes))
    u['tercile_U'] = t; u['in_U'] = True; u['p1_rank'] = 0; u.loc[[0, 6, 9], 'p1_rank'] = [1, 2, 3]
    u['trade_date'] = pd.Timestamp('2020-01-02')
    tix = {c: i for i, c in enumerate(u.stock_code)}
    w = sim.cell_weights(len(tix), u, 'in_U', 'tercile_U', tix)
    check('cells: weights sum to 1, cell share = P1 count / N', abs(w.sum() - 1) < 1e-12 and abs(w[[4, 5, 6]].sum() - 1 / 3) < 1e-12)
    bad = u.copy(); bad.loc[bad.tercile_U == 0, 'in_U'] = False
    try:
        sim.cell_weights(len(tix), bad, 'in_U', 'tercile_U', tix); check('cells: P1 cell without U raises', False)
    except RuntimeError:
        check('cells: P1 cell without U raises', True)


def t_labels():
    import pool_vs_index_verdict as vd
    check('label: exactly +1 with 3 positive -> pool_edge', vd.label(1.0, [0.5, 0.2, 0.1, -1, -1])[0] == 'pool_edge')
    check('label: exactly -1 with 3 negative -> pool_drag', vd.label(-1.0, [-0.5, -0.2, -0.1, 1, 1])[0] == 'pool_drag')
    check('label: -1, 2 neg + 3 ties -> no_clear', vd.label(-1.0, [-0.5, -0.2, 1e-11, -1e-11, 0.0])[0] == 'no_clear_difference')
    rows = [{'k': k, 'div': d, 'delist': x, 'valid': True, 'label': 'pool_edge'} for k in vd.COMBO_KS for d in vd.COMBO_DIVS for x in vd.COMBO_DELISTS]
    combos = pd.DataFrame(rows)
    check('final: 12 same labels -> that label', vd.final_label(combos, [])[0] == 'pool_edge')
    flip = combos.copy(); flip.loc[5, 'label'] = 'no_clear_difference'
    check('final: one combo flips / missing / duplicate / failed check -> undetermined',
          all(vd.final_label(c, f)[0] == 'undetermined' for c, f in ((flip, []), (combos.iloc[:11], []),
              (pd.concat([combos.iloc[:11], combos.iloc[[0]]]), []), (combos, ['x']))))


def t_cap():
    try:
        pc.cap_date('2025-12-31'); check('cap: later than END_CAP rejected', False)
    except ValueError:
        check('cap: later than END_CAP rejected', True)
    import stage2_common as sc
    orig = sc.q; sc.q = lambda conn, sql, params=None: pd.DataFrame({'date': ['2025-12-30', '2026-01-02']})
    try:
        pc.capped_query(None, 'x', [], pc.END_CAP); ok = False
    except RuntimeError:
        ok = True
    finally:
        sc.q = orig
    check('cap: rows after cap rejected (0 rows allowed after 2025-12-30)', ok)


# ---------------- F1·F9·F10: 판정 연결 ----------------

def toy_sims(drop_date=None, offset_k5=5):
    """2013-12-30~2025-12-30 영업일 달력으로 P1·M·S20e·E0 곡선을 만든 두 k 모의 실행(판정기 입력 모양 그대로)."""
    cal = pd.bdate_range('2013-12-30', '2025-12-30')
    t = np.arange(len(cal)) / 252.0
    cols = {}
    for a, g in (('P1', 0.06), ('M', 0.05), ('S20e', 0.04)):
        for div in ('none', 'observed', 'late'):
            for dl in ('last', 'zero'):
                for cm in ('net', 'gross'):
                    cols[f'{a}|{div}|{dl}|{cm}'] = pc.CASH0 * np.exp(g * t) * (1 + 0.01 * np.sin(t * 7))
    curves = pd.DataFrame(cols, index=cal)
    if drop_date: curves = curves.drop(pd.Timestamp(drop_date))
    e = pd.DataFrame({f'{pc.E0}|net': pc.CASH0 * np.exp(0.045 * t), f'{pc.E0}|gross': pc.CASH0 * np.exp(0.046 * t), f'{pc.E0}|ok': True}, index=cal)
    sims = {}
    for k, off in (('k0', 0), ('k5', offset_k5)):
        sims[k] = {'curves': curves, 'opens': curves, 'etf': e, 'full_cal': cal, 'priced_cal': cal,
                   'checks': {'k': off, 'etf_entry': {pc.KODEX200: {'date': str(cal[1].date()), 'shares': 1.0}}}}
    return sims


def t_f1_q1_q3_rows():
    import pool_vs_index_verdict as vd
    mt = vd.metrics_table(toy_sims())
    q1, q3 = vd.q1_rows(mt), vd.q3_rows(mt)
    n1 = {k: int(q1[q1.k == k].valid.sum()) for k in vd.COMBO_KS}
    check('F1 Q1 filled through real metrics_table: 7 valid rows per k', n1 == {'k0': 7, 'k5': 7}, str(n1))
    check('F1 Q3 filled through real metrics_table: 4 valid rows', int(q3.valid.sum()) == 4)
    check('F1 formal output row-count check passes on complete input', vd.output_failures(q1, q3, 'formal') == [])
    check('F1 formal output row-count check fails on empty Q1/Q3', len(vd.output_failures(q1.assign(valid=False), q3.assign(valid=False), 'formal')) >= 3)


def t_f9_q3_inputs():
    import pool_vs_index_verdict as vd
    mt = vd.metrics_table(toy_sims())
    q3 = vd.q3_rows(mt)
    check('F9 Q3 arithmetic check residual ~0 and inputs consistent', (q3.arithmetic_check_residual.abs() < 1e-12).all() and q3.inputs_consistent.all())
    bad = mt.copy(); m = (bad.account == 'S20e') & (bad.period == 'main'); bad.loc[m, 'base'] = '2014-01-02'
    check('F9 Q3 inputs from different base date are flagged', 'q3 inputs not from the same k/combo/period' in vd.output_failures(vd.q1_rows(bad), vd.q3_rows(bad), 'smoke'))


def t_f10_key_and_calendar():
    import pool_vs_index_verdict as vd
    sims = toy_sims(offset_k5=0)
    base_checks = {'skipped_months': [], 'weights_all_ok': True, 'min_cash_after_orders': 0, 'max_abs_conservation_err': 0, 'base_value_all_cash0': True,
                   'n_months': 2, 'base_date': '2013-12-30', 'end_date': pc.END_CAP,
                   'bundle_checks': {'p1_not_in_U': 0, 'p1_not_in_U0': 0, 'snapshot_after_signal': 0, 'tier_row_after_signal': 0, 'signal_not_before_trade': 0,
                                     'p1_mcap_missing': 0, 'p1_mcap_missing_months': [], 'p1_missing_from_snapshot': {}, 'price_max_date': pc.END_CAP, 'months': 2}}
    s = {**sims['k5'], 'checks': {**base_checks, 'k': 0}}
    check('F10 verdict rejects sim whose offset does not match key (k5 with offset 0)', any('offset' in x for x in vd.sim_failures('k5', s, 'smoke')))
    sims2 = toy_sims(drop_date='2015-12-31')     # 달력의 구간 직전 거래일을 곡선에서 뺀다(Astra: 12-30 누락 → 12-29 사용)
    r = vd.period_row(sims2['k0']['curves']['P1|observed|last|net'], sims2['k0'], (2016, 2017))
    check('F10 segment base missing from curve -> invalid (no silent shift to the previous day)', not r['valid'])
    r2 = vd.period_row(toy_sims()['k0']['curves']['P1|observed|last|net'], toy_sims()['k0'], (2016, 2017))
    check('F10 segment base is calendar trading day before 2016-01-01', r2['valid'] and r2['base'] == '2015-12-31', str(r2.get('base')))


def t_f10_s20_checker():
    import pool_vs_index_legacy_s20 as lg
    idx = pd.bdate_range('2014-01-02', periods=5); ref = pd.Series(100.0, index=idx)
    ok1 = lg.compare(ref + 1.0, ref, 101.0)['pass']
    mid = ref.copy(); mid.iloc[2] += 2.0
    bad_mid = lg.compare(mid, ref, 100.0)['pass']
    endv = ref.copy(); endv.iloc[-1] += 2.0
    bad_end = lg.compare(endv, ref, 100.0)['pass']
    check('F10 S20 checker: exactly 1 won passes, 2 won middle or end fails', ok1 and not bad_mid and not bad_end)


# ---------------- F2: 미체결 몫 ----------------

def t_f2_blocked_buy_stays_cash():
    """Astra 반례: A 60 정지, 현금 40, B·C 각 50%. C 거래량 1 → B 20·C 20·현금 0, C 거래량 0 → B 20·현금 20."""
    out = {}
    for c_vol in (1.0, 0.0):
        ao = np.full((3, 3), 10.0); ro = ao.copy(); vol = np.full((3, 3), 100.0); ro[2, 0] = 0.0; vol[2, 0] = 0.0; vol[2, 2] = c_vol
        mk = market(('2020-01-01', 3), 3, ao, ao, ro, volume=vol)
        st = ac.State(q=np.array([6.0, 0.0, 0.0]), cash=40.0)
        rec = ac.rebalance(st, mk, 2, np.array([0.0, 0.5, 0.5]), NO_COST)
        out[c_vol] = (st.q[1] * 10, st.q[2] * 10, st.cash, rec)
    b, c, cash, rec = out[1.0]
    check('F2 counterexample: all tradable -> B 20, C 20, cash 0', abs(b - 20) < 1e-9 and abs(c - 20) < 1e-9 and abs(cash) < 1e-9)
    b, c, cash, rec = out[0.0]
    check('F2 counterexample: C volume 0 -> B 20, C 0, cash 20 (no reallocation)', abs(b - 20) < 1e-9 and c == 0 and abs(cash - 20) < 1e-9, str(out[0.0][:3]))
    check('F2 blocked unbought cash recorded', abs(rec['blocked_unbought_cash'] - 20) < 1e-9 and abs(rec['blocked_target_weight'] - 0.5) < 1e-12)
    ao = np.full((3, 2), 10.0); vol = np.full((3, 2), 100.0); vol[1, 1] = 0.0
    mk = market(('2020-01-01', 3), 2, ao, ao, volume=vol)
    v, lg = ac.run_fractional(mk, events({1: np.array([0.5, 0.5])}), NO_COST, False, 1000.0)
    check('F2 unfilled (funds ample): volume 0 with positive open not filled, share stays cash', abs(lg['orders'][0]['cash_after'] - 500) < 1e-9)


def t_f2_integer_blocked():
    ao = np.full((3, 3), 10.0); ro = ao.copy(); vol = np.full((3, 3), 100.0); vol[1, 2] = 0.0
    mk = market(('2020-01-01', 3), 3, ao, ao, ro, volume=vol)
    w = {1: np.array([0.0, 0.5, 0.5])}
    v, lg = ac.run_integer(mk, events(w, order={1: [1, 2]}, markets={1: np.array(['KOSPI'] * 3)}), NO_COST, False, 1000.0, FLAT_TICK)
    check('F2 integer: blocked target share stays cash', abs(lg['orders'][0]['cash_after'] - 500) < 1e-9, str(lg['orders'][0]))


# ---------------- F3: 패널 ----------------

class FakeDH:
    """get_stock_row_as_of의 조회 끝이 신호일 뒤면 신호일 뒤 결측 때문에 실패하는 실제 조회기를 흉내 낸다."""
    def __init__(self, source='TIER_1_SNAPSHOT_ASOF', codes=('000001',)): self.source, self.codes, self.ends = source, list(codes), []

    def get_candidates_with_tier_fallback_pit_gated(self, sd, **kw): return self.codes, self.source

    def get_stock_row_as_of(self, t, sd, start, end):
        self.ends.append(end)
        if pd.Timestamp(end) > pd.Timestamp(sd): raise ValueError('future NaN in loaded window')
        return pd.Series({'atr_14_ratio': 0.03, 'close_price': 1000.0, 'market_cap': 1e10, 'cheap_score': 0.5, 'cheap_score_confidence': 1.0, 'flow5_mcap': 0.01})


def t_f3_panel_pit_and_status():
    import pool_vs_index_panel as pn
    strat = types.SimpleNamespace(min_liquidity_20d_avg_value=0, min_tier12_coverage_ratio=0.2, strict_hysteresis_enabled=True)
    d, sd = pd.Timestamp('2023-12-01'), pd.Timestamp('2023-11-30')
    dh = FakeDH()
    df, cal = pn.month_panel(dh, strat, d, sd, ('2023-10-22', '2023-12-06'))
    check('F3 PIT: row lookup ends at signal date; future NaN cannot change status', cal['status'] == 'ok' and dh.ends == ['2023-11-30'], str(cal))
    _, cal2 = pn.month_panel(FakeDH(source='NO_CANDIDATES_NO_UNIVERSE', codes=()), strat, d, sd, ('2023-10-22', '2023-12-06'))
    check('F3 NO_CANDIDATES_NO_UNIVERSE -> missing_universe (not zero_candidates)', cal2['status'] == 'missing_universe')
    _, cal3 = pn.month_panel(FakeDH(source='NO_CANDIDATES_TIER', codes=()), strat, d, sd, ('2023-10-22', '2023-12-06'))
    check('F3 normal zero candidates -> zero_candidates', cal3['status'] == 'zero_candidates')
    check('F3 candidates but zero ranked -> zero_ranked', pn.month_status('TIER_1_SNAPSHOT_ASOF', 3, 0) == 'zero_ranked')
    out, months = pn.assemble([(None, cal3), (None, cal3)])
    check('F3 all-zero panel writes empty panel and calendar', len(out) == 0 and list(out.columns) == pn.PANEL_COLUMNS and len(months) == 2)


def t_f3_calendar_mismatch():
    import pool_vs_index_data as dat
    dates = pd.bdate_range('2023-11-01', '2024-01-31')
    days = pc.rebalance_days(dates, 0, '2023-12-01')
    pm = pd.DataFrame([{'trade_date': t, 'signal_date': s, 'status': 'ok', 'tier_source': 'TIER_1', 'raw_candidates': 1, 'ranked': 1} for t, s in days])
    panel = pd.DataFrame({'trade_date': [days[1][0]], 'signal_date': [days[1][1]], 'ticker': ['000001'], 'rank': [1]})
    m = dat.month_calendar(dates, 0, '2023-12-01', dates[-1], pm, panel)
    check('F3 ok with ranked=1 but no panel row -> panel_mismatch', m.status.tolist() == ['panel_mismatch', 'ok'], str(m.status.tolist()))


def t_f3_cell_error_recorded():
    import pool_vs_index_sim as sim
    dates = pd.bdate_range('2023-11-30', periods=5)
    u = pd.DataFrame({'stock_code': ['A', 'B'], 'market': ['KOSPI', 'KOSPI'], 'market_cap': [1.0, 2.0], 'liq': [2e9, 2e9],
                      'in_U': [False, True], 'in_U0': [True, True], 'tercile_U': [-1, 0], 'tercile_U0': [0, 1], 'p1_rank': [1, 0],
                      'trade_date': dates[1], 'signal_date': dates[0]})
    b = {'months': pd.DataFrame({'trade_date': [dates[1]], 'signal_date': [dates[0]], 'status': ['ok']}), 'u': u}
    reb, order, markets, wchk, cells, skipped = sim.schedules(b, dates, {'A': 0, 'B': 1})
    check('F3 P1 outside U: month skipped with cell_error reason (verdict undetermined)', len(skipped) == 1 and skipped[0]['reason'].startswith('cell_error'))


# ---------------- F4: 상한·절단 ----------------

def t_f4_window():
    try:
        pc.check_window('2023-12-01', '2024-01-31', '2023-12-28'); ok = False
    except ValueError:
        ok = True
    check('F4 cap 2023-12-28 with end 2024-01-31 rejected before SQL', ok)
    import pool_vs_index_data as dat
    seen = []
    import stage2_common as sc
    orig = sc.q; sc.q = lambda conn, sql, params=None: (seen.append(params), pd.DataFrame(columns=['stock_code', 'date']))[1]
    try:
        try:
            dat.load_prices(None, ['A'], pd.Timestamp('2023-11-01'), pd.Timestamp('2024-01-31'), pd.Timestamp('2023-12-28')); bad = False
        except ValueError:
            bad = True
    finally:
        sc.q = orig
    check('F4 load_prices: end beyond cap raises and no SQL is sent', bad and not seen)


def t_f4_etf_cap_lines():
    import subprocess
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, 'in'), os.path.join(d, 'out'); os.makedirs(src)
        rows = 'date,open,high,low,close,volume\n2023-12-27,1,1,1,1,1\n2023-12-28,2,2,2,2,2\n2023-12-29,NOT_A_NUMBER,x,x,x,x\n'
        for n in ['069500.csv'] + [f'{c}.csv' for c in pc.STYLE_ETFS]: open(os.path.join(src, n), 'w').write(rows)
        open(os.path.join(src, '069500_quality.csv'), 'w').write('date,ohlc_ok\n2023-12-27,True\n2023-12-28,True\n2023-12-29,True\n')
        tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pool_vs_index_etf_cap.py')
        r = subprocess.run([sys.executable, '-I', '-B', tool, src, out, '2023-12-28'], capture_output=True, text=True)
        txt = open(os.path.join(out, '069500.csv')).read() if r.returncode == 0 else ''
        check('F4 ETF cap: only rows <= cap copied, later price fields never parsed', r.returncode == 0 and 'NOT_A_NUMBER' not in txt and txt.count('\n') == 3, r.stderr[-200:])


CUT_DATES = pd.DatetimeIndex(['2023-11-30', '2023-12-01', '2023-12-04', '2023-12-28'])
FULL_DATES = CUT_DATES.append(pd.DatetimeIndex(['2024-01-02']))
ROLE = {'full': ('2025-12-30', '2024-01-02', FULL_DATES), 'cut': ('2023-12-28', '2023-12-28', CUT_DATES)}


def sha_file(p):
    import hashlib
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def write_sim(d, role, panel, k=0, drop_last=False, drop_cols=(), int_files=True, fill_ticker='A', orders=None, fills=True,
              int_fills=True, claim=None, calendar=True):
    """절단 검사용 최소 모의 실행 폴더(고정 열 목록, 역할별 상한·종료일·패널 해시·ETF 입력 폴더). claim=다른 역할의 메타데이터를 주장."""
    os.makedirs(d)
    cap, end, dates = ROLE[role]; mcap, mend, _ = ROLE[claim or role]
    idx = dates[:-1] if drop_last else dates
    for n in pc.FRAME_FILES:
        if n == 'integer_curve.parquet' and not int_files: continue
        cols = [c for c in pc.frame_columns(n) if c not in drop_cols]
        pd.DataFrame(1.0, index=idx, columns=cols).to_parquet(os.path.join(d, n))
    key = {'account': 'P1', 'div': 'observed', 'delist': 'last', 'cost': 'net'}; day = str(idx[1].date())
    rows = {'orders.csv': orders if orders is not None else [{**key, 'date': day, 'buy_gross': 10.0, 'sell_gross': 0.0}],
            'fills.parquet': [{**key, 'date': day, 'ticker': fill_ticker, 'sell_qty': 0.0, 'buy_qty': 1.0, 'sell_value': 0.0, 'buy_value': 10.0}] if fills else [],
            'integer_orders.csv': [{'date': day, 'buy_gross': 10.0, 'sell_gross': 0.0}],
            'integer_fills.csv': [{'date': day, 'ticker': fill_ticker, 'sell_qty': 0.0, 'buy_qty': 1.0, 'sell_value': 0.0, 'buy_value': 10.0}] if int_fills else []}
    for n in pc.TABLE_SCHEMAS:
        if n.startswith('integer_') and not int_files: continue
        df = pc.table_frame(rows.get(n, []), n)
        df.to_parquet(os.path.join(d, n), index=False) if n.endswith('.parquet') else df.to_csv(os.path.join(d, n), index=False)
    if calendar: pd.DataFrame({'date': FULL_DATES.strftime('%Y-%m-%d'), 'priced': True}).to_csv(os.path.join(d, 'calendar.csv'), index=False)
    etf = os.path.join(d, 'etf'); os.makedirs(etf); json.dump({'cap': mcap}, open(os.path.join(etf, 'etf_cap_manifest.json'), 'w'))
    json.dump({'base_date': str(dates[0].date()), 'k': k, 'end_date': mend if not drop_last else mend, 'cap': mcap, 'etf_dir': etf,
               'bundle_checks': {'cap': mcap, 'end': mend, 'panel_sha256': sha_file(panel)}}, open(os.path.join(d, 'sim_checks.json'), 'w'))


def write_panel(d, name, role='cut', drop=None):
    import pool_vs_index_panel as pn
    p = os.path.join(d, name)
    rows = [{c: 1 for c in pn.PANEL_COLUMNS} | {'trade_date': '2023-12-01', 'signal_date': '2023-11-30', 'ticker': '000001'}]
    months = [{'trade_date': '2023-12-01', 'signal_date': '2023-11-30', 'status': 'ok', 'tier_source': 'T', 'raw_candidates': 1, 'ranked': 1}]
    if role == 'full':
        rows.append({c: 1 for c in pn.PANEL_COLUMNS} | {'trade_date': '2024-01-02', 'signal_date': '2023-12-28', 'ticker': '000002'})
        months.append({'trade_date': '2024-01-02', 'signal_date': '2023-12-28', 'status': 'ok', 'tier_source': 'T', 'raw_candidates': 1, 'ranked': 1})
    df = pd.DataFrame(rows)
    if drop: df = df.drop(columns=[drop])
    df.to_csv(p, index=False)
    pd.DataFrame(months).to_csv(p.replace('.csv', '') + '.months.csv', index=False)
    return p


def trunc(full, cut, pf, pcut, k=0, cutday='2023-12-28'):
    import pool_vs_index_truncation_check as tc
    a = types.SimpleNamespace(full=full, cut_dir=cut, cut=cutday, k=k, start='2023-12-01', panels=[pf, pcut], full_end='2024-01-31', full_cap='2025-12-30')
    return tc.run(a)


def run_case(full_kw=None, cut_kw=None, cut_panel_drop=None, full_role='full', full_panel_role='full', mutate=None):
    with tempfile.TemporaryDirectory() as d:
        pf = write_panel(d, 'pf.csv', full_panel_role); pcut = write_panel(d, 'pc.csv', 'cut', drop=cut_panel_drop)
        full, cut = os.path.join(d, 'full'), os.path.join(d, 'cut')
        write_sim(full, full_role, pf if full_panel_role == 'full' else pcut, **(full_kw or {})); write_sim(cut, 'cut', pcut, **(cut_kw or {}))
        if mutate: mutate(full, cut)
        return trunc(full, cut, pf if full_panel_role == 'full' else pcut, pcut)


def failed_names(r):
    return [p['name'] for p in r.get('parts', []) if not p['ok']] + [k for k, v in r.get('provenance', {}).items() if not v] + [r.get('reason') or '']


def t_f4_truncation_checker():
    both = lambda kw: (kw, kw)
    cases = {'baseline identical -> pass': ((None, None), {}, True),
             'G1 cut day removed from all outputs (sim_checks still says 12-28) -> fail': ((None, {'drop_last': True}), {}, False),
             'F4 missing ETF columns (277650) -> fail': ((None, {'drop_cols': [f'E_277650|{x}' for x in pc.ETF_SUFFIXES]}), {}, False),
             'F4 missing one curve column -> fail': ((None, {'drop_cols': ['P1|observed|last|net']}), {}, False),
             'G1 integer files absent on both sides -> fail': (both({'int_files': False}), {}, False),
             'F4 same-price ticker swap caught by per-ticker fills': ((None, {'fill_ticker': 'B'}), {}, False),
             'F4 missing last order -> fail': ((None, {'orders': [], 'fills': False, 'int_fills': True}), {}, False),
             'F10 wrong k passed by runner -> fail': (both({'k': 5}), {}, False),
             'G1 panel column (score_q) removed -> fail': ((None, None), {'cut_panel_drop': 'score_q'}, False),
             'H1 same cut copy on both sides (cut bundle and panel as full) -> fail': ((None, None), {'full_role': 'cut', 'full_panel_role': 'cut'}, False),
             'H1 cut run claims full cap/end/ETF cap -> fail': ((None, {'claim': 'full'}), {}, False),
             'H1 cut calendar.csv absent -> fail': ((None, {'calendar': False}), {}, False),
             'H2 fills of a buying account removed on both sides -> fail': (both({'fills': False}), {}, False),
             'H2 integer fills header-only on both sides -> fail': (both({'int_fills': False}), {}, False)}
    for name, ((fk, ck), extra, expect) in cases.items():
        r = run_case(fk, ck, **extra)
        check(f'truncation: {name}', r['pass'] == expect, str(failed_names(r))[:300])
    r = run_case(mutate=lambda full, cut: open(os.path.join(cut, 'pays.csv'), 'w').write('account,div\n'))
    check('truncation: G1 zero-row table with wrong header -> fail', not r['pass'])


def t_f4_fills_ticker_swap():
    ao = np.full((3, 2), 10.0); mk = market(('2020-01-01', 3), 2, ao, ao)
    _, la = ac.run_fractional(mk, events({1: np.array([1.0, 0.0])}, fills=True), NO_COST, False, 1000.0)
    _, lb = ac.run_fractional(mk, events({1: np.array([0.0, 1.0])}, fills=True), NO_COST, False, 1000.0)
    check('F4 fills record which ticker was bought (A vs B differ, totals same)',
          la['orders'][0]['buy_gross'] == lb['orders'][0]['buy_gross'] and list(la['fills'][0][1]) != list(lb['fills'][0][1]))


# ---------------- F5: ETF ----------------

def t_f5_etf_entry_and_calc():
    dates = pd.bdate_range('2020-03-30', periods=8)     # 첫 교체일 0번째(무효), 1번째 유효(교체일 아님)
    e = pd.DataFrame({'open': [10, 11, 12, 13, 14, 15, 16, 17.0], 'close': [10, 11, 99, 13, 14, 15, 16, 120.0],
                      'ok': [False, True, False, True, True, True, True, False]}, index=dates)
    r = ac.run_etf(dates, e, 0, 0.0, 1100.0)
    check('F5 ETF enters on first valid day (daily check), not next rebalance', r['entry'] == 1 and abs(r['calc'].iloc[1] - 1100) < 1e-9)
    check('F5 ETF calc price NaN on invalid day; display uses previous valid', np.isnan(r['calc'].iloc[2]) and abs(r['display'].iloc[2] - 1100) < 1e-9)
    import pool_vs_index_verdict as vd
    sim = {'priced_cal': dates, 'full_cal': dates}
    row = vd.period_row(r['calc'], sim, (2020, 2020))
    check('F5 invalid ETF endpoint is not annualized (period invalid)', not row['valid'])
    import pool_vs_index_sim as sm
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'open': 1.0, 'close': 1.0}).to_csv(os.path.join(d, '069500.csv'), index=False)
        try:
            sm.etf_frame(d, '069500', dates, dates[-1]); ok = False
        except RuntimeError:
            ok = True
    check('F5 069500 quality file missing -> error (no all-true default)', ok)


def t_f5_open_values():
    ao = np.array([[10.0], [10.0], [12.0], [12.0]]); acl = np.array([[10.0], [10.0], [11.0], [12.0]])
    mk = market(('2020-01-01', 4), 1, ao, acl)
    _, lg = ac.run_fractional(mk, events({1: np.array([1.0])}), NO_COST, False, 1000.0)
    check('F5 open value available on non-rebalance day', abs(lg['open_values'].iloc[2] - 1200.0) < 1e-9 and abs(lg['open_values'].iloc[0] - 1000.0) < 1e-9)


# ---------------- F6: 퇴출 순서, 권리, 장부 ----------------

def t_f6_holiday_snapshot_exit():
    import pool_vs_index_sim as sim
    dates = pd.DatetimeIndex(['2020-02-27', '2020-02-28', '2020-03-02', '2020-03-03'])   # 2/29(토) 스냅숏
    ex = pd.DataFrame({'ticker': ['A', 'B'], 'snapshot_date': pd.to_datetime(['2020-02-29', '2020-03-02']), 'event': ['exit', 'exit']})
    post, pre = sim.exit_positions(ex, dates, {'A': 0, 'B': 1})
    check('F6 holiday snapshot -> next trading day before orders; trading-day snapshot -> after orders',
          list(pre) == [2] and list(pre[2]) == [0] and list(post) == [2] and list(post[2]) == [1])
    ao = np.full((4, 2), 100.0); mk = market(None, 2, ao, ao, dates=dates)
    w = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    v, lg = ac.run_fractional(mk, events(w, pre={2: np.array([0])}), NO_COST, False, 1000.0)
    o = lg['orders'][1]
    check('F6 counterexample: Saturday exit is not sold on Monday open; exit recorded', o['sell_gross'] == 0 and len(lg['exits']) == 1 and lg['exits'][0]['when'] == 'pre', str(o))


def ratio_market(rt, implied=None, n=4):
    mk = market(('2020-12-24', n), 1, np.full((n, 1), 10.0), np.full((n, 1), 10.0), ratio=rt)
    raw = np.array(rt, dtype=float)
    last = np.maximum.accumulate(np.where(np.isfinite(raw), np.arange(n)[:, None], -1), axis=0)
    mk.ratio_last_idx = last; mk.implied_ratio = np.array(implied, dtype=float) if implied is not None else np.full((n, 1), np.nan)
    return mk


def rights_case(mk, rec_i, pay_i):
    ev = events({1: np.array([1.0])}, record={rec_i: 2020}, pay={pay_i: 2020}, dps={2020: dps([4.0], [True])})
    v, lg = ac.run_fractional(mk, ev, NO_COST, False, 100.0)
    return lg['rights'][0], lg['pays'][0]


def t_f6_ratio_and_book():
    r, p = rights_case(ratio_market([[0.5], [0.5], [np.nan], [0.5]]), 2, 3)
    check('G3 record-date ratio missing, previous valid within 20 days, no change -> previous ratio, paid (v2.2-note 1)',
          r['n_ratio_fallback'] == 1 and r['ratio_cases'][0]['basis'] == 'previous_valid_ratio' and abs(p['amount'] - 10 * 0.5 * 4) < 1e-9, str(r['ratio_cases']))
    r, p = rights_case(ratio_market([[0.5], [0.5], [np.nan], [0.5]], implied=[[0.5], [0.5], [1.0], [0.5]]), 2, 3)
    check('G3 Astra counterexample: ratio changed in gap -> rights undetermined, not paid',
          r['n_rights_undetermined'] == 1 and r['ratio_cases'][0]['basis'] == 'undetermined_ratio_changed' and p['amount'] == 0, str(r['ratio_cases']))
    n = 25; rt = [[0.5]] + [[np.nan]] * (n - 1)
    r, p = rights_case(ratio_market(rt, n=n), 23, 24)
    check('G3 previous valid ratio older than 20 trading days -> undetermined, not paid', r['ratio_cases'][0]['basis'] == 'undetermined_stale' and p['amount'] == 0)
    check('G4 record-day log has no DPS-derived fields', not any(k.startswith('dps') or 'flag' in k for k in r))
    acl = np.array([[10.0], [10.0], [0.0], [10.0]])
    mk2 = market(('2020-01-01', 4), 1, np.full((4, 1), 10.0), acl)
    check('F6 zero close does not update book; marked as missing', mk2.book[2, 0] == 10.0 and bool(mk2.close_missing[2, 0]))
    import pool_vs_index_sim as sim
    px = pd.DataFrame({'stock_code': 'A', 'date': pd.bdate_range('2020-01-01', periods=3), 'adj_close': [10.0, 0.0, -1.0], 'close_price': [10.0, 0.0, 5.0],
                       'adj_open': 10.0, 'open_price': 10.0, 'adj_ratio': [1.0, np.nan, 1.0], 'volume': 1.0})
    m = sim.build_market(px, pd.DatetimeIndex(px.date), ['A'], pd.DatetimeIndex(px.date))
    check('F6 sim build_market: only finite positive closes; ratio_raw keeps NaN; last valid ratio index',
          list(m.book[:, 0]) == [10, 10, 10] and np.isnan(m.ratio_raw[1, 0]) and list(m.ratio_last_idx[:, 0]) == [0, 0, 2])


def t_f6_dps_observed_vs_absent():
    n = 6; ao = np.full((n, 3), 10.0); mk = market(('2020-12-28', n), 3, ao, ao)
    ev = events({1: np.array([1 / 3, 1 / 3, 1 / 3])}, record={2: 2020}, pay={4: 2020}, dps={2020: dps([5.0, 0.0, 0.0], [True, True, False])})
    _, lg = ac.run_fractional(mk, ev, NO_COST, False, 900.0)
    p = lg['pays'][0]
    check('F6 observed-zero and unobserved DPS counted separately', p['holders_observed_zero'] == 1 and p['holders_unobserved'] == 1 and p['holders_observed_positive'] == 1, str(p))
    ev2 = events({1: np.array([1.0, 0, 0])}, record={2: 2020}, pay={4: 2020}, dps={2020: dps([4.0, 0.0, 0.0], [True, True, True])})
    v, lg2 = ac.run_fractional(mk, ev2, NO_COST, False, 1000.0)
    check('DPS > 30% is still paid (not excluded)', abs(lg2['pays'][0]['amount'] - 400) < 1e-9)


# ---------------- F7·F8 ----------------

def t_f7_ticks():
    check('F7 KOSDAQ 2014 tick keeps 213,600 (100 won)', pc.tick_up(213600, 'KOSDAQ', '2014-12-01') == 213600)
    check('F7 KOSPI 2014 tick 500 above 100,000', pc.tick_up(213600, 'KOSPI', '2014-12-01') == 214000)
    check('F7 unified table from 2023-01-25', pc.tick_up(213600, 'KOSDAQ', '2023-01-25') == 214000 and pc.tick_up(1999, 'KOSPI', '2023-02-01') == 1999)
    ao = np.full((3, 1), 10.0); ao[1, 0] = np.nan; ro = np.full((3, 1), 10.0)
    mk = market(('2020-01-01', 3), 1, ao, np.full((3, 1), 10.0), ro)
    v, lg = ac.run_integer(mk, events({1: np.array([1.0])}, order={1: [0]}, markets={1: np.array(['KOSPI'])}), NO_COST, False, 1000.0, pc.tick_up)
    o = lg['orders'][0]
    check('F7 integer: missing adjusted open does not block raw order', o['buy_gross'] == 1000 and o['n_adj_open_missing_but_raw_tradable'] == 1, str(o))


def t_f8_turnover():
    ao = np.full((3, 2), 10.0); mk = market(('2020-01-01', 3), 2, ao, ao)
    _, lg = ac.run_fractional(mk, events({1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}), NO_COST, False, 1000.0)
    o = lg['orders'][1]
    check('F8 monthly turnover: one-way = min(buy, sell)/total, two-way = sum/total', abs(o['turnover_one_way'] - 1.0) < 1e-12 and abs(o['turnover_two_way'] - 2.0) < 1e-12)


# ---------------- 3차(G1~G6, R1~R4): Astra 2차 반례 ----------------

def t_g2_fills_everywhere():
    import pool_vs_index_sim as sim
    ao = np.full((3, 2), 10.0); mk = market(('2020-01-01', 3), 2, ao, ao)
    reb = {a: {1: np.array([0.5, 0.5])} for a in pc.ACCOUNTS}
    ev_none = (dict(), dict(), {})
    curves, logs = sim.run_accounts(mk, reb, {'post': {}, 'pre': {}, 'marks': ()}, {m: ev_none for m in pc.DIV_MODES})
    check('G2 fills recorded for all 156 account-combos', len(logs) == 156 and all(lg['fills'] for lg in logs.values()))
    out = {}
    for t in (0, 1):
        w = np.zeros(2); w[t] = 1.0
        v, lg = ac.run_integer(mk, events({1: w}, order={1: [t]}, markets={1: np.array(['KOSPI'] * 2)}), NO_COST, False, 1000.0, FLAT_TICK)
        out[t] = (v, sim.fills_frame(lg['fills'], mk.dates, mk.tickers, keys=False))
    check('G2 integer same-price ticker swap: curves equal but integer fills differ',
          out[0][0].equals(out[1][0]) and out[0][1].ticker.tolist() != out[1][1].ticker.tolist(), str([out[t][1].ticker.tolist() for t in (0, 1)]))


def t_g4_dps_quality():
    import pool_vs_index_sim as sim
    dates = pd.bdate_range('2019-12-20', '2020-06-30'); n = len(dates)
    mk = market(None, 2, np.full((n, 2), 10.0), np.full((n, 2), 10.0), dates=dates)
    mk.raw_close_missing = np.zeros((n, 2), dtype=bool)
    full_cal = dates
    dps = pd.DataFrame({'ticker': ['T0', 'T1', 'T0'], 'fy': [2019, 2019, 2020], 'dps': [1.0, 5.0, 9.0]})
    tix = {'T0': 0, 'T1': 1}
    rd = pc.record_date(full_cal, 2019); ri = int(dates.searchsorted(rd))
    qty = np.array([7.0, 0.0])
    logs = {f'{a}|observed|last|net': {'rights': [{'fy': 2019, 'qty_array': qty}]} for a in ('P1', 'M')}
    q_full = sim.dps_quality(mk, dates, full_cal, dps, tix, logs, pd.Timestamp('2020-06-30'))
    q_cut = sim.dps_quality(mk, dates, full_cal, dps, tix, logs, pd.Timestamp('2020-05-14'))
    hi = q_full[q_full.ticker == 'T1']
    check('G4 DPS quality only after observation window ends (cut before 5/15 has no FY2019 rows)', len(q_cut) == 0 and len(q_full) == 2)
    check('G4 high-DPS non-holder stays in quality table with 0 rights', len(hi) == 1 and bool(hi.flag_gt30.iloc[0]) and hi.P1_rights_qty.iloc[0] == 0)
    check('G4 quality table links P1/M rights qty', float(q_full[q_full.ticker == 'T0'].P1_rights_qty.iloc[0]) == 7.0)


def t_g5_absence_rebuy_and_ca():
    import pool_vs_index_sim as sim
    dates = pd.DatetimeIndex(['2020-02-27', '2020-02-28', '2020-03-02', '2020-03-03', '2020-03-04'])
    ex = pd.DataFrame({'ticker': ['A', 'A'], 'snapshot_date': pd.to_datetime(['2020-02-29', '2020-03-03']), 'event': ['exit', 'reentry']})
    ab = sim.absence_matrix(ex, dates, {'A': 0, 'B': 1})
    check('G5 absence from Saturday exit until day after trading-day reentry', ab[:, 0].tolist() == [False, False, True, True, False])
    ao = np.full((5, 2), 10.0); mk = market(None, 2, ao, ao, dates=dates); mk.absent = ab
    w = {1: np.array([0.4, 0.0]), 2: np.array([1.0, 0.0])}         # 금요일 A 40%, 월요일(부재 뒤) 같은 신호로 A 100%
    v, lg = ac.run_fractional(mk, events(w, pre={2: np.array([0])}), NO_COST, False, 100.0)
    o = lg['orders'][1]
    check('G5 Astra counterexample: after holiday absence, stale target does not rebuy A (cash kept)',
          o['buy_gross'] == 0 and abs(o['cash_after'] - 60) < 1e-9 and len(lg['exits']) == 1, str(o))
    ro = np.full((5, 1), 10.0); rt = [[0.5], [0.5], [1.0], [1.0], [1.0]]
    mk2 = market(None, 1, ro, ro, ro, ro, ratio=rt, dates=dates)
    st_q = {}
    v2, lg2 = ac.run_integer(mk2, events({1: np.array([1.0])}, order={1: [0]}, markets={1: np.array(['KOSPI'])}, pre={2: np.array([0])}),
                             NO_COST, False, 100.0, FLAT_TICK)
    e = lg2['exits'][0]
    check('G5 integer: holiday exit settled before corporate action (10 shares, 100 won, no CA on old generation)',
          e['qty'] == 10 and abs(e['book_written'] - 100) < 1e-9 and not lg2['ca'], str(lg2))


class NoSQL:
    def cursor(self): raise AssertionError('SQL sent for empty snapshot')


def t_g6_universe_and_calendar():
    import pool_vs_index_data as dat
    snaps = pd.DataFrame({'date': pd.to_datetime(['2023-11-24']), 'stock_code': ['X'], 'market_type': ['KONEX']})
    row = types.SimpleNamespace(signal_date=pd.Timestamp('2023-11-30'), trade_date=pd.Timestamp('2023-12-01'))
    try:
        u, miss, empty = dat.month_universe(NoSQL(), snaps, row, None); ok = empty and len(u) == 0
    except AssertionError:
        ok = False
    check('G6 empty two-market snapshot: no SQL sent, flagged as missing universe', ok)
    months = pd.DataFrame({'trade_date': [row.trade_date], 'signal_date': [row.signal_date], 'status': ['ok'], 'status_reason': ['']})
    dat.build_universe(NoSQL(), snaps, months, pd.DataFrame(columns=['trade_date', 'ticker', 'rank']))
    check('G6 build_universe marks month missing_universe with reason', months.status.iloc[0] == 'missing_universe' and months.status_reason.iloc[0] != '')
    dates = pd.bdate_range('2023-11-01', '2023-12-29'); days = pc.rebalance_days(dates, 0, '2023-12-01')
    t, s = days[0]
    pm = lambda ranked: pd.DataFrame([{'trade_date': t, 'signal_date': s, 'status': 'ok', 'tier_source': 'T', 'raw_candidates': 2, 'ranked': ranked}])
    good = pd.DataFrame({'trade_date': [t, t], 'signal_date': [s, s], 'ticker': ['A', 'B'], 'rank': [1, 2]})
    cal = lambda p, r: dat.month_calendar(dates, 0, '2023-12-01', dates[-1], pm(r), p).status.iloc[0]
    check('G6 calendar ok for consistent month', cal(good, 2) == 'ok')
    check('G6 ok with ranked=0 and no rows -> panel_mismatch', cal(good.iloc[0:0], 0) == 'panel_mismatch')
    check('G6 panel row with different signal date -> panel_mismatch', cal(good.assign(signal_date=[s, s - pd.Timedelta(days=1)]), 2) == 'panel_mismatch')
    check('G6 duplicated rank -> panel_mismatch', cal(good.assign(rank=[1, 1]), 2) == 'panel_mismatch')
    check('G6 duplicated ticker -> panel_mismatch', cal(good.assign(ticker=['A', 'A']), 2) == 'panel_mismatch')


def t_r1_etf_inputs():
    import pool_vs_index_sim as sm
    dates = pd.bdate_range('2020-01-01', periods=4)
    e = pd.DataFrame({'open': [np.nan] * 4, 'close': [np.nan] * 4, 'ok': [False] * 4}, index=dates)
    r = ac.run_etf(dates, e, 0, 0.0, 100.0)
    check('R1 ETF never entered: cash, valid, entered=False (same as longer run before entry)',
          r['entry'] is None and (r['calc'] == 100).all() and r['ok'].all() and not r['entered'].any())
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'open': 1.0, 'close': 1.0, 'ohlc_ok': True}).to_csv(os.path.join(d, '069500.csv'), index=False)
        try:
            sm.etf_frame(d, '069500', dates, dates[-1]); ok = False
        except RuntimeError:
            ok = True
        check('R1 069500 with inline ohlc_ok but no quality file -> error', ok)
        pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'ohlc_ok': True}).to_csv(os.path.join(d, '069500_quality.csv'), index=False)
        try:
            sm.etf_frame(d, '069500', dates, dates[-2]); ok = False
        except RuntimeError:
            ok = True
        check('R1 ETF input with rows after cap rejected by date-first reader', ok)


def t_r2_e0_same_day():
    import pool_vs_index_verdict as vd
    day = pd.Timestamp('2020-03-02')
    s = {'checks': {'etf_entry': {pc.KODEX200: {'date': '2020-03-02', 'shares': 9.9985}}},
         'etf': pd.DataFrame({f'{pc.E0}|open': [100.0]}, index=[day])}
    check('R2 E0 first-buy day uses pre-order CASH0 as base', vd.e0_value_at_open(s, day) == pc.CASH0)


def t_r3_checks_and_internal_nan():
    import pool_vs_index_verdict as vd
    r = vd.load_check('/nonexistent/check.json')
    check('R3 missing check JSON -> recorded failure (no crash)', r['pass'] is False and 'unreadable' in r['reason'])
    check('R3 failure listed', vd.external_failures({'panel': r, 'truncation_k0': {'pass': True}, 'truncation_k5': {'pass': True}, 'stage2_s20': {'pass': True}},
                                                    'smoke') == ['panel check failed'])
    sims = toy_sims(); c = sims['k0']['curves']['P1|observed|last|net'].copy()
    c.loc['2016-02-11'] = np.nan
    check('R3 internal stock valuation NaN -> period invalid', not vd.period_row(c, sims['k0'], (2016, 2017))['valid'])


# ---------------- 4차(H1~H4, 권장): Astra 3차 반례 ----------------

def t_h3_zero_candidates_and_rank():
    import pool_vs_index_data as dat
    dates = pd.bdate_range('2023-11-01', '2023-12-29'); t, s = pc.rebalance_days(dates, 5, '2023-12-01')[0]
    pm = lambda status, raw, ranked: pd.DataFrame([{'trade_date': t, 'signal_date': s, 'status': status, 'tier_source': 'T', 'raw_candidates': raw, 'ranked': ranked}])
    empty = pd.DataFrame(columns=['trade_date', 'signal_date', 'ticker', 'rank'])
    cal = lambda m, p: dat.month_calendar(dates, 5, '2023-12-01', dates[-1], m, p).iloc[0]
    r = cal(pm('zero_candidates', 63, 63), empty)
    check('H3 Astra counterexample: zero_candidates with raw=63 ranked=63 and no rows -> panel_mismatch with reason',
          r.status == 'panel_mismatch' and 'raw=63' in r.status_reason, r.status_reason)
    check('H3 genuine zero_candidates (raw=0, ranked=0, rows=0) kept', cal(pm('zero_candidates', 0, 0), empty).status == 'zero_candidates')
    p = pd.DataFrame({'trade_date': [t, t], 'signal_date': [s, s], 'ticker': ['A', 'B'], 'rank': [1.9, 2.9]})
    r = cal(pm('ok', 2, 2), p)
    check('H3 non-integer ranks (1.9, 2.9) -> panel_mismatch before int conversion', r.status == 'panel_mismatch' and 'integer' in r.status_reason, r.status_reason)


def t_h4_strict_hash_and_pass_string():
    import subprocess
    run_all = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pool_vs_index_run_all.sh')).read()
    check('H4 run_all approval checks use sha256sum --strict (code and design)',
          run_all.count('sha256sum -c --strict --quiet "$AP/code.sha256"') == 1 and run_all.count('sha256sum -c --strict --quiet "$AP/design.sha256"') == 1)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'a.txt'), 'w').write('x'); open(os.path.join(d, 'b.txt'), 'w').write('y')
        good = subprocess.run(['sha256sum', 'a.txt', 'b.txt'], cwd=d, capture_output=True, text=True).stdout.splitlines()
        open(os.path.join(d, 'm.sha256'), 'w').write(good[0] + '\n' + 'NOT_A_HASH  b.txt\n')
        loose = subprocess.run(['sha256sum', '-c', '--quiet', 'm.sha256'], cwd=d, capture_output=True).returncode
        strict = subprocess.run(['sha256sum', '-c', '--strict', '--quiet', 'm.sha256'], cwd=d, capture_output=True).returncode
    check('H4 Astra counterexample: malformed hash line passes without --strict, fails with --strict', loose == 0 and strict != 0, f'loose={loose} strict={strict}')
    import pool_vs_index_verdict as vd
    ok = {'truncation_k0': {'pass': True}, 'truncation_k5': {'pass': True}, 'stage2_s20': {'pass': True}}
    check('H4 Astra counterexample: {"pass": "false"} string is a failure', vd.external_failures({**ok, 'panel': {'pass': 'false'}}, 'smoke') == ['panel check failed'])
    check('H4 non-dict check JSON is a failure', vd.external_failures({**ok, 'panel': [True]}, 'smoke') == ['panel check failed'])
    check('H4 pass true (bool) accepted', vd.external_failures({**ok, 'panel': {'pass': True}}, 'smoke') == [])


def t_quality_window_and_tickers():
    import pool_vs_index_sim as sim
    dates = pd.bdate_range('2019-12-20', '2020-06-30'); n = len(dates)
    mk = market(None, 3, np.full((n, 3), 10.0), np.full((n, 3), 10.0), dates=dates); mk.raw_close_missing = np.zeros((n, 3), dtype=bool)
    dps = pd.DataFrame({'ticker': ['T0', 'T1', 'T2'], 'fy': [2019, 2019, 2019], 'dps': [1.0, 5.0, 1.0]})
    tix = {'T0': 0, 'T1': 1, 'T2': 2}
    logs = {f'{a}|observed|last|net': {'rights': [{'fy': 2019, 'qty_array': np.array([7.0, 0, 0])}]} for a in ('P1', 'M')}
    short = dates[dates <= '2020-01-31']
    q = sim.dps_quality(mk, short, short, dps, tix, logs, pd.Timestamp('2025-12-30'))
    check('Recommended: DPS quality window limited to min(cap, last run day) (Jan-end run has no 5/15 rows)', len(q) == 0, str(len(q)))
    u = pd.DataFrame({'trade_date': ['2020-01-02', '2020-01-02', '2020-07-01'], 'stock_code': ['T0', 'T1', 'T2'], 'in_U0': [True, True, True], 'p1_rank': [1, 0, 0]})
    q2 = sim.dps_quality(mk, dates, dates, dps, tix, logs, pd.Timestamp('2025-12-30'), u)
    check('Recommended: quality tickers fixed to U0/P1 known by window end (later entrant T2 excluded)', sorted(q2.ticker) == ['T0', 'T1'], str(sorted(q2.ticker)))


def t_duplicate_codes_and_empty_query():
    import pool_vs_index_data as dat
    dates = pd.bdate_range('2023-11-01', '2023-12-29'); t, s = pc.rebalance_days(dates, 0, '2023-12-01')[0]
    pm = pd.DataFrame([{'trade_date': t, 'signal_date': s, 'status': 'ok', 'tier_source': 'T', 'raw_candidates': 2, 'ranked': 2}])
    p = pd.DataFrame({'trade_date': [t, t], 'signal_date': [s, s], 'ticker': ['A', 'A'], 'rank': [1, 2]})
    m = dat.month_calendar(dates, 0, '2023-12-01', dates[-1], pm, p)
    snaps = pd.DataFrame({'date': pd.to_datetime(['2023-11-24']), 'stock_code': ['A'], 'market_type': ['KOSPI']})
    class Conn:
        def cursor(self): raise AssertionError('no SQL expected in this test')
    import stage2_common as sc
    orig = sc.q
    sc.q = lambda conn, sql, params=None: (pd.DataFrame({'stock_code': ['A'], 'market_cap': [1e10]}) if 'MarketCapDaily' in sql
                                           else pd.DataFrame({'stock_code': ['A'], 'tier_date': ['2023-11-30'], 'tier': [1], 'liq': [2e9]}))
    try:
        u, miss = dat.build_universe(None, snaps, m, p); crashed = False
    except Exception as e:      # noqa: BLE001
        crashed = repr(e)
    finally:
        sc.q = orig
    check('Recommended: duplicated code month flagged and universe build does not crash (rank not mapped)',
          m.status.iloc[0] == 'panel_mismatch' and not crashed and int((u.p1_rank > 0).sum()) == 0, str(crashed))
    with tempfile.TemporaryDirectory() as d:
        try:
            dat.require_tickers([], d, {'offset': 0}); stopped = False
        except SystemExit:
            stopped = True
        reason = json.load(open(os.path.join(d, 'data_checks.json'))).get('fatal', '')
    check('Recommended: all-empty query stops before SQL with reason in data_checks.json', stopped and 'no tickers' in reason)


def t_h2_consistency_function():
    o = pd.DataFrame([{'account': 'M', 'div': 'observed', 'delist': 'last', 'cost': 'net', 'date': '2023-12-01', 'buy_gross': 50.0, 'sell_gross': 0.0}])
    f = pd.DataFrame(columns=pc.TABLE_SCHEMAS['fills.parquet'])
    bad, n = pc.fills_consistency(o, f, pc.KEY4 + ['date'])
    check('H2 order with positive buy but no fills -> inconsistent', n == 1)
    f2 = pd.DataFrame([{'account': 'M', 'div': 'observed', 'delist': 'last', 'cost': 'net', 'date': '2023-12-01', 'ticker': 'A', 'sell_qty': 0, 'buy_qty': 5,
                        'sell_value': 0.0, 'buy_value': 50.0}])
    check('H2 fills value sum equals order amount -> consistent', pc.fills_consistency(o, f2, pc.KEY4 + ['date'])[1] == 0)
    ao = np.array([[10.0, 7.0], [10.0, 7.0], [10.0, 7.0]]); mk = market(('2020-01-01', 3), 2, ao, ao)
    import pool_vs_index_sim as sim
    costs = ac.Costs(0.00015, True, lambda d: 0.002)
    _, lg = ac.run_fractional(mk, events({1: np.array([0.3, 0.7]), 2: np.array([0.6, 0.4])}, fills=True), costs, False, 1000.0)
    orders = pd.DataFrame(lg['orders']).assign(account='P1', div='observed', delist='last', cost='net')
    fills = sim.fills_frame({'P1|observed|last|net': lg['fills']}, mk.dates, mk.tickers)
    check('H2 real accounting output: fills values sum to order buy/sell gross (1e-6 won)', pc.fills_consistency(orders, fills, pc.KEY4 + ['date'])[1] == 0)


# ---------------- DB ----------------

def t_db_reentry():
    import stage2_common as sc
    import pool_vs_index_data as dat
    conn = sc.connect()
    s = pc.capped_query(conn, "SELECT snapshot_date AS date, stock_code, market_type FROM TickerUniverseSnapshot "
                              "WHERE snapshot_date >= '2013-01-01' AND snapshot_date <= %s", [pc.END_CAP], pc.END_CAP)
    s['date'] = pd.to_datetime(s.date)
    ev = dat.exit_events(s, ['036220', '101970'])
    for t, ex, re_ in (('036220', '2016-05', '2024-03'), ('101970', '2015-03', '2025-03')):
        g = ev[ev.ticker == t]
        got_ex = g[g.event == 'exit'].snapshot_date.dt.strftime('%Y-%m').tolist(); got_re = g[g.event == 'reentry'].snapshot_date.dt.strftime('%Y-%m').tolist()
        check(f'db reentry {t}: exit {ex}, reentry {re_}', ex in got_ex and re_ in got_re, f'exit={got_ex} reentry={got_re}')
    return conn


def t_db_ticks(conn):
    """F7: 2014~2022 k=0 패널 후보의 교체일 양수 원 시가가 당시 호가표로 바뀌지 않아야 한다(이전 단일 표는 13건 변경)."""
    import stage2_common as sc
    p = pd.read_csv('/mnt/volume/workspace/split-research/runs/selection-vs-etf-r1/panel_v2.csv', dtype={'ticker': str}, parse_dates=['trade_date', 'signal_date'])
    p = p[p.trade_date <= '2022-12-31']; t = sorted(p.ticker.unique()); ph = ','.join(['%s'] * len(t))
    px = sc.q(conn, f"SELECT stock_code, date, open_price FROM DailyStockPrice WHERE stock_code IN ({ph}) AND date BETWEEN '2014-01-01' AND '2022-12-31'", t)
    px['date'] = pd.to_datetime(px.date); px['open_price'] = pd.to_numeric(px.open_price)
    mk = sc.q(conn, f"SELECT snapshot_date, stock_code, market_type FROM TickerUniverseSnapshot WHERE stock_code IN ({ph}) "
                    f"AND snapshot_date BETWEEN '2013-12-01' AND '2022-12-31'", t)
    mk['snapshot_date'] = pd.to_datetime(mk.snapshot_date)
    m = p.merge(px, left_on=['ticker', 'trade_date'], right_on=['stock_code', 'date'])
    m = m[m.open_price > 0].sort_values('signal_date')
    mk = mk.sort_values('snapshot_date')
    m = pd.merge_asof(m, mk.rename(columns={'stock_code': 'ticker'}), left_on='signal_date', right_on='snapshot_date', by='ticker')
    changed = int(sum(pc.tick_up(o, mk_, d) != o for o, mk_, d in zip(m.open_price, m.market_type, m.trade_date)))
    old = sum(float(np.ceil(round(o / _flat(o), 5)) * _flat(o)) != o for o in m.open_price)   # 이전 단일 표(BasicExecutionHandler와 같음)
    check('F7 db: historical tick keeps all 2014-2022 candidate raw opens (old flat table changed some)', changed == 0 and old > 0, f'n={len(m)} changed={changed} old_changed={old}')


def _flat(p):
    for upper, tick in pc.TICK_TABLES['UNIFIED']:
        if upper is None or p < upper: return tick


def main():
    tests = (t_split_conservation, t_suspension_across_boundary, t_open_attribution, t_dividend_after_observation, t_cash_nonnegative_after_costs,
             t_frozen_budget_example, t_zero_candidate_month, t_delist_generations, t_cells_and_terciles, t_labels, t_cap,
             t_f1_q1_q3_rows, t_f9_q3_inputs, t_f10_key_and_calendar, t_f10_s20_checker, t_f2_blocked_buy_stays_cash, t_f2_integer_blocked,
             t_f3_panel_pit_and_status, t_f3_calendar_mismatch, t_f3_cell_error_recorded, t_f4_window, t_f4_etf_cap_lines, t_f4_truncation_checker,
             t_f4_fills_ticker_swap, t_f5_etf_entry_and_calc, t_f5_open_values, t_f6_holiday_snapshot_exit, t_f6_ratio_and_book,
             t_f6_dps_observed_vs_absent, t_f7_ticks, t_f8_turnover, t_g2_fills_everywhere, t_g4_dps_quality, t_g5_absence_rebuy_and_ca,
             t_g6_universe_and_calendar, t_r1_etf_inputs, t_r2_e0_same_day, t_r3_checks_and_internal_nan,
             t_h3_zero_candidates_and_rank, t_h4_strict_hash_and_pass_string, t_quality_window_and_tickers, t_duplicate_codes_and_empty_query,
             t_h2_consistency_function)
    for f in tests:
        try:
            f()
        except Exception as e:          # noqa: BLE001 - 검사 자체의 예외도 실패로 센다
            check(f'{f.__name__} raised', False, f'{type(e).__name__}: {e}')
    if '--db' in sys.argv:
        conn = t_db_reentry(); t_db_ticks(conn)
    bad = [r for r in RESULTS if not r[1]]
    print(f'selftest: {len(RESULTS) - len(bad)}/{len(RESULTS)} passed')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
