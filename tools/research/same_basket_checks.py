"""같은 종목 대조(same-basket-r1) 산출물 읽기와 기술 검사(설계 10절 2·3·5·6·7과 형식·대사). 성과 값을 출력하지 않는다.

same_basket_verdict.py(조립 전 필수 검사)와 same_basket_smoke_check.py가 함께 쓴다. 모든 검사는 오류 문자열 목록을 돌려준다.
"""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import same_basket_common as sb
import same_basket_ledger as ledger

MAIN_RATIO = 0.022                  # 3월 승인 후보의 주문율(주 시험·연장)
VALUE_TOL = 1e-3                    # 원
F32_STEP = 1.0                      # float32 현금의 배당 입금 한 번당 허용 오차(원)
NOT_READY = ('LOOKUP_ERROR', 'NO_SIGNAL_DATE', 'NOT_EVALUATED')
SWR_ARGS = lambda div: ['--tax', 'period', '--div', div, '--hold', 'none']


# ---------------- 읽기 ----------------
def csv_or_empty(path, **kw):
    return pd.read_csv(path, **kw) if os.path.getsize(path) > 2 else pd.DataFrame()


def read_x(prefix):
    curve = pd.read_csv(prefix + '_curve.csv', parse_dates=['date']).set_index('date')
    state = pd.read_csv(prefix + '_state.csv', parse_dates=['date']).set_index('date')
    fills = csv_or_empty(prefix + '_fills.csv', dtype={'ticker': str}, parse_dates=['date'])
    if fills.empty: fills = pd.DataFrame(columns=['date', 'ticker', 'type', 'dq', 'dcash', 'order', 'reason'])
    el = pd.read_csv(prefix + '_eligible.csv', dtype={'signal_date': str, 'tier_source': str}, parse_dates=['date']).set_index('date')
    el[['signal_date', 'tier_source']] = el[['signal_date', 'tier_source']].fillna('')
    return {'curve': curve, 'state': state, 'fills': fills, 'divs': csv_or_empty(prefix + '_divs.csv', dtype={'ticker': str}), 'eligible': el,
            'endpos': pd.read_csv(prefix + '_endpos.csv', dtype={'ticker': str}),
            'meta': json.load(open(prefix + '_meta.json')), 'sbmeta': json.load(open(prefix + '_sbmeta.json'))}


def read_h(prefix):
    return {'curve': pd.read_csv(prefix + '_H_curve.csv', parse_dates=['date']).set_index('date'),
            'pos': csv_or_empty(prefix + '_H_pos.csv', dtype={'ticker': str}, parse_dates=['date']),
            'buys': csv_or_empty(prefix + '_H_buys.csv', dtype={'ticker': str}, parse_dates=['date']),
            'divs': csv_or_empty(prefix + '_H_divs.csv', dtype={'ticker': str}), 'meta': json.load(open(prefix + '_H_meta.json'))}


def read_k(prefix):
    return {'curve': pd.read_csv(prefix + '_K_curve.csv', parse_dates=['date']).set_index('date').value, 'meta': json.load(open(prefix + '_K_meta.json'))}


def read_job(run_dir, job):
    """작업 한 줄의 산출물. 보조 작업은 X만 있다. 읽기 실패는 (None, 오류)로 돌려준다."""
    prefix = os.path.join(run_dir, job.prefix)
    try:
        out = {'x': read_x(prefix)}
        if job.kind != 'aux': out.update(h=read_h(prefix), k=read_k(prefix))
        return out, None
    except Exception as exc:
        return None, f'{job.prefix}: cannot read outputs ({type(exc).__name__}: {exc})'


def paid_by_date(divs):
    if divs.empty or 'event' not in divs: return pd.Series(dtype=float)
    p = divs[divs.event == 'paid']
    return p.groupby(pd.to_datetime(p.date)).amount.sum()


# ---------------- X ----------------
def expect_account(job):
    aux = job.kind == 'aux'
    return {'initial_cash': sb.SLOT_CASH if aux else sb.CASH0, 'order_investment_ratio': sb.AUX_RATIOS[job.ratio] if aux else MAIN_RATIO,
            'only': job.ticker if aux else None}


def x_meta_errors(job, run, dates, basket_sha):
    m, s, want, p = run['meta'], run['sbmeta'], expect_account(job), job.prefix
    errs = []
    if not m.get('success') or m.get('degraded_run'): errs.append(f'{p}: engine success={m.get("success")} degraded={m.get("degraded_run")}')
    if m.get('universe_mode') != 'strict_pit' or m.get('env_universe_mode') != 'strict_pit': errs.append(f'{p}: universe mode is not strict_pit')
    if m.get('candidate_lookup_error_count', 1) or m.get('source_lookup_error_days', 1): errs.append(f'{p}: candidate lookup errors')
    base = str(sb.prev_day(dates, job.start).date())
    for key, val in (('account_start', job.start), ('engine_start', base), ('end', job.end), ('warmup', True), ('restrict', True), ('delist', job.delist),
                     ('eligible_all', job.kind == 'ext'),
                     ('basket_sha256', basket_sha), ('only', want['only']), ('swr_args', SWR_ARGS(job.div)), ('fills_before_start', 0),
                     ('buys_outside_allowed', []), ('signal_ready', True)):
        if s.get(key) != val: errs.append(f'{p}: sbmeta {key}={s.get(key)!r} (expected {val!r})')
    for key in ('ever_candidate', 'start_unaffordable', 'start_no_open', 'basket_delisted'):
        if not isinstance(s.get(key), (list, dict)): errs.append(f'{p}: sbmeta {key} missing')
    for key in ('initial_cash', 'order_investment_ratio'):
        if not np.isclose(s.get(key, np.nan), want[key], rtol=0, atol=1e-12): errs.append(f'{p}: {key}={s.get(key)} (expected {want[key]})')
    return errs


def x_value_errors(job, run, dates):
    c, st, el, p = run['curve'], run['state'], run['eligible'], job.prefix
    days = dates[(dates >= sb.prev_day(dates, job.start)) & (dates <= pd.Timestamp(job.end))]
    errs = [f'{p}: {name} dates differ from trading dates' for name, idx in (('curve', c.index), ('state', st.index), ('eligible', el.index)) if not idx.equals(days)]
    if errs: return errs
    num = np.column_stack([c[['value', 'value_engine', 'receivable']].to_numpy(float), st[['engine_value', 'cash', 'n_tickers', 'n_lots']].to_numpy(float)])
    if not np.isfinite(num).all(): return [f'{p}: missing or infinite values']
    if (c.value <= 0).any(): errs.append(f'{p}: non-positive value')
    if not np.allclose(c.value, c.value_engine + c.receivable, rtol=0, atol=VALUE_TOL): errs.append(f'{p}: value != engine + receivable')
    if not np.allclose(c.value_engine, st.engine_value, rtol=0, atol=VALUE_TOL): errs.append(f'{p}: state engine_value != curve')
    cash0 = expect_account(job)['initial_cash']
    if c.value.iloc[0] != cash0 or st.cash.iloc[0] != cash0 or st.n_tickers.iloc[0] != 0: errs.append(f'{p}: account is not all cash on the day before start')
    return errs


def x_signal_errors(job, run, dates):
    """10절 3: 첫 주문 가능일은 등록한 시작일이고 그날의 신호일은 직전 거래일이다. 직전 거래일에는 신호도 체결도 없다."""
    el, fills, p = run['eligible'], run['fills'], job.prefix
    start, base = pd.Timestamp(job.start), sb.prev_day(dates, job.start)
    errs = []
    if base not in el.index or start not in el.index: return [f'{p}: eligible log misses start or prior day']
    if el.loc[base, 'signal_date'] != '' or el.loc[base, 'n_entry_signals'] != 0: errs.append(f'{p}: signals on the day before start')
    if el.loc[start, 'signal_date'] != str(base.date()): errs.append(f'{p}: start-day signal date is not the prior trading day')
    if str(el.loc[start, 'tier_source']).startswith(NOT_READY): errs.append(f'{p}: prior-day signal was not prepared')
    if el.tier_source.str.startswith('LOOKUP_ERROR').any(): errs.append(f'{p}: candidate lookup error in the eligibility log')
    if job.kind == 'ext' and (el.tier_source.iloc[1:].str.startswith(NOT_READY).any() or el.n_eligible.iloc[1:].isna().any()):
        errs.append(f'{p}: eligibility log of the extension run has unevaluated days')
    if len(fills) and fills.date.min() < start: errs.append(f'{p}: fill before start')
    prev = pd.Series(el.index[:-1].strftime('%Y-%m-%d'), index=el.index[1:])
    if not (el.signal_date.iloc[1:] == prev).all(): errs.append(f'{p}: signal date is not always the prior trading day')
    return errs


def x_basket_errors(job, run, allowed):
    """10절 2: 매수는 모두 허용 명단 안이다(신규 진입이 명단 안이면 추가매수도 명단 안)."""
    buys = run['fills'][run['fills'].type == 'BUY']
    outside = sorted(set(buys.ticker) - set(allowed))
    return [f'{job.prefix}: buys outside the basket {outside}'] if outside else []


def x_cash_errors(job, run):
    """현금 보존: 일별 현금 = 시작 현금 + 체결 현금 변동 누계 + 배당 입금 누계."""
    st, fills = run['state'], run['fills']
    flow = fills.groupby('date').dcash.sum() if len(fills) else pd.Series(dtype=float)
    paid = paid_by_date(run['divs'])
    want = expect_account(job)['initial_cash'] + flow.reindex(st.index).fillna(0).cumsum() + paid.reindex(st.index).fillna(0).cumsum()
    tol = VALUE_TOL + F32_STEP * (paid.reindex(st.index).fillna(0) != 0).cumsum()
    bad = (st.cash - want).abs() > tol
    errs = [f'{job.prefix}: cash does not reconcile with fills and dividends on {int(bad.sum())} days'] if bad.any() else []
    held = fills.groupby('ticker').dq.sum() if len(fills) else pd.Series(dtype=float)
    held = held[held != 0]; end = run['endpos'].set_index('ticker').quantity
    if not held.sort_index().astype(float).equals(end.sort_index().astype(float)): errs.append(f'{job.prefix}: end positions differ from cumulative fills')
    return errs


NEVER_BOUGHT_REASONS = ('order_below_one_share', 'never_tier1_candidate', 'other')


def never_bought(basket, bought, unaffordable, ever_candidate):
    """기간 안에 X가 한 번도 사지 못한 명단 종목과 사유: order_below_one_share(시작일 주문액 < 1주 가격) /
    never_tier1_candidate(기간 중 한 번도 1등급 후보에 없었음) / other(후보였던 날이 있었지만 사지 못함: 빈자리 없음·재진입 대기 등)."""
    out = {}
    for t in basket:
        if t in bought: continue
        out[t] = 'order_below_one_share' if t in unaffordable else ('never_tier1_candidate' if t not in ever_candidate else 'other')
    return out


def x_delist_errors(job, run):
    """폐지일 다음 거래일부터는 체결이 없어야 한다(그 뒤 가격 행이 없음을 입력 점검이 보장하고, 여기서 다시 확인)."""
    f, dl = run['fills'], run['sbmeta'].get('basket_delisted') or {}
    bad = [t for t, d in dl.items() if len(f) and ((f.ticker == t) & (f.date > pd.Timestamp(d))).any()]
    return [f'{job.prefix}: fills after delisting for {bad}'] if bad else []


def x_errors(job, run, dates, basket, basket_sha):
    allowed = [job.ticker] if job.kind == 'aux' else basket
    errs = x_meta_errors(job, run, dates, basket_sha) + x_value_errors(job, run, dates)
    return errs if errs else x_signal_errors(job, run, dates) + x_basket_errors(job, run, allowed) + x_cash_errors(job, run) + x_delist_errors(job, run)


# ---------------- H ----------------
def h_buy_errors(job, h, basket):
    b, p = h['buys'], job.prefix
    if b.empty: return [] if len(h['meta']['skipped_at_start']) == sb.N_BASKET else [f'{p}: H has no buys']
    init = b[b.kind == 'initial']; errs = []
    if (init.date != pd.Timestamp(job.start)).any(): errs.append(f'{p}: H initial buy not on the start day')
    if set(init.ticker) | set(h['meta']['skipped_at_start']) != set(basket) or init.ticker.duplicated().any(): errs.append(f'{p}: H initial buys do not cover the basket once')
    if not set(b.ticker) <= set(basket): errs.append(f'{p}: H bought outside the basket')
    if (init.budget != sb.SLOT_CASH).any(): errs.append(f'{p}: H initial budget is not the slot cash')
    for r in b.itertuples():
        ok = (r.price == ledger.tick_up(r.open) and r.cost == ledger.buy_cost(r.quantity, r.price) and r.cost <= r.budget + VALUE_TOL
              and ledger.buy_cost(r.quantity + 1, r.price) > r.budget and r.quantity >= 0)
        if not ok: errs.append(f'{p}: H buy of {r.ticker} on {r.date.date()} breaks the price/quantity/fee rule'); break
    return errs


def h_value_errors(job, h, dates):
    c, p = h['curve'], job.prefix
    days = dates[(dates >= sb.prev_day(dates, job.start)) & (dates <= pd.Timestamp(job.end))]
    if not c.index.equals(days): return [f'{p}: H dates differ from trading dates']
    if not np.isfinite(c.to_numpy(float)).all(): return [f'{p}: H has missing or infinite values']
    errs = []
    if not np.allclose(c.value, c.cash + c.stock_value + c.receivable, rtol=0, atol=VALUE_TOL): errs.append(f'{p}: H value != cash + stock + receivable')
    if c.value.iloc[0] != sb.CASH0 or c.cash.iloc[0] != sb.CASH0: errs.append(f'{p}: H is not all cash on the day before start')
    cost = h['buys'].groupby('date').cost.sum() if len(h['buys']) else pd.Series(dtype=float)
    want = sb.CASH0 - cost.reindex(c.index).fillna(0).cumsum() + paid_by_date(h['divs']).reindex(c.index).fillna(0).cumsum()
    if not np.allclose(c.cash, want, rtol=0, atol=VALUE_TOL): errs.append(f'{p}: H cash does not reconcile with buys and dividends')
    pos = h['pos']
    stock = (pos.quantity * pos.price).groupby(pos.date).sum().reindex(c.index).fillna(0) if len(pos) else pd.Series(0.0, index=c.index)
    if not np.allclose(c.stock_value, stock, rtol=0, atol=VALUE_TOL): errs.append(f'{p}: H stock value differs from positions')
    return errs + h_quantity_errors(job, h)


def h_quantity_errors(job, h):
    """H는 팔지 않는다: 종목별 보유 수량은 줄지 않고 마지막 수량은 매수 수량 합과 같다."""
    pos, buys = h['pos'], h['buys']
    if pos.empty: return []
    wide = pos.pivot(index='date', columns='ticker', values='quantity').fillna(0)
    errs = [f'{job.prefix}: H quantity decreased'] if (wide.diff().iloc[1:] < 0).any().any() else []
    bought = buys.groupby('ticker').quantity.sum(); bought = bought[bought > 0]
    if not wide.iloc[-1].sort_index().astype(float).equals(bought.sort_index().astype(float)): errs.append(f'{job.prefix}: H final quantity differs from buys')
    return errs


def h_errors(job, h, dates, basket, basket_sha):
    m, p = h['meta'], job.prefix
    errs = [f'{p}: H meta {k}={m.get(k)!r} (expected {v!r})' for k, v in (('account_start', job.start), ('end', job.end), ('div', job.div), ('delist', job.delist),
            ('basket_sha256', basket_sha), ('price_basis', 'adjusted')) if m.get(k) != v]
    if job.div == 'none' and len(h['divs']): errs.append(f'{p}: H has dividends under the no-dividend assumption')
    return errs + h_value_errors(job, h, dates) + h_buy_errors(job, h, basket)


# ---------------- K ----------------
def k_errors(job, k, dates, etf, archived=None):
    """K는 etf_curve 식을 다시 계산한 값과 같아야 한다. archived(보관된 KODEX 곡선)가 있으면 겹치는 날짜에서 같아야 한다(10절 5)."""
    c, p = k['curve'], job.prefix
    days = dates[(dates >= sb.prev_day(dates, job.start)) & (dates <= pd.Timestamp(job.end))]
    if not c.index.equals(days): return [f'{p}: K dates differ from trading dates']
    again, _ = ledger.kodex_curve(etf, days[1:])
    errs = [] if c.iloc[0] == sb.CASH0 and np.allclose(c.iloc[1:], again, rtol=0, atol=VALUE_TOL) else [f'{p}: K differs from the etf_curve formula']
    if archived is not None:
        arch = archived.dropna(); want = c.index[1:][c.index[1:] <= arch.index.max()]
        if not len(want) or not want.isin(arch.index).all() or not np.allclose(c[want], arch[want], rtol=0, atol=VALUE_TOL):
            errs.append(f'{p}: K differs from the archived KODEX curve')
    return errs


def archived_kodex(year, path_template=sb.STAGE2_CURVES):
    c = pd.read_csv(path_template.format(year=year), parse_dates=['date']).set_index('date')
    return c[f'{year}_ETF']


# ---------------- 계좌 사이 ----------------
def first_day_errors(job, out):
    """H와 K는 등록한 시작일에 산다(X의 체결 여부와 무관). H가 하나도 못 샀으면 명단 전체가 그날 거래 불가여야 한다."""
    buys, start = out['h']['buys'], pd.Timestamp(job.start)
    errs = []
    if len(buys) and buys.date.min() != start: errs.append(f'{job.prefix}: H first buy is not the start day')
    k = out['k']['curve']
    if k.index[1] != start or out['k']['meta']['shares'] <= 0: errs.append(f'{job.prefix}: K did not buy on the start day')
    return errs


def same_until(a, b, cut, name, prefix):
    a, b = a[a.index <= cut], b[b.index <= cut]
    same = a.index.equals(b.index) and np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=VALUE_TOL, equal_nan=True)
    return [] if same else [f'{prefix}: {name} of the extension run differs from the 2-year run before the cut']


def cut_errors(main_job, main_out, ext_out):
    """10절 6: 연장 계좌를 2년 지점에서 자르면 2년 실행과 같다(X·H·K의 곡선, X의 체결·현금)."""
    cut, p = pd.Timestamp(main_job.end), main_job.prefix
    errs = same_until(ext_out['x']['curve'], main_out['x']['curve'], cut, 'X curve', p)
    errs += same_until(ext_out['x']['state'], main_out['x']['state'], cut, 'X state', p)
    errs += same_until(ext_out['h']['curve'], main_out['h']['curve'], cut, 'H curve', p)
    errs += same_until(ext_out['k']['curve'].to_frame(), main_out['k']['curve'].to_frame(), cut, 'K curve', p)
    cols = ['date', 'ticker', 'type', 'dq', 'dcash']
    fe = ext_out['x']['fills']; fe = fe[fe.date <= cut][cols].reset_index(drop=True); fm = main_out['x']['fills'][cols].reset_index(drop=True)
    if not fe.equals(fm): errs.append(f'{p}: X fills of the extension run differ from the 2-year run before the cut')
    return errs


def no_delisting_errors(freeze_job, freeze_out, zero_out):
    """폐지 종목이 없으면 0원 가정과 동결 가정의 X 곡선이 같아야 한다(가정 처리가 다른 값을 건드리지 않는지)."""
    if zero_out['h']['meta']['delisted']: return []          # 기간 안에 폐지된 명단 종목이 있으면 두 가정은 달라도 된다
    same = zero_out['x']['curve'].equals(freeze_out['x']['curve']) and zero_out['h']['curve'].equals(freeze_out['h']['curve'])
    return [] if same else [f'{freeze_job.prefix}: zero-delisting run differs from freeze without any delisting']
