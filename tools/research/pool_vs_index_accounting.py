"""연구 (b) 회계: 지수식 소수 주 계좌(주)와 원주 정수 계좌(보조), ETF. DB·파일을 읽지 않는 순수 계산이다.

설계 DESIGN-v2.md 2절 + DESIGN-v2.1.md A1~A4·B, 구현 리뷰 수정 F2·F5·F6·F7·F8(review-reconciled-impl.md).
- 소수 주: 수정 시가·종가 단위. 분할 배수를 수량에 다시 곱하지 않는다(수정가격이 이미 연결).
- 체결 가능(소수 주): 원 시가 > 0, 거래량 > 0, 수정 시가 > 0. 정수 계좌는 원 시가·거래량만 본다(F7).
- 목표액_i = w_i × (현금 + 거래 가능 보유액(시가) + 거래 불가 보유액(직전 장부) + 퇴출 보유분 장부).
- 매도 먼저(수수료·결제일 기준 매도세) → 매수 배수 = 매도 후 현금 / (거래 불가 포함 신호 목표 전체 매수 수요 × (1+수수료)).
  거래 불가 몫은 배수를 적용한 금액만큼 현금으로 남는다(F2: 다른 종목으로 재배분하지 않음).
- 퇴출: 거래일 스냅숏은 그날 주문 뒤(장 마감), 휴장일 스냅숏은 다음 거래일 주문 전에 세대 기록으로 옮긴다(F6).
- 장부는 유한한 양수 종가로만 갱신한다(F6). 장부가 없는 보유는 held_without_book으로 센다.
- 배당: 권리일 종가 보유 × 그날 adj_ratio. 그날 값이 없으면 v2.2-note 1(G3): 직전 유효 값(20거래일 이내, 그 사이 계수 변경 기록 없음)을
  쓰고 근거를 기록한다. 아니면 그 종목 권리는 미확정(지급 안 함)으로 두고 수량을 기록한다.
  DPS 유효값 = 유한한 비음수. 관측 0원·미관측을 따로 센다. 권리일에는 수량·근거만 저장하고(G4), 30% 표시는 sim의 품질표에서 만든다.
- 휴장일 스냅숏 부재는 다음 거래일의 기업행위·체결 판단·주문보다 먼저 처리한다(G5). 부재 ~ 재등장 사이 주문은 미체결(Market.absent).
"""
from dataclasses import dataclass, field
import numpy as np, pandas as pd

CASH_EPS = 1e-6               # 부동소수 오차로 생긴 음수 현금 허용 폭(원). 넘으면 오류
CA_MIN_CHANGE = 0.002         # 정수 계좌: adj_ratio 변화가 이보다 크면 기업행위로 수량 환산(2단계와 같은 값)
RATIO_FALLBACK_MAX_DAYS = 20  # v2.2-note 1: 권리일 계수 결측 시 직전 유효 값을 쓸 수 있는 최대 거래일 수


@dataclass
class Market:
    dates: pd.DatetimeIndex
    tickers: list
    adj_open: np.ndarray
    book: np.ndarray            # 직전 유효(유한·양수) 수정 종가
    raw_open: np.ndarray
    raw_book: np.ndarray        # 직전 유효(유한·양수) 원 종가
    volume: np.ndarray
    ratio: np.ndarray           # adj_ratio(직전 유효 값으로 채움)
    close_missing: np.ndarray   # 그날 유효 종가 없음(장부는 직전값)
    full_calendar: pd.DatetimeIndex
    ratio_raw: np.ndarray = None   # 그날 adj_ratio(채우지 않음)
    ratio_last_idx: np.ndarray = None  # 그날까지 마지막 유효 adj_ratio의 날짜 위치(-1: 없음)
    implied_ratio: np.ndarray = None   # 그날 수정 종가 / 원 종가(adj_ratio 결측일의 변경 기록 확인용)
    absent: np.ndarray = None      # 스냅숏 부재 ~ 재등장 사이(주문 미체결, G5)

    def tradable_raw(self, i):
        r, v = self.raw_open[i], self.volume[i]
        with np.errstate(invalid='ignore'):
            ok = np.isfinite(r) & (r > 0) & np.isfinite(v) & (v > 0)
        return ok & ~self.absent[i] if self.absent is not None else ok

    def tradable(self, i):
        o = self.adj_open[i]
        with np.errstate(invalid='ignore'):
            return self.tradable_raw(i) & np.isfinite(o) & (o > 0)

    def prev_book(self, i):
        return self.book[i - 1] if i > 0 else self.book[i]


@dataclass
class Events:
    rebal: dict                 # 날짜 위치 → 목표 비중(길이 NT)
    exits_post: dict            # 날짜 위치 → 종목 위치 배열(그날 주문 뒤, 장 마감)
    exits_pre: dict             # 날짜 위치 → 종목 위치 배열(그날 주문 전; 휴장일 스냅숏)
    record: dict                # 날짜 위치 → 회계연도
    pay: dict                   # 날짜 위치 → 회계연도(장 마감 후 지급)
    dps: dict                   # 회계연도 → (DPS 배열(미관측 0), 관측 여부 배열)
    order: dict = field(default_factory=dict)    # 정수 계좌 매수 순서
    markets: dict = field(default_factory=dict)  # 정수 계좌: 날짜 위치 → 종목별 시장 배열(호가용)
    marks: tuple = ()           # 보유 상태를 기록할 날짜 위치(월말·연말)
    log_fills: bool = False     # 종목별 체결 수량 기록


@dataclass
class Costs:
    commission: float
    use_tax: bool
    tax_fn: object = None       # trade_date → 세율

    def tax(self, d):
        return self.tax_fn(d) if self.use_tax else 0.0


@dataclass
class State:
    q: np.ndarray
    cash: float
    dead: list = field(default_factory=list)    # (ticker, generation, qty, book_value, exit_date)
    gen: dict = field(default_factory=dict)
    rights: dict = field(default_factory=dict)
    stale_held_days: int = 0
    held_without_book_days: int = 0

    def dead_value(self):
        return float(sum(x[3] for x in self.dead))


def holdings_value(q, px):
    m = q > 0
    return float(np.dot(q[m], np.nan_to_num(px[m])))


def total_at_open(st, mk, i, trad):
    return st.cash + holdings_value(np.where(trad, st.q, 0), mk.adj_open[i]) + holdings_value(np.where(trad, 0, st.q), mk.prev_book(i)) + st.dead_value()


def sell_to_targets(st, mk, i, tq, trad, costs):
    sell = np.where(trad, np.maximum(st.q - tq, 0.0), 0.0)
    gross = holdings_value(sell, mk.adj_open[i])
    fee, tax = gross * costs.commission, gross * costs.tax(mk.dates[i])
    st.q = st.q - sell; st.cash += gross - fee - tax
    return gross, fee, tax, sell


def blocked_demand(st, mk, i, w, total, trad):
    """거래 불가 종목의 신호 목표 매수 수요(금액). 배수 계산에 넣고, 그 몫은 현금으로 남긴다(F2)."""
    cur = np.nan_to_num(st.q * mk.prev_book(i))
    return np.where(~trad & (w > 0), np.maximum(w * total - cur, 0.0), 0.0)


def buy_to_targets(st, mk, i, tq, trad, blocked, costs):
    buy = np.where(trad, np.maximum(tq - st.q, 0.0), 0.0)
    gross = holdings_value(buy, mk.adj_open[i]); demand = gross + float(blocked.sum())
    need = demand * (1 + costs.commission)
    scale = 1.0 if need <= st.cash or need == 0 else max(st.cash, 0.0) / need
    st.q = st.q + scale * buy; st.cash -= scale * gross * (1 + costs.commission)
    if st.cash < -CASH_EPS: raise RuntimeError(f'negative cash {st.cash}')
    st.cash = max(st.cash, 0.0)
    return gross * scale, gross * scale * costs.commission, scale, scale * buy


def order_record(mk, i, total, s, b, w, trad, st, blocked):
    s_gross, s_fee, s_tax = s; b_gross, b_fee, scale = b
    frozen = (~trad) & (st.q > 0)
    return {'date': str(mk.dates[i].date()), 'total_at_open': total, 'sell_gross': s_gross, 'sell_fee': s_fee, 'sell_tax': s_tax,
            'buy_gross': b_gross, 'buy_fee': b_fee, 'buy_scale': scale, 'n_target': int((w > 0).sum()),
            'blocked_target_weight': float(w[~trad & (blocked > 0)].sum()), 'blocked_unbought_cash': float(blocked.sum() * scale),
            'n_frozen': int(frozen.sum()), 'frozen_book': holdings_value(np.where(frozen, st.q, 0), mk.prev_book(i)),
            'turnover_one_way': min(s_gross, b_gross) / total if total > 0 else 0.0,
            'turnover_two_way': (s_gross + b_gross) / total if total > 0 else 0.0, 'cash_after': st.cash}


def rebalance(st, mk, i, w, costs, fills=None):
    trad = mk.tradable(i)
    total = total_at_open(st, mk, i, trad)
    with np.errstate(invalid='ignore', divide='ignore'):
        tq = np.where(trad, w * total / mk.adj_open[i], st.q)
    blocked = blocked_demand(st, mk, i, w, total, trad)
    s_gross, s_fee, s_tax, sold = sell_to_targets(st, mk, i, tq, trad, costs)
    b_gross, b_fee, scale, bought = buy_to_targets(st, mk, i, tq, trad, blocked, costs)
    rec = order_record(mk, i, total, (s_gross, s_fee, s_tax), (b_gross, b_fee, scale), w, trad, st, blocked)
    rec['conservation_err'] = (total - s_fee - s_tax - b_fee) - total_at_open(st, mk, i, trad)
    if fills is not None: append_fills(fills, i, sold, bought, mk.adj_open[i])
    return rec


def append_fills(fills, i, sold, bought, price):
    """종목별 체결(G2·H2). 배열로 모은다: (위치 i, 종목 위치, 매도량, 매수량, 체결가). 체결 금액 합 = 주문 금액(H2 검사)."""
    j = np.where((sold > 0) | (bought > 0))[0]
    if len(j): fills.append((i, j.astype(np.int32), sold[j].astype(float), bought[j].astype(float), np.asarray(price, dtype=float)[j]))


def confirm_exits(st, mk, i, idx, delist_zero, when):
    """when='post': 그날 종가 장부, 'pre': 직전 장부(그날 주문 전)."""
    px = mk.book[i] if when == 'post' else mk.prev_book(i); out = []
    for j in idx:
        if st.q[j] <= 0: continue
        t = mk.tickers[j]; g = st.gen.get(t, 0); book = 0.0 if delist_zero else float(np.nan_to_num(st.q[j] * px[j]))
        st.dead.append((t, g, float(st.q[j]), book, str(mk.dates[i].date())))
        st.gen[t] = g + 1
        out.append({'date': str(mk.dates[i].date()), 'when': when, 'ticker': t, 'generation': g, 'qty': float(st.q[j]), 'book_written': book})
        st.q[j] = 0.0
    return out


def ratio_change_in_gap(mk, i, j, last):
    """직전 유효일 뒤 ~ 권리일 사이에 수정 종가/원 종가로 본 계수가 직전 값과 달라진 기록이 있는지."""
    imp = mk.implied_ratio[last + 1:i + 1, j] if mk.implied_ratio is not None else np.array([])
    imp = imp[np.isfinite(imp) & (imp > 0)]
    return bool(len(imp) and (np.abs(imp / mk.ratio[i, j] - 1) > CA_MIN_CHANGE).any())


def fallback_basis(mk, i, j):
    """v2.2-note 1: 직전 유효 계수 사용. 그 사이 계수 변경 기록이 있거나 20거래일보다 오래되면 미확정."""
    last = int(mk.ratio_last_idx[i, j]) if mk.ratio_last_idx is not None else -1
    if last < 0 or not np.isfinite(mk.ratio[i, j]): return 'undetermined_no_ratio', last
    if i - last > RATIO_FALLBACK_MAX_DAYS: return 'undetermined_stale', last
    if ratio_change_in_gap(mk, i, j, last): return 'undetermined_ratio_changed', last
    return 'previous_valid_ratio', last


def rights_quantity(st, mk, i):
    """그날 adj_ratio가 있으면 그 값. 없으면 fallback_basis에 따라 직전 유효 값 또는 미확정(지급 안 함, G3)."""
    held = st.q > 0
    today = mk.ratio_raw[i] if mk.ratio_raw is not None else mk.ratio[i]
    ok = np.isfinite(today) & (today > 0)
    ratio = np.where(ok, today, np.nan); cases = []
    for j in np.where(held & ~ok)[0]:
        basis, last = fallback_basis(mk, i, j)
        if basis == 'previous_valid_ratio': ratio[j] = mk.ratio[i, j]
        cases.append({'ticker': mk.tickers[j], 'qty': float(st.q[j]), 'basis': basis, 'ratio_used': float(ratio[j]) if np.isfinite(ratio[j]) else None,
                      'last_valid_ratio_date': str(mk.dates[last].date()) if last >= 0 else None,
                      'undetermined_raw_qty_est': float(st.q[j] * np.nan_to_num(mk.ratio[i, j])) if basis != 'previous_valid_ratio' else 0.0})
    return st.q * np.nan_to_num(ratio), cases


def record_rights(st, mk, i, fy, qty_raw, cases):
    """권리일 기록은 수량·당일 계수·가격 근거만(G4). DPS 이상치 표시는 관측 창 뒤 품질표에서 만든다."""
    st.rights[fy] = qty_raw
    valid_close = ~mk.close_missing[i]
    held = qty_raw > 0
    und = [c for c in cases if c['basis'] != 'previous_valid_ratio']
    return {'fy': fy, 'date': str(mk.dates[i].date()), 'holders': int(held.sum()), 'rights_qty_total': float(qty_raw.sum()),
            'n_ratio_fallback': len(cases) - len(und), 'n_rights_undetermined': len(und),
            'undetermined_qty': float(sum(c['qty'] for c in und)), 'holders_price_stale': int((held & ~valid_close).sum()),
            'ratio_cases': cases, 'qty_array': qty_raw}


def pay_dividend(st, mk, i, fy, dps_entry):
    r = st.rights.pop(fy, None)
    if r is None: return {'fy': fy, 'date': str(mk.dates[i].date()), 'amount': 0.0, 'holders': 0}
    dps, obs = dps_entry if dps_entry is not None else (np.zeros(len(r)), np.zeros(len(r), dtype=bool))
    held = r > 0; amt = float(np.dot(np.nan_to_num(r), dps)); st.cash += amt
    return {'fy': fy, 'date': str(mk.dates[i].date()), 'amount': amt, 'holders': int(held.sum()),
            'holders_observed_positive': int((held & obs & (dps > 0)).sum()), 'holders_observed_zero': int((held & obs & (dps == 0)).sum()),
            'holders_unobserved': int((held & ~obs).sum()), 'rights_qty_unobserved': float(r[held & ~obs].sum())}


def mark_state(st, mk, i):
    trad = mk.tradable(i); held = st.q > 0; frozen = held & ~trad; px = mk.book[i]
    v = holdings_value(st.q, px); total = st.cash + v + st.dead_value()
    fb = holdings_value(np.where(frozen, st.q, 0), px)
    rec = {'date': str(mk.dates[i].date()), 'n_held': int(held.sum()), 'n_frozen': int(frozen.sum()), 'frozen_qty': float(st.q[frozen].sum()),
           'frozen_book': fb, 'frozen_disposable': 0.0, 'frozen_weight': fb / total if total > 0 else 0.0,
           'n_dead': len(st.dead), 'dead_qty': float(sum(x[2] for x in st.dead)), 'dead_book': st.dead_value(), 'dead_disposable': 0.0,
           'dead_weight': st.dead_value() / total if total > 0 else 0.0, 'stale_close_held': int((held & mk.close_missing[i]).sum()),
           'stale_held_days_cum': st.stale_held_days, 'held_without_book_days_cum': st.held_without_book_days, 'holdings': v, 'cash': st.cash}
    detail = [{'date': rec['date'], 'kind': 'frozen', 'ticker': mk.tickers[j], 'generation': st.gen.get(mk.tickers[j], 0),
               'qty': float(st.q[j]), 'book': float(np.nan_to_num(st.q[j] * px[j])), 'disposable': 0.0} for j in np.where(frozen)[0]]
    detail += [{'date': rec['date'], 'kind': 'dead', 'ticker': t, 'generation': g, 'qty': q, 'book': b, 'disposable': 0.0} for t, g, q, b, _ in st.dead]
    return rec, detail


def close_bookkeeping(st, mk, i):
    held = st.q > 0
    st.stale_held_days += int((held & mk.close_missing[i]).sum())
    st.held_without_book_days += int((held & ~np.isfinite(mk.book[i])).sum())


def run_fractional(mk, ev, costs, delist_zero, cash0):
    """반환: (일별 종가 평가액 Series, 기록 dict). 0번째 날은 기준일(현금 cash0). log['open_values']는 매일 주문 전 시가 평가액."""
    st = State(q=np.zeros(len(mk.tickers)), cash=cash0)
    log = {'orders': [], 'exits': [], 'rights': [], 'pays': [], 'marks': [], 'mark_detail': [], 'fills': [] if ev.log_fills else None}
    vals, opens = np.empty(len(mk.dates)), np.empty(len(mk.dates))
    for i in range(len(mk.dates)):
        if i in ev.exits_pre: log['exits'] += confirm_exits(st, mk, i, ev.exits_pre[i], delist_zero, 'pre')   # 그날 모든 처리보다 먼저(G5)
        opens[i] = total_at_open(st, mk, i, mk.tradable(i)) if i > 0 else st.cash
        if i in ev.rebal: log['orders'].append(rebalance(st, mk, i, ev.rebal[i], costs, log['fills']))
        if i in ev.exits_post: log['exits'] += confirm_exits(st, mk, i, ev.exits_post[i], delist_zero, 'post')
        if i in ev.record:
            qty, cases = rights_quantity(st, mk, i)
            log['rights'].append(record_rights(st, mk, i, ev.record[i], qty, cases))
        if i in ev.pay: log['pays'].append(pay_dividend(st, mk, i, ev.pay[i], ev.dps.get(ev.pay[i])))
        close_bookkeeping(st, mk, i)
        vals[i] = st.cash + holdings_value(st.q, mk.book[i]) + st.dead_value()
        if i in ev.marks:
            rec, det = mark_state(st, mk, i); log['marks'].append(rec); log['mark_detail'] += det
    log['unpaid_rights_fy'] = sorted(st.rights)
    log['open_values'] = pd.Series(opens, index=mk.dates)
    return pd.Series(vals, index=mk.dates), log


# ---------------- 원주 정수 계좌(보조) ----------------

def apply_corporate_actions(st, mk, i, log):
    if i == 0: return
    with np.errstate(invalid='ignore', divide='ignore'):
        ca = mk.ratio[i] / mk.ratio[i - 1]
    ca = np.where(np.isfinite(ca) & (np.abs(ca - 1) > CA_MIN_CHANGE), ca, 1.0)
    for j in np.where((ca != 1.0) & (st.q > 0))[0]:
        nq = st.q[j] * ca[j]; whole = np.floor(nq + 1e-9); frac = max(nq - whole, 0.0)
        price = mk.raw_book[i - 1, j] / ca[j]; st.cash += np.floor(frac * price)
        log.append({'date': str(mk.dates[i].date()), 'ticker': mk.tickers[j], 'factor': float(ca[j]), 'fraction': float(frac),
                    'fraction_cash': float(np.floor(frac * price))})
        st.q[j] = whole


def int_total(st, mk, i, trad):
    prev = mk.raw_book[i - 1] if i > 0 else mk.raw_book[i]
    return st.cash + holdings_value(np.where(trad, st.q, 0), mk.raw_open[i]) + holdings_value(np.where(trad, 0, st.q), prev) + st.dead_value()


def int_targets(st, mk, i, w, trad, price):
    total = int_total(st, mk, i, trad)
    tq = st.q.copy(); vacated = 0
    for j in np.where(trad & (w > 0))[0]:
        tq[j] = np.floor(w[j] * total / price[j]); vacated += int(tq[j] == 0)
    tq[trad & (w <= 0)] = 0.0
    return tq, total, vacated


def int_sell(st, mk, i, tq, trad, costs, price):
    gross = 0.0
    for j in np.where(trad & (st.q > tq))[0]:
        dq = st.q[j] - tq[j]; rev = dq * price[j]
        st.cash += np.floor(rev * (1 - costs.commission - costs.tax(mk.dates[i]))); st.q[j] -= dq; gross += rev
    return gross


def int_buy_caps(st, mk, i, w, tq, trad, price, total, costs):
    """정수 계좌도 거래 불가 몫의 수요를 배수에 넣는다(F2): 각 매수 상한 = floor(배수 × 목표 차이)."""
    prev = mk.raw_book[i - 1] if i > 0 else mk.raw_book[i]
    blocked = np.where(~trad & (w > 0), np.maximum(w * total - np.nan_to_num(st.q * prev), 0.0), 0.0).sum()
    want = np.where(trad, np.maximum(tq - st.q, 0.0), 0.0)
    need = (float(np.dot(want, np.nan_to_num(price))) + blocked) * (1 + costs.commission)
    scale = 1.0 if need <= st.cash or need == 0 else st.cash / need
    return np.floor(want * scale + 1e-9) if scale < 1.0 else want, scale


def int_buy(st, mk, i, caps, order, costs, price):
    gross, short = 0.0, 0
    for j in order:
        dq = caps[j]
        if dq <= 0: continue
        p = price[j]; cost = lambda n: n * p + np.floor(n * p * costs.commission)
        if cost(dq) > st.cash:
            short += 1; dq = np.floor(st.cash / (p * (1 + costs.commission)))
            while dq > 0 and cost(dq) > st.cash: dq -= 1
        if dq <= 0: continue
        st.cash -= cost(dq); st.q[j] += dq; gross += dq * p
    return gross, short


def int_prices(mk, i, markets, tick_up):
    """당시 시장·날짜 호가로 올림(유효한 실제 시가는 그대로). 바뀐 건수를 센다."""
    raw = mk.raw_open[i]; d = mk.dates[i]
    px = np.array([tick_up(raw[j], markets[j], d) if np.isfinite(raw[j]) and raw[j] > 0 else np.nan for j in range(len(raw))])
    return px, int(np.nansum((px != raw) & np.isfinite(px)))


def int_rebalance(st, mk, i, w, order, costs, markets, tick_up, fills=None):
    trad = mk.tradable_raw(i)
    price, changed = int_prices(mk, i, markets, tick_up)
    tq, total, vacated = int_targets(st, mk, i, w, trad, price)
    q0 = st.q.copy()
    s = int_sell(st, mk, i, tq, trad, costs, price)
    q1 = st.q.copy()
    caps, scale = int_buy_caps(st, mk, i, w, tq, trad, price, total, costs)
    b, short = int_buy(st, mk, i, caps, [j for j in order if trad[j]], costs, price)
    if st.cash < 0: raise RuntimeError('negative cash (integer account)')
    if fills is not None: append_fills(fills, i, q0 - q1, st.q - q1, price)      # 정수 계좌 종목별 체결(G2), 호가 반영 체결가
    return {'date': str(mk.dates[i].date()), 'total_at_open': total, 'sell_gross': s, 'buy_gross': b, 'buy_scale': scale,
            'vacated_slots': vacated, 'budget_short': short, 'n_target': int((w > 0).sum()), 'open_changed_by_tick': changed,
            'n_adj_open_missing_but_raw_tradable': int((trad & ~mk.tradable(i)).sum()), 'cash_after': st.cash}


def run_integer(mk, ev, costs, delist_zero, cash0, tick_up):
    st = State(q=np.zeros(len(mk.tickers)), cash=cash0)
    log = {'orders': [], 'exits': [], 'ca': [], 'pays': [], 'fills': []}
    vals = np.empty(len(mk.dates))
    raw_mk = Market(mk.dates, mk.tickers, mk.raw_open, mk.raw_book, mk.raw_open, mk.raw_book, mk.volume, mk.ratio,
                    mk.close_missing, mk.full_calendar, mk.ratio_raw)
    mkt = None
    for i in range(len(mk.dates)):
        if i in ev.exits_pre: log['exits'] += confirm_exits(st, raw_mk, i, ev.exits_pre[i], delist_zero, 'pre')   # 기업행위보다 먼저(G5)
        apply_corporate_actions(st, mk, i, log['ca'])
        mkt = ev.markets.get(i, mkt)
        if i in ev.rebal: log['orders'].append(int_rebalance(st, mk, i, ev.rebal[i], ev.order.get(i, []), costs, mkt, tick_up, log['fills']))
        if i in ev.exits_post: log['exits'] += confirm_exits(st, raw_mk, i, ev.exits_post[i], delist_zero, 'post')
        if i in ev.record: st.rights[ev.record[i]] = st.q.copy()
        if i in ev.pay: log['pays'].append(pay_dividend(st, mk, i, ev.pay[i], ev.dps.get(ev.pay[i])))
        vals[i] = st.cash + holdings_value(st.q, mk.raw_book[i]) + st.dead_value()
    return pd.Series(vals, index=mk.dates), log


# ---------------- ETF ----------------

def etf_entry(etf, start_i):
    """start_i(첫 교체일) 이후 매일 확인해 첫 유효(ok·시가 양수) 날."""
    ok = etf.ok.values.astype(bool) & np.isfinite(etf.open.values) & (etf.open.values > 0)
    cand = np.where(ok[start_i:])[0]
    return start_i + int(cand[0]) if len(cand) else None


def run_etf(dates, etf, start_i, commission, cash0):
    """첫 유효일 시가에 소수 주 매수, 보유. 반환 dict: calc(계산용: 무효 날 NaN, 진입 전 현금), display(그림용: 직전 유효값),
    ok(낙폭·월말 유효일), entered(진입 여부), entry(위치, 없으면 None), shares.
    끝까지 진입하지 못하면 진입 전과 같은 현금·유효 상태를 기록한다(R1: 절단 실행과 긴 실행이 같은 날짜에 같은 값)."""
    e = etf_entry(etf, start_i)
    if e is None:
        cash = pd.Series(cash0, index=dates, dtype=float)
        return {'calc': cash, 'display': cash.copy(), 'ok': pd.Series(True, index=dates), 'entered': pd.Series(False, index=dates),
                'entry': None, 'shares': 0.0}
    n = cash0 / (etf.open.values[e] * (1 + commission))
    c = etf.close.values
    valid = etf.ok.values.astype(bool) & np.isfinite(c) & (c > 0)
    calc = np.where(np.arange(len(dates)) < e, cash0, np.where(valid, n * c, np.nan))
    calc_s = pd.Series(calc, index=dates)
    ok = pd.Series(np.where(np.arange(len(dates)) < e, True, valid), index=dates)
    return {'calc': calc_s, 'display': calc_s.ffill(), 'ok': ok, 'entered': pd.Series(np.arange(len(dates)) >= e, index=dates), 'entry': e, 'shares': n}
