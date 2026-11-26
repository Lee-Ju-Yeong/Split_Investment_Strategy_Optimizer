"""same-basket-r2(80% 보유 + 20% 분할) 공통: 상수·작업표·계좌 금액 전환·S 합산·투자 비중. DB·엔진을 읽지 않는 순수 모듈이다.

설계: runs/same-basket-r2/DESIGN-v1.md(이어받는 규칙: runs/same-basket-r1/DESIGN-v1.1.md). r1 도구(same_basket_*.py, 커밋 e7569f5)는
바꾸지 않고 불러 쓴다. r1 장부(same_basket_ledger)는 계좌 금액을 same_basket_common.CASH0·SLOT_CASH에서 읽으므로, 보유 몫(40만 원 × 20)과
대조 C의 작은 몫(10만 원 × 20)은 account() 안에서 그 두 값을 잠시 바꿔 같은 장부 코드를 돌린다(파일은 바꾸지 않는다).
"""
import contextlib, os
import numpy as np, pandas as pd
import same_basket_common as sb

R = sb.R
RUN_DIR = f'{R}/runs/same-basket-r2'
R1_DIR = sb.RUN_DIR
R1_FULL = f'{R1_DIR}/full'                       # r1 본실행 폴더(H·K 바이트 대조 기준, SHA256SUMS만 읽는다)
BASKET_DIR = sb.BASKET_DIR                       # r1 고정 명단을 그대로 쓴다
DESIGN = 'DESIGN-v1.md'
TOTAL_CASH = sb.CASH0                            # 1,000만 원
CORE = {'cash': 8e6, 'slot': 4e5}                # 보유 몫: 종목당 40만 원 한도
SPLIT_CASH = 2e6                                 # 분할 몫: 20종목 공용 200만 원(r1 X 실행기 --cash)
SMALL = {'cash': 2e6, 'slot': 1e5}               # 대조 C의 작은 보유 몫: 종목당 10만 원
SPLIT_RATIO = 0.022                              # 분할 몫 주문율(3월 승인 후보, 엔진 기본값 그대로)
WEIGHT_MIN = 0.80                                # 설계 4절: S 평균 투자 비중 80% 이상
WEIGHT_GAP = 0.10                                # 설계 4절: |S − H| ≤ 10%p
WEIGHT_EPS = 1e-12                               # 일평균 계산의 부동소수 오차(약 1e-16 × 거래일 수)를 경계 포함으로 흡수한다
REASON_ADDBUY, REASON_PROFIT = '추가 매수(하락)', '수익 실현'     # 엔진 reason_for_trade
N_JOBS = {'main': 108, 'ctrl': 108}
H_FILES = ('H_curve.csv', 'H_pos.csv', 'H_buys.csv', 'H_divs.csv', 'H_divskips.csv', 'H_meta.json', 'K_curve.csv', 'K_meta.json')
assert abs(CORE['cash'] + SPLIT_CASH - TOTAL_CASH) < 1e-9 and abs(CORE['cash'] + SMALL['cash'] - TOTAL_CASH) < 1e-9
assert abs(CORE['slot'] * sb.N_BASKET - CORE['cash']) < 1e-9 and abs(SMALL['slot'] * sb.N_BASKET - SMALL['cash']) < 1e-9


@contextlib.contextmanager
def account(cash, slot):
    """r1 장부·검사가 읽는 계좌 금액(same_basket_common.CASH0·SLOT_CASH)을 잠시 바꾼다. 끝나면 반드시 되돌린다."""
    old = sb.CASH0, sb.SLOT_CASH
    sb.CASH0, sb.SLOT_CASH = float(cash), float(slot)
    try: yield
    finally: sb.CASH0, sb.SLOT_CASH = old


# ---------------- 작업표 ----------------
def prefix(kind, year, k, div, delist):
    return f'{kind}_b{year}_k{k}_{div}_{delist}'


def rows_for(kind, dates, year, end=None):
    """한 바구니의 18가지(주 값 k=0·approx·freeze가 맨 앞). end가 없으면 개설 이듬해 마지막 거래일."""
    combos = sorted(((k, d, z) for k in sb.OFFSETS for d in sb.DIVS for z in sb.DELISTS), key=lambda c: c != sb.MAIN)
    e = sb.main_end(dates, year) if end is None else end
    return [(kind, year, k, d, z, '-', '-', sb.start_day(dates, year, k), e, prefix(kind, year, k, d, z)) for k, d, z in combos]


def build_jobs(dates):
    """정식 작업표: 주 108(S·H·K) + 대조 C 108(보유 몫 + 작은 몫). 조립기가 같은 함수로 기대 행을 다시 만든다."""
    rows = [r for kind in ('main', 'ctrl') for y in sb.YEARS for r in rows_for(kind, dates, y)]
    return sb.jobs_frame(rows)


def build_smoke_jobs(dates, year, end):
    """smoke: 한 바구니 k=0의 배당 3 × 상폐 2, 주·대조 모두 end에서 끝난다."""
    e = dates[dates <= pd.Timestamp(end)][-1]
    rows = [r for kind in ('main', 'ctrl') for r in rows_for(kind, dates, year, e) if r[2] == 0]
    return sb.jobs_frame(rows)


# ---------------- S 합산 ----------------
S_COLS = ['date', 'core_value', 'core_stock', 'core_cash', 'core_receivable', 'split_value', 'split_stock', 'split_cash', 'split_receivable',
          'value', 'stock_value', 'cash', 'receivable']


def combine(core_curve, split_curve, split_state):
    """S = 보유 몫 + 분할 몫(일별, 이체 없음). core_curve: r1 H 장부 곡선 열(value·cash·receivable·stock_value),
    split_curve: X 곡선(value·value_engine·receivable), split_state: X 상태(cash). 모두 같은 날짜 색인이어야 한다."""
    idx = core_curve.index
    if not (split_curve.index.equals(idx) and split_state.index.equals(idx)): raise ValueError('core and split dates differ')
    out = pd.DataFrame({'core_value': core_curve.value, 'core_stock': core_curve.stock_value, 'core_cash': core_curve.cash,
                        'core_receivable': core_curve.receivable, 'split_value': split_curve.value,
                        'split_stock': split_curve.value_engine - split_state.cash, 'split_cash': split_state.cash,
                        'split_receivable': split_curve.receivable}, index=idx)
    for col in ('value', 'stock', 'cash', 'receivable'):
        out[col if col != 'stock' else 'stock_value'] = out[f'core_{col}'] + out[f'split_{col}']
    out.index.name = 'date'
    return out[S_COLS[1:]]


def daily_weight(stock, value):
    """일별 투자 비중 = 주식평가액 ÷ 배당 미수금 포함 순자산(설계 4절)."""
    v = np.asarray(value, dtype=float)
    if (v <= 0).any() or not np.isfinite(v).all(): raise ValueError('non-positive or missing net asset value')
    return np.asarray(stock, dtype=float) / v


def weight_stats(s_stock, s_value, h_stock, h_value):
    """시작일~종료일(시작 직전 거래일 제외) 일별 비중의 평균과 S − H 일별 차이."""
    s, h = daily_weight(s_stock, s_value), daily_weight(h_stock, h_value)
    d = s - h
    return {'s_weight': float(s.mean()), 'h_weight': float(h.mean()), 'weight_gap': float(s.mean() - h.mean()),
            'daily_gap_mean': float(d.mean()), 'daily_gap_min': float(d.min()), 'daily_gap_max': float(d.max())}


def weight_ok(s_weight, h_weight):
    """설계 4절 합격선: S ≥ 80%이고 |S − H| ≤ 10%p(경계 포함). 값이 없으면 불합격.
    경계는 WEIGHT_EPS(1e-12)만큼 넓혀 비교한다: 매일 정확히 80%·90%인 장부도 평균 차이가 0.10000000000000009로 계산되기 때문이다.
    1e-12는 실제 비중 차이로는 의미 없는 크기라 바로 바깥 조건(예: 1e-9 초과)은 그대로 불합격이다."""
    if not (np.isfinite(s_weight) and np.isfinite(h_weight)): return False
    return bool(s_weight >= WEIGHT_MIN - WEIGHT_EPS and abs(s_weight - h_weight) <= WEIGHT_GAP + WEIGHT_EPS)


def split_trades(fills):
    """분할 몫 체결 집계: 추가매수 수, 익절 수(포지션 단위 체결 행), 매수한 종목 수."""
    if fills is None or len(fills) == 0: return {'split_addbuys': 0, 'split_takeprofits': 0, 'split_tickers_bought': 0}
    buys = fills[fills.type == 'BUY']
    return {'split_addbuys': int((buys.reason == REASON_ADDBUY).sum()), 'split_takeprofits': int(((fills.type == 'SELL') & (fills.reason == REASON_PROFIT)).sum()),
            'split_tickers_bought': int(buys.ticker.nunique())}


def split_effect_held(trades):
    """설계 4절: 분할 몫의 추가매수 또는 익절이 0건이면 '분할 효과 보류'."""
    return trades['split_addbuys'] == 0 or trades['split_takeprofits'] == 0
