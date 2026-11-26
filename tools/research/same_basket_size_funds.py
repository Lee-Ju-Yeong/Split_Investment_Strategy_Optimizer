"""주문율 격자(same-basket-size-r1) 자금 운용 지표와 기준·경계 구간(설계 3절). 순수 함수이며 DB·파일을 읽지 않는다.

지표(시작일~종료일의 거래일 종가, 시작 전날 제외):
- avg_invested: 주식평가액 ÷ 배당 미수금 포함 순자산의 평균. 주식평가액 = 엔진 평가액 − 현금, 순자산 = 곡선 value.
- short_days_ratio: 마감 현금 < 그날 적용 중인 그달 한 번 주문액(엔진 investment_per_order)인 날의 비율.
- add_buy_share: 추가매수('추가 매수(하락)') 체결액 ÷ 전체 매수 체결액(체결액 = 현금 감소액, 수수료 포함). 매수가 없으면 NaN.
- n_profit_sells: 익절('수익 실현') 체결 수.
- first_day_invested, min_cash_share(현금 ÷ 순자산의 최솟값), n_bought(체결 종목 수), n_never_bought(명단 중 한 주도 못 산 종목 수).
기준(반올림 전 값): 평균 비중 ≥ 90%, 여력 부족일 ≤ 20%, 분할 활동(추가매수 비중 ≥ 20% 이고 익절 ≥ 1건).
비중·여력은 통과했지만 분할 활동이 미달이면 '분할 효과 보류'.
경계 구간: 기본 조건에서 여섯 바구니가 세 기준을 함께 통과한 주문율이 격자 순서로 세 값 이상 연속인 구간(둘 이상이면 모두).
"""
import math
import numpy as np, pandas as pd

MIN_INVESTED = 0.90
MAX_SHORT_DAYS = 0.20
MIN_ADD_SHARE = 0.20
MIN_PROFIT_SELLS = 1
MIN_RUN = 3
RATIO_EPS = 1e-12  # 일평균과 비율의 부동소수 계산 오차
REASON_ADD, REASON_NEW, REASON_PROFIT = '추가 매수(하락)', '신규 진입', '수익 실현'
SPLIT_HOLD = '분할 효과 보류'
METRIC_KEYS = ('avg_invested', 'short_days_ratio', 'add_buy_share', 'n_profit_sells', 'first_day_invested', 'min_cash_share', 'n_bought', 'n_never_bought', 'n_days')


def order_amount_by_day(orderlog, days):
    """그날 적용 중인 주문액. 기록이 없는 날이 있으면 KeyError(엔진은 매일 매도 신호 생성에서 계산한다)."""
    s = pd.Series(orderlog.investment_per_order.to_numpy(float), index=pd.to_datetime(orderlog.date))
    missing = days.difference(s.index)
    if len(missing): raise KeyError(f'order amount not logged on {len(missing)} days')
    return s.reindex(days)


def fund_metrics(curve, state, fills, orderlog, basket):
    """curve·state: 시작 전날 행을 포함한 날짜 색인 표. fills: date·ticker·type·dcash·reason. 반환 dict는 METRIC_KEYS."""
    c, st = curve.iloc[1:], state.iloc[1:]
    stock = c.value_engine - st.cash
    ipo = order_amount_by_day(orderlog, c.index)
    buys = fills[fills.type == 'BUY'] if len(fills) else fills
    spent = float(-buys.dcash.sum()) if len(buys) else 0.0
    add = float(-buys[buys.reason == REASON_ADD].dcash.sum()) if len(buys) else 0.0
    bought = set(buys.ticker) if len(buys) else set()
    profit = int((fills.type == 'SELL').mul(fills.reason == REASON_PROFIT).sum()) if len(fills) else 0
    return {'avg_invested': float((stock / c.value).mean()), 'short_days_ratio': float((st.cash < ipo).mean()),
            'add_buy_share': add / spent if spent > 0 else float('nan'), 'n_profit_sells': profit,
            'first_day_invested': float(stock.iloc[0] / c.value.iloc[0]), 'min_cash_share': float((st.cash / c.value).min()),
            'n_bought': len(bought & set(basket)), 'n_never_bought': len(set(basket) - bought), 'n_days': int(len(c))}


def ge(x, bound):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x) and x >= bound


def le(x, bound):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x) and x <= bound


def criteria(m):
    """세 기준과 표시. 값이 없거나 유한하지 않으면 그 기준은 미달."""
    invest = ge(m.get('avg_invested'), MIN_INVESTED - RATIO_EPS)
    capacity = le(m.get('short_days_ratio'), MAX_SHORT_DAYS + RATIO_EPS)
    split = ge(m.get('add_buy_share'), MIN_ADD_SHARE - RATIO_EPS) and ge(m.get('n_profit_sells'), MIN_PROFIT_SELLS)
    label = '' if invest and capacity and split else (SPLIT_HOLD if invest and capacity else '기준 미달')
    return {'invest_ok': invest, 'capacity_ok': capacity, 'split_ok': split, 'all_ok': invest and capacity and split, 'label': label}


def ratio_passes(base_ok, ratios, baskets):
    """base_ok: {(주문율, 바구니): bool}. 주문율마다 여섯 바구니가 모두 통과했는지(값이 없으면 미통과)."""
    return [(r, all(base_ok.get((r, y)) is True for y in baskets)) for r in ratios]


def boundary_runs(passes, min_run=MIN_RUN):
    """passes: [(주문율, 통과)] 격자 순서. 통과가 min_run개 이상 연속인 구간들(각 구간은 주문율 목록)."""
    runs, cur = [], []
    for r, ok in passes:
        if ok: cur.append(r)
        else:
            if len(cur) >= min_run: runs.append(cur)
            cur = []
    if len(cur) >= min_run: runs.append(cur)
    return runs
