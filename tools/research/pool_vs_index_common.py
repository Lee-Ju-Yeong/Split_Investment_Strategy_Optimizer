"""연구 (b) 1등급 후보군 대 시장·스타일 지수: 공통 상수·기간·조회 상한·지표.

설계: runs/tier1-pool-vs-index-r1/DESIGN-v2.md. DB는 읽기만 한다(SELECT). 가격·배당 조회는 END_CAP 이하로 강제한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc

END_CAP = '2025-12-30'                 # 가격·배당 조회 상한(설계 1절)
CALENDAR_TAIL_DAYS = 15                # 결제일 계산용 이후 거래일(날짜만)
DATA_START = '2013-12-01'
PRICE_LEAD_DAYS = 30                   # 첫 신호일 앞 가격 조회 여유(장부 직전값용)
CASH0 = 1e7
COMMISSION = 0.00015
LIQ_THRESHOLD_U = 1_000_000_000        # 1등급 출발 문턱(daily_stock_tier_batch prime_liquidity)
LIQ_THRESHOLD_U0 = 0
MARKETS = ('KOSPI', 'KOSDAQ')
N_TERCILES = 3
S20_N, S20_WEIGHT = 20, 0.05
CW_TOP = {'KOSPI': 200, 'KOSDAQ': 150}
DPS_WINDOW = ('04-25', '05-15')        # 이듬해 이 기간의 마지막 관측 DPS(stage2_common.dps_table과 같은 창)
PAY_AFTER = {'observed': '05-15', 'late': '06-15'}   # 이 날 다음 첫 거래일에 현금 지급
DPS_FLAG_YIELD = 0.3                   # 주 분석은 제외하지 않고 관측 창 뒤 품질표에 표시만(과거 S20 재현만 이 값으로 제외)
SEGMENTS = [(2014, 2015), (2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023)]
MAIN_PERIOD, EXT_PERIOD, FULL_PERIOD = (2014, 2023), (2024, 2025), (2014, 2025)
TIE_EPS_PP = 1e-10                     # 동률(%p)
EDGE_PP = 1.0                          # 판정 문턱(%p, 포함)
MIN_SAME_SIGN = 3
STYLE_ETFS = ('229200', '226980', '252650', '277650')
KODEX200 = '069500'
E0 = f'E_{KODEX200}'                   # 결과표의 KODEX 200 계좌 이름(F1: 한 가지로 고정)
TICK_UNIFIED_FROM = '2023-01-25'       # 코스피·코스닥 호가 단위 통일일
TICK_TABLES = {                        # (가격 상한, 호가) 오름차순. 마지막은 상한 없음
    'KOSPI_OLD': ((1000, 1), (5000, 5), (10000, 10), (50000, 50), (100000, 100), (500000, 500), (None, 1000)),
    'KOSDAQ_OLD': ((1000, 1), (5000, 5), (10000, 10), (50000, 50), (None, 100)),
    'UNIFIED': ((2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500), (None, 1000))}
REF_CORR_MIN, REF_ANNUAL_DIFF_MAX_PP = 0.97, 3.0
REF_MIN_YEARS, REF_MIN_MONTHS = 5, 60


def cap_date(cap=END_CAP):
    c = pd.Timestamp(cap)
    if c > pd.Timestamp(END_CAP):
        raise ValueError(f'cap {c.date()} exceeds END_CAP {END_CAP}')
    return c


def check_window(start, end, cap):
    """start ≤ end ≤ cap ≤ END_CAP을 조회 전에 강제한다(F4)."""
    s, e, c = pd.Timestamp(start), pd.Timestamp(end), cap_date(cap)
    if not (s <= e <= c): raise ValueError(f'window must satisfy start<=end<=cap<=END_CAP: {s.date()} {e.date()} {c.date()}')
    return s, e, c


def tick_size(price, market, day):
    """당시 시장·날짜의 호가 단위(F7). 2023-01-25 이전은 시장별 표."""
    key = 'UNIFIED' if pd.Timestamp(day) >= pd.Timestamp(TICK_UNIFIED_FROM) else f'{market}_OLD'
    for upper, tick in TICK_TABLES[key]:
        if upper is None or price < upper: return tick


def tick_up(price, market, day):
    t = tick_size(price, market, day)
    return float(np.ceil(round(price / t, 5)) * t)


def capped_query(conn, sql, params, cap):
    """date 열이 있는 결과는 상한 이하만 허용한다(SQL에도 상한을 넣고, 결과로 다시 확인)."""
    df = sc.q(conn, sql, params)
    if 'date' in df.columns and len(df) and pd.to_datetime(df['date']).max() > cap_date(cap):
        raise RuntimeError('query returned rows after cap')
    return df


def trading_calendar(conn, cap):
    """cap 이하 거래일과, 결제일 계산용 이후 거래일(날짜만)."""
    tail = (cap_date(cap) + pd.Timedelta(days=CALENDAR_TAIL_DAYS)).strftime('%Y-%m-%d')
    d = sc.q(conn, "SELECT DISTINCT date FROM DailyStockPrice WHERE date >= '2013-01-01' AND date <= %s ORDER BY date", [tail])
    allx = pd.DatetimeIndex(pd.to_datetime(d.date))
    return allx[allx <= cap_date(cap)], allx


def rebalance_days(dates, k, start='2014-01-01'):
    """달마다 (k+1)번째 거래일(교체일)과 그 직전 거래일(신호일)."""
    s = pd.Series(dates, index=dates)
    s = s[s >= pd.Timestamp(start)]
    out = []
    for _, g in s.groupby([s.index.year, s.index.month]):
        if len(g) > k:
            t = g.iloc[k]; out.append((t, dates[dates.searchsorted(t) - 1]))
    return out


def tax_rate(full_calendar, trade_date):
    i = full_calendar.searchsorted(pd.Timestamp(trade_date))
    return sc.tax_by_settlement(full_calendar[min(i + 2, len(full_calendar) - 1)])


def record_date(dates, year):
    y = dates[dates.year == year]
    return y[-2] if len(y) >= 2 else None    # 마지막 거래일 전날 보유 = 권리 수량(근사)


def pay_date(dates, year, mode):
    after = pd.Timestamp(f'{year + 1}-{PAY_AFTER[mode]}')
    nxt = dates[dates > after]
    return nxt[0] if len(nxt) else None


def segment_bounds(dates, years, base_date):
    """구간 직전 거래일 종가(계좌 기준일보다 앞이면 기준일) → 구간 말 종가."""
    y0, y1 = years
    prior = dates[dates < pd.Timestamp(f'{y0}-01-01')]
    base = max(prior[-1], base_date) if len(prior) else base_date
    end = dates[dates <= pd.Timestamp(f'{y1}-12-31')][-1]
    return base, end


def annualized(v0, v1, d0, d1):
    yrs = (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25
    return ((v1 / v0) ** (1 / yrs) - 1) * 100 if v0 > 0 and v1 > 0 and yrs > 0 else np.nan


def max_drawdown(v):
    v = np.asarray(v, dtype=float)
    return float((v / np.maximum.accumulate(v) - 1).min() * 100) if len(v) else np.nan


def period_metrics(curve, base, end, dd_mask=None):
    """curve: 날짜 색인 Series(기준일 포함). dd_mask: 낙폭 계산에 쓸 날(True)."""
    c = curve[(curve.index >= base) & (curve.index <= end)]
    dd = c if dd_mask is None else c[dd_mask.reindex(c.index).fillna(False).values | (c.index == base)]
    cagr = annualized(c.iloc[0], c.iloc[-1], c.index[0], c.index[-1])
    mdd = max_drawdown(dd.dropna().values)      # ETF 계산용 곡선은 무효 날이 NaN(F5)
    return {'base': str(c.index[0].date()), 'end': str(c.index[-1].date()), 'cagr': cagr, 'mdd': mdd,
            'calmar': cagr / abs(mdd) if mdd < 0 else np.nan, 'v0': float(c.iloc[0]), 'v1': float(c.iloc[-1])}


# ---------------- 출력 구조(G1: 절단 비교기가 기대하는 고정 열·파일 목록) ----------------
ACCOUNTS = ('P1', 'P1_KOSPI', 'P1_KOSDAQ', 'M', 'M0', 'U_EW', 'U_KOSPI_EW', 'U_KOSDAQ_EW', 'U_small_EW',
            'U_KOSPI_CW200', 'U_KOSDAQ_CW150', 'S20e', 'S20')
DIV_MODES, DELIST_MODES, COST_MODES = ('none', 'observed', 'late'), ('last', 'zero'), ('net', 'gross')
ETF_CODES = (KODEX200,) + STYLE_ETFS
ETF_SUFFIXES = ('open', 'net', 'net_display', 'ok', 'entered', 'gross', 'gross_display')
KEY4 = ['account', 'div', 'delist', 'cost']
TABLE_SCHEMAS = {
    'orders.csv': KEY4 + ['date', 'total_at_open', 'sell_gross', 'sell_fee', 'sell_tax', 'buy_gross', 'buy_fee', 'buy_scale', 'n_target',
                          'blocked_target_weight', 'blocked_unbought_cash', 'n_frozen', 'frozen_book', 'turnover_one_way', 'turnover_two_way',
                          'cash_after', 'conservation_err'],
    'fills.parquet': KEY4 + ['date', 'ticker', 'sell_qty', 'buy_qty', 'sell_value', 'buy_value'],
    'exits.csv': KEY4 + ['date', 'when', 'ticker', 'generation', 'qty', 'book_written'],
    'pays.csv': KEY4 + ['fy', 'date', 'amount', 'holders', 'holders_observed_positive', 'holders_observed_zero', 'holders_unobserved',
                        'rights_qty_unobserved'],
    'rights.csv': KEY4 + ['fy', 'date', 'holders', 'rights_qty_total', 'n_ratio_fallback', 'n_rights_undetermined', 'undetermined_qty',
                          'holders_price_stale'],
    'rights_detail.csv': KEY4 + ['fy', 'date', 'ticker', 'qty', 'basis', 'ratio_used', 'last_valid_ratio_date', 'undetermined_raw_qty_est'],
    'marks.csv': KEY4 + ['date', 'n_held', 'n_frozen', 'frozen_qty', 'frozen_book', 'frozen_disposable', 'frozen_weight', 'n_dead', 'dead_qty',
                         'dead_book', 'dead_disposable', 'dead_weight', 'stale_close_held', 'stale_held_days_cum', 'held_without_book_days_cum',
                         'holdings', 'cash'],
    'mark_detail.csv': KEY4 + ['date', 'kind', 'ticker', 'generation', 'qty', 'book', 'disposable'],
    'dps_quality.csv': ['fy', 'window_end', 'ticker', 'dps', 'record_date', 'record_raw_close', 'price_basis', 'yield', 'flag_gt30',
                        'P1_rights_qty', 'M_rights_qty'],
    'integer_orders.csv': ['date', 'total_at_open', 'sell_gross', 'buy_gross', 'buy_scale', 'vacated_slots', 'budget_short', 'n_target',
                           'open_changed_by_tick', 'n_adj_open_missing_but_raw_tradable', 'cash_after'],
    'integer_fills.csv': ['date', 'ticker', 'sell_qty', 'buy_qty', 'sell_value', 'buy_value'],
    'integer_exits.csv': ['date', 'when', 'ticker', 'generation', 'qty', 'book_written'],
    'integer_ca.csv': ['date', 'ticker', 'factor', 'fraction', 'fraction_cash'],
    'integer_pays.csv': ['fy', 'date', 'amount', 'holders', 'holders_observed_positive', 'holders_observed_zero', 'holders_unobserved',
                         'rights_qty_unobserved'],
    'months.csv': ['trade_date', 'signal_date', 'status', 'status_reason', 'tier_source', 'raw_candidates', 'ranked', 'panel_rows']}
TABLE_DATE_COL = {n: ('trade_date' if n == 'months.csv' else 'window_end' if n == 'dps_quality.csv' else 'date') for n in TABLE_SCHEMAS}
TABLE_KEYS = {'orders.csv': KEY4 + ['date'], 'fills.parquet': KEY4 + ['date', 'ticker'], 'exits.csv': KEY4 + ['date', 'ticker', 'generation'],
              'pays.csv': KEY4 + ['fy'], 'rights.csv': KEY4 + ['fy'], 'rights_detail.csv': KEY4 + ['fy', 'ticker'], 'marks.csv': KEY4 + ['date'],
              'mark_detail.csv': KEY4 + ['date', 'kind', 'ticker', 'generation'], 'dps_quality.csv': ['fy', 'ticker'], 'integer_orders.csv': ['date'],
              'integer_fills.csv': ['date', 'ticker'], 'integer_exits.csv': ['date', 'ticker', 'generation'], 'integer_ca.csv': ['date', 'ticker'],
              'integer_pays.csv': ['fy'], 'months.csv': ['trade_date']}


def frame_columns(name):
    """고정 열 목록: curves/open_values는 13계좌 × 12조합, ETF는 5종 × 7열, 정수 계좌는 P1_int."""
    if name in ('curves.parquet', 'open_values.parquet'):
        return [f'{a}|{d}|{x}|{c}' for d in DIV_MODES for x in DELIST_MODES for c in COST_MODES for a in ACCOUNTS]
    if name == 'etf_curves.parquet': return [f'E_{e}|{s}' for e in ETF_CODES for s in ETF_SUFFIXES]
    if name == 'integer_curve.parquet': return ['P1_int']
    raise KeyError(name)


FRAME_FILES = ('curves.parquet', 'open_values.parquet', 'etf_curves.parquet', 'integer_curve.parquet')


def table_frame(rows, name):
    """기록 목록을 고정 열로(0건이어도 열 머리는 남긴다)."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    return df.reindex(columns=TABLE_SCHEMAS[name])


FILL_TOL_WON = 1e-6          # H2: 계좌·조합·날짜별 체결 금액 합과 주문 금액의 허용 차


def fills_consistency(orders, fills, keys):
    """H2: 양수 매수·매도 금액이 있는 주문(keys별)에는 체결 행이 있어야 하고, 체결 금액 합 = 주문 금액(1e-6원).
    체결만 있고 주문 금액이 0인 경우도 실패. 반환: (문제 예시 최대 20개, 문제 수)."""
    num = lambda df, cols: df[keys + cols].astype({c: float for c in cols}).assign(**{k: df[k].astype(str) for k in keys})
    o = num(orders, ['buy_gross', 'sell_gross']).groupby(keys, as_index=False).sum()
    f = num(fills, ['buy_value', 'sell_value']).groupby(keys, as_index=False).sum()
    j = o.merge(f, on=keys, how='outer').fillna(0.0)
    bad = j[((j.buy_gross - j.buy_value).abs() > FILL_TOL_WON) | ((j.sell_gross - j.sell_value).abs() > FILL_TOL_WON)]
    return bad.head(20).to_dict('records'), int(len(bad))
