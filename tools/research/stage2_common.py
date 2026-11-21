
"""2단계 비교(시장 ETF·종목 선정만·현행 분할)의 공통 자료와 규칙. DB는 읽기만 한다."""
import mysql.connector, pandas as pd, numpy as np, yaml

def connect(cfg_path='config/config.yaml'):
    db = yaml.safe_load(open(cfg_path, encoding='utf-8'))['database']
    return mysql.connector.connect(host=db['host'], port=int(db.get('port', 3306)), user=db['user'], password=db['password'],
                                   database=db['database'], use_pure=True)

def q(conn, sql, params=None):
    cur = conn.cursor(); cur.execute(sql, params or ()); cols = [c[0] for c in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols); cur.close(); return df

def trading_dates(conn):
    d = q(conn, "SELECT DISTINCT date FROM DailyStockPrice WHERE date >= '2013-01-01' ORDER BY date")
    return pd.DatetimeIndex(pd.to_datetime(d.date))

# 증권거래세(코스피는 거래세+농특세 0.15%, 코스닥 단일 세율, 합계 같음). 결제일(T+2 거래일) 기준.
TAX_STEPS = [('2019-06-03', 0.0030), ('2021-01-01', 0.0025), ('2023-01-01', 0.0023), ('2024-01-01', 0.0020),
             ('2025-01-01', 0.0018), ('2026-01-01', 0.0015)]
def tax_by_settlement(settle):
    s = pd.Timestamp(settle)
    for until, rate in TAX_STEPS:
        if s < pd.Timestamp(until): return rate
    return 0.0020

class Calendar:
    def __init__(self, dates):
        self.dates = pd.DatetimeIndex(dates)
    def shift(self, d, n):
        i = self.dates.searchsorted(pd.Timestamp(d))
        return self.dates[min(i + n, len(self.dates) - 1)]
    def sell_tax(self, trade_date):
        return tax_by_settlement(self.shift(trade_date, 2))
    def record_date(self, year):
        y = self.dates[self.dates.year == year]
        return y[-2]          # 마지막 거래일 전날 종가 보유자 = 연말 배당 권리자(근사)
    def pay_date(self, year):
        m = self.dates[(self.dates.year == year + 1) & (self.dates.month == 5)]
        return m[0] if len(m) else None

def dps_table(conn, tickers, years):
    """FY y 주당배당금 = y+1년 4/25~5/15 사이 마지막 FinancialData.dps. 없으면(2025 등) 직전 연도 값으로 대체하고 표시."""
    t = list(tickers); out = []
    if not t: return pd.DataFrame(columns=['ticker', 'fy', 'dps', 'dps_fallback'])
    ph = ','.join(['%s'] * len(t))
    for y in years:
        a, b = f'{y+1}-04-25', f'{y+1}-05-15'
        d = q(conn, f"SELECT stock_code, date, dps FROM FinancialData WHERE stock_code IN ({ph}) AND date BETWEEN %s AND %s", t + [a, b])
        d = d.sort_values('date').groupby('stock_code').last().reset_index()
        for r in d.itertuples(): out.append((r.stock_code, y, float(r.dps or 0), False))
    df = pd.DataFrame(out, columns=['ticker', 'fy', 'dps', 'dps_fallback'])
    have = set(zip(df.ticker, df.fy)); add = []
    for y in years:
        for tk in t:
            if (tk, y) not in have:
                prev = df[(df.ticker == tk) & (df.fy == y - 1)]
                if len(prev): add.append((tk, y, float(prev.dps.iloc[0]), True))
    if add: df = pd.concat([df, pd.DataFrame(add, columns=df.columns)], ignore_index=True)
    return df

def adj_ratio_on(conn, tickers, day):
    t = list(tickers)
    if not t: return {}
    ph = ','.join(['%s'] * len(t))
    d = q(conn, f"SELECT stock_code, date, adj_ratio, close_price FROM DailyStockPrice WHERE stock_code IN ({ph}) AND date <= %s AND date >= %s",
          t + [str(pd.Timestamp(day).date()), str((pd.Timestamp(day) - pd.Timedelta(days=40)).date())])
    d = d.sort_values('date').groupby('stock_code').last()
    return {k: (float(v.adj_ratio) if v.adj_ratio is not None else np.nan, float(v.close_price)) for k, v in d.iterrows()}
