"""연구 (b) 자료 묶음: 교체 달력, 비교 모집단 U/U0(신호일 스냅숏 한 장 + 최신 등급 행 + 신호일 시총), 가격, 관측 DPS, 스냅숏 퇴출 사건.

usage: python -B tools/research/pool_vs_index_data.py PANEL.csv OUT_DIR --offset K [--cap 2025-12-30] [--start 2014-01-01] [--end CAP]
3월 실행 체크아웃(wt-79051b5)을 작업 디렉터리로 실행(config/config.yaml의 DB 접속). DB는 SELECT만 한다.
PANEL.months.csv(pool_vs_index_panel.py 산출)가 PANEL.csv 옆에 있어야 한다. 가격·배당·스냅숏·등급·시총은 모두 --cap 이하만 읽는다.
산출(OUT_DIR, 새 폴더만): months.csv, universe.parquet, prices.parquet, dps.csv, exits.csv, calendar.csv, data_checks.json, bundle.sha256
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import pool_vs_index_common as pc

PRICE_COLS = 'stock_code, date, open_price, close_price, adj_open, adj_close, adj_ratio, volume'


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('panel'); ap.add_argument('outdir')
    ap.add_argument('--offset', type=int, required=True)
    ap.add_argument('--cap', default=pc.END_CAP)
    ap.add_argument('--start', default='2014-01-01'); ap.add_argument('--end', default=None)
    return ap


def load_panel(path):
    p = pd.read_csv(path, dtype={'ticker': str}, parse_dates=['trade_date', 'signal_date'])
    m = pd.read_csv(path.replace('.csv', '') + '.months.csv', parse_dates=['trade_date', 'signal_date'])
    return p, m


USABLE_STATUS = ('ok', 'zero_candidates')


def month_calendar(dates, k, start, end, pmonths, panel):
    """교체일·신호일을 달력에서 다시 만들고 패널 달력(status)과 맞춘다(F3).
    패널 달력에 없는 달은 missing, ok인데 패널 행 수가 ranked와 다르면 panel_mismatch, zero_candidates인데 행이 있으면 panel_mismatch."""
    days = pd.DataFrame([(t, s) for t, s in pc.rebalance_days(dates, k, start) if t <= end], columns=['trade_date', 'signal_date'])
    m = days.merge(pmonths, on=['trade_date', 'signal_date'], how='left')
    m['status'] = m.status.fillna('missing'); m['status_reason'] = np.where(m.status == 'missing', 'month absent from panel calendar', '')
    rows = panel.groupby('trade_date').size()
    m['panel_rows'] = m.trade_date.map(rows).fillna(0).astype(int)
    groups = {d: g for d, g in panel.groupby('trade_date')}
    for idx, r in m.iterrows():
        reason = panel_month_problem(r, groups.get(r.trade_date))
        if reason: m.loc[idx, 'status'], m.loc[idx, 'status_reason'] = 'panel_mismatch', reason
    extra = sorted(set(panel.trade_date) - set(m.trade_date))
    if extra: raise RuntimeError(f'panel has trade dates outside the calendar: {extra[:5]}')
    return m


def panel_month_problem(r, g):
    """G6: 달력 상태와 패널 행이 맞지 않으면 사유. ok인데 ranked·행 0, 행 수 ≠ ranked, 신호일 불일치, 순위·종목 중복, 순위가 1..n이 아님."""
    n = 0 if g is None else len(g)
    if r.status == 'zero_candidates':      # H3: 정상 0건은 원후보·순위·패널 행이 모두 0
        if n or r.raw_candidates != 0 or r.ranked != 0: return f'zero_candidates inconsistent: raw={r.raw_candidates} ranked={r.ranked} rows={n}'
        return ''
    if r.status != 'ok': return ''
    if not r.ranked or n == 0: return 'ok with zero ranked rows'
    if n != r.ranked: return f'panel rows {n} != ranked {int(r.ranked)}'
    if (pd.to_datetime(g.signal_date) != r.signal_date).any(): return 'panel signal_date differs from calendar'
    if g.ticker.duplicated().any(): return 'duplicated ticker'
    rk = pd.to_numeric(g['rank'], errors='coerce')
    if not (np.isfinite(rk).all() and (rk == np.floor(rk)).all()): return 'rank not finite integer'   # H3: 변환 전에 확인
    if sorted(rk.astype(int)) != list(range(1, n + 1)): return 'rank not unique 1..n'
    return ''


def snapshot_members(conn, cap):
    s = pc.capped_query(conn, "SELECT snapshot_date AS date, stock_code, market_type FROM TickerUniverseSnapshot "
                              "WHERE snapshot_date >= '2013-01-01' AND snapshot_date <= %s", [str(cap.date())], cap)
    s['date'] = pd.to_datetime(s.date)
    return s


def latest_tier(conn, sd, codes):
    if not codes: raise ValueError('latest_tier called with empty code list')   # 빈 IN () SQL을 보내지 않는다(G6)
    ph = ','.join(['%s'] * len(codes))
    sql = (f"SELECT t.stock_code, t.date AS tier_date, t.tier, t.liquidity_20d_avg_value AS liq FROM DailyStockTier t JOIN "
           f"(SELECT stock_code, MAX(date) md FROM DailyStockTier WHERE date <= %s AND stock_code IN ({ph}) GROUP BY stock_code) m "
           f"ON t.stock_code = m.stock_code AND t.date = m.md")
    d = sc.q(conn, sql, [str(sd.date())] + codes)
    d['tier_date'] = pd.to_datetime(d.tier_date)
    return d


def signal_mcap(conn, sd, codes):
    if not codes: raise ValueError('signal_mcap called with empty code list')
    ph = ','.join(['%s'] * len(codes))
    return sc.q(conn, f"SELECT stock_code, market_cap FROM MarketCapDaily WHERE date = %s AND stock_code IN ({ph})", [str(sd.date())] + codes)


def tercile(df):
    """시장 안 시총 오름차순(동률은 코드 순)으로 3칸. 나머지는 작은 칸부터 1개씩(7 → 3·2·2)."""
    out = pd.Series(-1, index=df.index)
    for _, g in df.groupby('market'):
        g = g.sort_values(['market_cap', 'stock_code'])
        for c, part in enumerate(np.array_split(g.index.values, pc.N_TERCILES)): out.loc[part] = c
    return out


UNIVERSE_COLUMNS = ['stock_code', 'market', 'tier_date', 'tier', 'liq', 'market_cap', 'in_U0', 'in_U', 'tercile_U', 'tercile_U0', 'p1_rank',
                    'trade_date', 'signal_date', 'snapshot_date']


def month_universe(conn, snaps, row, panel_month):
    """반환 (U 표, P1 중 스냅숏에 없는 코드, 목록 없음 여부). 스냅숏이 없거나 두 시장 종목이 0이면 SQL 없이 빈 표(G6)."""
    sd = row.signal_date
    sdates = snaps.date[snaps.date <= sd]
    snap_date = sdates.max() if len(sdates) else pd.NaT
    mem = snaps[(snaps.date == snap_date) & snaps.market_type.isin(pc.MARKETS)][['stock_code', 'market_type']]
    codes = list(mem.stock_code)
    rk = panel_month.set_index('ticker')['rank'] if panel_month is not None else pd.Series(dtype=int)
    if not codes: return pd.DataFrame(columns=UNIVERSE_COLUMNS), sorted(rk.index), True
    u = mem.rename(columns={'market_type': 'market'}).merge(latest_tier(conn, sd, codes), on='stock_code', how='left')
    u = u.merge(signal_mcap(conn, sd, codes), on='stock_code', how='left')
    u['market_cap'] = pd.to_numeric(u.market_cap, errors='coerce')
    u['liq'] = pd.to_numeric(u.liq, errors='coerce')
    u['in_U0'] = u.liq.notna() & (u.liq >= pc.LIQ_THRESHOLD_U0) & (u.market_cap > 0)
    u['in_U'] = u.in_U0 & (u.liq >= pc.LIQ_THRESHOLD_U)
    u['tercile_U'] = tercile(u[u.in_U]).reindex(u.index).fillna(-1).astype(int)
    u['tercile_U0'] = tercile(u[u.in_U0]).reindex(u.index).fillna(-1).astype(int)
    u['p1_rank'] = u.stock_code.map(rk).fillna(0).astype(int)
    u['trade_date'], u['signal_date'], u['snapshot_date'] = row.trade_date, sd, snap_date
    missing = sorted(set(rk.index) - set(u.stock_code))
    return u[UNIVERSE_COLUMNS], missing, False


def build_universe(conn, snaps, months, panel):
    """빈 스냅숏 달은 months.status를 missing_universe로 바꾸고 사유를 남긴다(G6)."""
    groups = {d: g for d, g in panel.groupby('trade_date')}
    parts, miss = [], {}
    for idx, row in zip(months.index, months.itertuples()):
        pm = groups.get(row.trade_date) if row.status in USABLE_STATUS else None   # 이미 오류로 확인한 달의 순위는 매핑하지 않는다
        u, m, empty = month_universe(conn, snaps, row, pm)
        if empty: months.loc[idx, 'status'], months.loc[idx, 'status_reason'] = 'missing_universe', 'empty snapshot membership at signal date'
        parts.append(u)
        if m: miss[str(row.trade_date.date())] = m
        print('universe', row.trade_date.date(), int(u.in_U.sum()), int(u.in_U0.sum()), int((u.p1_rank > 0).sum()), flush=True)
    return pd.concat(parts, ignore_index=True), miss


def load_prices(conn, tickers, first, end, cap):
    """first~end 가격. check_window로 end ≤ cap ≤ END_CAP이 먼저 강제되고, 결과는 cap으로 다시 확인한다(F4)."""
    pc.check_window(first, end, cap); parts = []
    for y in range(first.year, end.year + 1):
        a = max(first, pd.Timestamp(f'{y}-01-01')); b = min(end, pd.Timestamp(f'{y}-12-31'))
        d = pc.capped_query(conn, f"SELECT {PRICE_COLS} FROM DailyStockPrice WHERE date BETWEEN %s AND %s",
                            [str(a.date()), str(b.date())], cap)
        parts.append(d[d.stock_code.isin(tickers)])
    px = pd.concat(parts, ignore_index=True)
    px['date'] = pd.to_datetime(px.date)
    for c in px.columns[2:]: px[c] = pd.to_numeric(px[c], errors='coerce')
    return px


def require_tickers(tickers, outdir, checks):
    """전체 조회 종목이 비면 SQL 전에 사유를 data_checks.json에 남기고 중단한다(빈 IN () 방지)."""
    if tickers: return
    checks['fatal'] = 'no tickers in any month (all universes empty); price/DPS SQL not sent'
    json.dump(checks, open(os.path.join(outdir, 'data_checks.json'), 'w'), indent=1, default=str)
    raise SystemExit(checks['fatal'])


def load_dps(conn, tickers, cap):
    """FY y DPS = y+1년 4/25~5/15의 마지막 유효(유한한 비음수) 값. 창 끝이 cap 뒤인 해는 읽지 않는다(미관측 → 0원)."""
    t = list(tickers); ph = ','.join(['%s'] * len(t)); out = []
    for y in range(2013, cap.year):
        a, b = pd.Timestamp(f'{y + 1}-{pc.DPS_WINDOW[0]}'), pd.Timestamp(f'{y + 1}-{pc.DPS_WINDOW[1]}')
        if b > cap: continue
        d = pc.capped_query(conn, f"SELECT stock_code, date, dps FROM FinancialData WHERE stock_code IN ({ph}) AND date BETWEEN %s AND %s "
                                  f"AND dps IS NOT NULL AND dps >= 0", t + [str(a.date()), str(b.date())], cap)
        d = d[np.isfinite(pd.to_numeric(d.dps, errors='coerce'))].sort_values('date').groupby('stock_code').last().reset_index()
        out += [(r.stock_code, y, float(r.dps), str(r.date)) for r in d.itertuples()]
    return pd.DataFrame(out, columns=['ticker', 'fy', 'dps', 'observed_date'])


def exit_events(snaps, tickers):
    """두 시장(코스피·코스닥) 합친 스냅숏을 시간순으로 비교: 직전 스냅숏에 있고 이번에 없으면 exit, 이전에 있었다 다시 나오면 reentry.
    스냅숏 날이 거래일인지는 sim이 판단한다(거래일: 주문 뒤, 휴장일: 다음 거래일 주문 전, F6)."""
    snaps = snaps[snaps.market_type.isin(pc.MARKETS)]
    dates = np.sort(snaps.date.unique())
    pres = snaps[snaps.stock_code.isin(tickers)].assign(v=1).pivot_table(index='date', columns='stock_code', values='v', aggfunc='max')
    pres = pres.reindex(dates).fillna(0).astype(int)
    ch = pres.diff().iloc[1:]
    seen = pres.cummax().shift(1).iloc[1:]            # 이전에 한 번이라도 있었는지(처음 등장은 listing)
    ev = ch.stack()
    ev = ev[ev != 0].reset_index()
    ev.columns = ['snapshot_date', 'ticker', 'change']
    prior = seen.stack().reindex(pd.MultiIndex.from_frame(ev[['snapshot_date', 'ticker']])).fillna(0).values > 0
    ev['event'] = np.where(ev.change < 0, 'exit', np.where(prior, 'reentry', 'listing'))
    return ev[['ticker', 'snapshot_date', 'event']].sort_values(['ticker', 'snapshot_date'])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def tier_lag_stats(u):
    g = u[u.in_U0].assign(lag=lambda x: (x.signal_date - x.tier_date).dt.days)
    return g.groupby('trade_date').agg(n=('lag', 'size'), prior_rows=('lag', lambda s: int((s > 0).sum())),
                                       lag_max=('lag', 'max'), lag_median=('lag', 'median')).reset_index()


def data_checks(months, u, miss, px, cap):
    p1 = u[u.p1_rank > 0]
    return {'cap': str(cap.date()), 'price_max_date': str(px.date.max().date()), 'months': len(months),
            'month_status': months.status.value_counts().to_dict(),
            'p1_missing_from_snapshot': miss,
            'p1_not_in_U': int((~p1.in_U).sum()), 'p1_not_in_U0': int((~p1.in_U0).sum()),
            'p1_mcap_missing': int((~(p1.market_cap > 0)).sum()),
            'p1_mcap_missing_months': sorted({str(d.date()) for d in p1[~(p1.market_cap > 0)].trade_date}),
            'snapshot_after_signal': int((u.snapshot_date > u.signal_date).sum()),
            'tier_row_after_signal': int((u.tier_date > u.signal_date).sum()),
            'signal_not_before_trade': int((months.signal_date >= months.trade_date).sum()),
            'month_problems': [{'trade_date': str(r.trade_date.date()), 'status': r.status, 'reason': r.status_reason}
                               for r in months.itertuples() if r.status not in USABLE_STATUS]}


def write_bundle(out, parts, checks):
    json.dump(checks, open(os.path.join(out, 'data_checks.json'), 'w'), indent=1, default=str)
    for name, (df, kind) in parts.items():
        path = os.path.join(out, name)
        df.to_parquet(path, index=False) if kind == 'parquet' else df.to_csv(path, index=False)
    with open(os.path.join(out, 'bundle.sha256'), 'w') as f:
        for name in sorted(os.listdir(out)):
            if name != 'bundle.sha256': f.write(f'{sha256(os.path.join(out, name))}  {name}\n')


def main():
    a = parser().parse_args()
    _, end, cap = pc.check_window(a.start, a.end or a.cap, a.cap)     # F4: start ≤ end ≤ cap ≤ END_CAP
    if os.path.exists(a.outdir) and os.listdir(a.outdir): raise SystemExit('OUT_DIR must be new or empty')
    os.makedirs(a.outdir, exist_ok=True)
    conn = sc.connect()
    dates, full = pc.trading_calendar(conn, cap)
    panel, pmonths = load_panel(a.panel)
    panel = panel[panel.trade_date <= end]
    months = month_calendar(dates, a.offset, a.start, end, pmonths, panel)
    snaps = snapshot_members(conn, cap)
    u, miss = build_universe(conn, snaps, months, panel)
    tickers = sorted(set(u.stock_code[u.in_U0.astype(bool)]) | set(panel.ticker))
    require_tickers(tickers, a.outdir, {'offset': a.offset, 'start': a.start, 'end': str(end.date()), 'cap': str(cap.date()),
                                        'month_status': months.status.value_counts().to_dict()})
    first = max(pd.Timestamp(pc.DATA_START), months.signal_date.min() - pd.Timedelta(days=pc.PRICE_LEAD_DAYS))
    px = load_prices(conn, tickers, first, end, cap)
    dps = load_dps(conn, tickers, cap)
    ev = exit_events(snaps, tickers)
    checks = data_checks(months, u, miss, px, cap)
    checks.update({'offset': a.offset, 'start': a.start, 'end': str(end.date()), 'n_tickers': len(tickers), 'dps_rows': len(dps),
                   'reentry_events': int((ev.event == 'reentry').sum()), 'exit_events': int((ev.event == 'exit').sum()), 'panel_sha256': sha256(a.panel)})
    cal = pd.DataFrame({'date': full, 'priced': full <= cap})
    write_bundle(a.outdir, {'months.csv': (months, 'csv'), 'universe.parquet': (u, 'parquet'), 'prices.parquet': (px, 'parquet'),
                            'dps.csv': (dps, 'csv'), 'exits.csv': (ev, 'csv'), 'calendar.csv': (cal, 'csv'),
                            'tier_lag.csv': (tier_lag_stats(u), 'csv')}, checks)
    print(json.dumps(checks, indent=1, default=str))


if __name__ == '__main__':
    main()
