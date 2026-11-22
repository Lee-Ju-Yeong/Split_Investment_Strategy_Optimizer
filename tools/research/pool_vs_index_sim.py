"""연구 (b) 모의 실행: 자료 묶음(pool_vs_index_data.py)으로 P1·M·M0·U 계열·S20e·S20·ETF 계좌를 12조합(+무비용)으로 계산한다.

usage: python -B tools/research/pool_vs_index_sim.py BUNDLE_DIR OUT_DIR --etf-dir RUN_ETF_DIR [--skip-integer]
3월 실행 체크아웃(wt-79051b5)을 작업 디렉터리로 실행. DB를 읽지 않는다.
- --etf-dir는 pool_vs_index_etf_cap.py로 묶음 상한까지 자른 실행별 ETF 입력이다(F4). 069500_quality.csv가 없으면 오류(F5).
- 계좌는 start(정식 2014-01-01) 직전 거래일(정식 2013-12-30) 종가에 현금 CASH0로 시작한다(v2.1 A4).
- 조합: 배당(none | observed | late) × 폐지(last | zero) × 비용(net | gross). k는 묶음 하나가 하나.
- 사용할 수 없는 달(missing·lookup_failed·missing_universe·zero_ranked·panel_mismatch, 칸 오류)은 주문하지 않고 skipped에 사유를 남긴다(판정기가 막는다).
- 퇴출: 거래일 스냅숏은 그날 주문 뒤, 휴장일 스냅숏은 다음 거래일의 모든 처리 전(F6, G5). 부재 ~ 재등장 사이 주문은 미체결.
- 종목별 체결 수량은 모든 계좌·조합과 정수 계좌에 기록한다(G2). 출력 열은 pool_vs_index_common.TABLE_SCHEMAS로 고정(G1).
- DPS 30% 표시는 관측 창이 끝난 회계연도만 dps_quality.csv에 만든다(G4). 권리 0주 종목도 남긴다.
- ETF: 첫 교체일부터 매일 확인해 첫 유효일 시가에 진입. 계산용(무효 날 NaN)과 그림용(직전 유효값)을 분리(F5).
산출: curves.parquet, open_values.parquet, etf_curves.parquet, orders.csv, fills.parquet, marks.csv, mark_detail.csv, exits.csv,
      pays.csv, rights.csv, weights_check.csv, cells.csv, integer_*.csv, calendar.csv, months.csv, sim_checks.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pool_vs_index_common as pc
import pool_vs_index_accounting as ac

DIV_MODES, DELIST_MODES, COST_MODES = pc.DIV_MODES, pc.DELIST_MODES, pc.COST_MODES
MAIN_COMBO = ('observed', 'last', 'net')
WEIGHT_SUM_TOL = 1e-12
USABLE_STATUS = ('ok', 'zero_candidates')
ACCOUNTS, ETF_CODES = pc.ACCOUNTS, pc.ETF_CODES


def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('bundle'); ap.add_argument('outdir')
    ap.add_argument('--etf-dir', required=True)
    ap.add_argument('--skip-integer', action='store_true')
    return ap


def load_bundle(b):
    rd = lambda n, **k: pd.read_csv(os.path.join(b, n), **k)
    return {'months': rd('months.csv', parse_dates=['trade_date', 'signal_date']),
            'u': pd.read_parquet(os.path.join(b, 'universe.parquet')),
            'px': pd.read_parquet(os.path.join(b, 'prices.parquet')),
            'dps': rd('dps.csv', dtype={'ticker': str}), 'exits': rd('exits.csv', dtype={'ticker': str}, parse_dates=['snapshot_date']),
            'cal': rd('calendar.csv', parse_dates=['date']), 'checks': json.load(open(os.path.join(b, 'data_checks.json')))}


# ---------------- 가격 행렬 ----------------

def account_dates(cal, end, start):
    """기준일 = start 직전 거래일(정식 2013-12-30, k와 무관). 첫 주문 전까지는 현금(v2.1 A4)."""
    priced = pd.DatetimeIndex(cal.date[cal.priced])
    base = priced[priced < pd.Timestamp(start)][-1]
    return priced[(priced >= base) & (priced <= pd.Timestamp(end))]


def pivot(px, col, all_dates, tickers):
    return px.pivot(index='date', columns='stock_code', values=col).reindex(index=all_dates, columns=tickers)


def positive(df):
    return df.where(np.isfinite(df) & (df > 0))


def build_market(px, dates, tickers, full_cal):
    """장부는 유한한 양수 종가로만 갱신한다(F6). adj_ratio는 채운 값과 그날 값을 따로 둔다."""
    loaded = pd.DatetimeIndex(sorted(set(px.date) | set(dates)))
    get = lambda c: pivot(px, c, loaded, tickers).astype(float)
    adj_close, raw_close, ratio = positive(get('adj_close')), positive(get('close_price')), positive(get('adj_ratio'))
    sl = lambda df: df.reindex(dates).values.astype(float)
    rr = sl(ratio)
    last = np.where(np.isfinite(rr), np.arange(len(dates))[:, None], -1)
    last = np.maximum.accumulate(last, axis=0)                                    # 그날까지 마지막 유효 계수 위치(G3)
    mk = ac.Market(dates=dates, tickers=tickers, adj_open=sl(get('adj_open')), book=sl(adj_close.ffill()),
                   raw_open=sl(get('open_price')), raw_book=sl(raw_close.ffill()), volume=sl(get('volume')),
                   ratio=sl(ratio.ffill()), close_missing=adj_close.reindex(dates).isna().values, full_calendar=full_cal,
                   ratio_raw=rr, ratio_last_idx=last, implied_ratio=sl(adj_close) / sl(raw_close))
    mk.raw_close_missing = raw_close.reindex(dates).isna().values
    return mk


# ---------------- 목표 비중 ----------------

def equal(n_total, idx):
    w = np.zeros(n_total)
    if len(idx): w[idx] = 1.0 / len(idx)
    return w


def cell_weights(n_total, g, uflag, tcol, tix):
    """칸 목표 = 그 칸 P1 수 / 전체 P1 수, 칸 안은 U 동일 비중. 후보 있는 칸에 U가 없으면 오류."""
    w = np.zeros(n_total); p1 = g[g.p1_rank > 0]; u = g[g[uflag]]
    for (m, c), cnt in p1.groupby(['market', tcol]).size().items():
        members = u[(u.market == m) & (u[tcol] == c)].stock_code
        if c < 0 or not len(members): raise RuntimeError(f'cell {m},{c} has P1 but no {uflag} members ({g.trade_date.iloc[0].date()})')
        w[[tix[t] for t in members]] += (cnt / len(p1)) / len(members)
    return w


def cap_weights(n_total, g, market, top, tix):
    u = g[g.in_U & (g.market == market)].sort_values(['market_cap', 'stock_code'], ascending=[False, True]).head(top)
    w = np.zeros(n_total)
    if len(u): w[[tix[t] for t in u.stock_code]] = u.market_cap.values / u.market_cap.sum()
    return w


def month_weights(g, tix):
    n = len(tix); ix = lambda s: [tix[t] for t in s]
    p1 = g[g.p1_rank > 0].sort_values('p1_rank'); u = g[g.in_U]; k = min(pc.S20_N, len(p1))
    s20 = np.zeros(n); s20[ix(p1.stock_code.iloc[:pc.S20_N])] = pc.S20_WEIGHT
    return {'P1': equal(n, ix(p1.stock_code)),
            'P1_KOSPI': equal(n, ix(p1.stock_code[p1.market == 'KOSPI'])), 'P1_KOSDAQ': equal(n, ix(p1.stock_code[p1.market == 'KOSDAQ'])),
            'M': cell_weights(n, g, 'in_U', 'tercile_U', tix) if len(p1) else np.zeros(n),
            'M0': cell_weights(n, g, 'in_U0', 'tercile_U0', tix) if len(p1) else np.zeros(n),
            'U_EW': equal(n, ix(u.stock_code)), 'U_KOSPI_EW': equal(n, ix(u.stock_code[u.market == 'KOSPI'])),
            'U_KOSDAQ_EW': equal(n, ix(u.stock_code[u.market == 'KOSDAQ'])), 'U_small_EW': equal(n, ix(u.stock_code[u.tercile_U == 0])),
            'U_KOSPI_CW200': cap_weights(n, g, 'KOSPI', pc.CW_TOP['KOSPI'], tix),
            'U_KOSDAQ_CW150': cap_weights(n, g, 'KOSDAQ', pc.CW_TOP['KOSDAQ'], tix),
            'S20e': equal(n, ix(p1.stock_code.iloc[:k])), 'S20': s20}, list(ix(p1.stock_code))


def weight_checks(trade_date, ws, n_p1, g):
    row = {'trade_date': str(trade_date.date()), 'n_p1': n_p1}
    for a, w in ws.items():
        row[f'{a}_sum'] = float(w.sum())
    has = lambda m: bool(((g.p1_rank > 0) & (g.market == m)).any())
    expect = {'P1': float(n_p1 > 0), 'M': float(n_p1 > 0), 'M0': float(n_p1 > 0), 'S20e': float(n_p1 > 0),
              'S20': pc.S20_WEIGHT * min(n_p1, pc.S20_N), 'P1_KOSPI': float(has('KOSPI')), 'P1_KOSDAQ': float(has('KOSDAQ'))}
    for a, e in expect.items():
        row[f'{a}_ok'] = bool(abs(row[f'{a}_sum'] - e) <= WEIGHT_SUM_TOL and (ws[a] >= 0).all())
    return row


def cell_records(g):
    out = []
    for flag, tcol in (('in_U', 'tercile_U'), ('in_U0', 'tercile_U0')):
        u = g[g[flag]]
        for (m, c), cg in u.groupby(['market', tcol]):
            n1 = int((cg.p1_rank > 0).sum())
            out.append({'trade_date': str(g.trade_date.iloc[0].date()), 'universe': flag[3:], 'market': m, 'tercile': int(c), 'p1': n1,
                        'u': len(cg), 'p1_over_u': n1 / len(cg), 'liq_median': float(cg.liq.median()), 'mcap_median': float(cg.market_cap.median())})
    return out


# ---------------- 사건 ----------------

def exit_positions(exits, dates, tix):
    """거래일 스냅숏 → 그날 주문 뒤(post). 휴장일 스냅숏 → 다음 거래일 주문 전(pre). 기준일 이전 사건은 무시."""
    post, pre = {}, {}
    for r in exits[exits.event == 'exit'].itertuples():
        if r.ticker not in tix or r.snapshot_date <= dates[0]: continue
        i = int(dates.searchsorted(r.snapshot_date))
        if i >= len(dates): continue
        (post if dates[i] == r.snapshot_date else pre).setdefault(i, []).append(tix[r.ticker])
    arr = lambda d: {i: np.array(v) for i, v in d.items()}
    return arr(post), arr(pre)


def effect_position(dates, day):
    """스냅숏 사건이 주문에 반영되는 첫 위치: 거래일이면 다음 거래일, 휴장일이면 그다음 첫 거래일."""
    i = int(dates.searchsorted(day))
    return i + 1 if i < len(dates) and dates[i] == day else i


def absence_matrix(exits, dates, tix):
    """부재(exit) ~ 재등장(reentry) 사이를 True로(G5: 그 사이 그 종목 주문은 미체결)."""
    ab = np.zeros((len(dates), len(tix)), dtype=bool)
    ev = exits[exits.ticker.isin(tix) & exits.event.isin(['exit', 'reentry'])].sort_values(['ticker', 'snapshot_date'])
    for t, g in ev.groupby('ticker'):
        start = None
        for r in g.itertuples():
            if r.event == 'exit': start = effect_position(dates, r.snapshot_date)
            elif start is not None: ab[start:effect_position(dates, r.snapshot_date), tix[t]] = True; start = None
        if start is not None: ab[start:, tix[t]] = True
    return ab


def dps_arrays(dps, tix, y):
    val, obs = np.zeros(len(tix)), np.zeros(len(tix), dtype=bool)
    for r in dps[dps.fy == y].itertuples():
        if r.ticker in tix and np.isfinite(r.dps) and r.dps >= 0: val[tix[r.ticker]] = r.dps; obs[tix[r.ticker]] = True
    return val, obs


def dividend_events(dates, full_cal, dps, tix, mode):
    rec, pay, table = {}, {}, {}
    for y in sorted(set(dates.year)):
        rd = pc.record_date(full_cal, y)
        if rd is not None and dates[0] < rd <= dates[-1]: rec[int(dates.searchsorted(rd))] = y
        table[y] = dps_arrays(dps, tix, y)
        if mode == 'none': continue
        pdt = pc.pay_date(full_cal, y, mode)
        if pdt is not None and pdt <= dates[-1] and rd is not None and rd < pdt: pay[int(dates.searchsorted(pdt))] = y
    return rec, pay, table


def mark_positions(dates):
    s = pd.Series(np.arange(len(dates)), index=dates)
    return tuple(int(x) for x in s.groupby([dates.year, dates.month]).tail(1).values)


# ---------------- 실행 ----------------

def month_markets(g, tix, prev):
    m = prev.copy() if prev is not None else np.array(['KOSPI'] * len(tix), dtype=object)
    for t, mk in zip(g.stock_code, g.market):
        if t in tix: m[tix[t]] = mk          # U0 밖 종목은 계좌에 없다
    return m


def schedules(b, dates, tix):
    reb = {a: {} for a in ACCOUNTS}; order, markets = {}, {}; wchk, cells, skipped = [], [], []
    groups = {d: g for d, g in b['u'].groupby('trade_date')}; prev = None
    for m in b['months'].itertuples():
        i = int(dates.searchsorted(m.trade_date))
        if i >= len(dates) or dates[i] != m.trade_date: raise RuntimeError(f'trade date {m.trade_date.date()} not in calendar')
        g = groups.get(m.trade_date)
        if g is not None and len(g): prev = markets[i] = month_markets(g, tix, prev)
        if m.status not in USABLE_STATUS:
            reason = getattr(m, 'status_reason', '')
            skipped.append({'trade_date': str(m.trade_date.date()), 'reason': f'{m.status}: {reason}' if isinstance(reason, str) and reason else m.status}); continue
        try:
            ws, p1_order = month_weights(g, tix)
        except RuntimeError as e:          # P1 ⊄ U 등: 사유를 남기고 판정 보류(F3)
            skipped.append({'trade_date': str(m.trade_date.date()), 'reason': f'cell_error: {e}'}); continue
        for a, w in ws.items(): reb[a][i] = w
        order[i] = p1_order; wchk.append(weight_checks(m.trade_date, ws, len(p1_order), g)); cells += cell_records(g)
    return reb, order, markets, pd.DataFrame(wchk), pd.DataFrame(cells), skipped


def costs_for(mode, full_cal):
    if mode == 'gross': return ac.Costs(0.0, False)
    return ac.Costs(pc.COMMISSION, True, lambda d: pc.tax_rate(full_cal, d))


def run_accounts(mk, reb, base_ev, div_ev):
    curves, logs = {}, {}
    for div in DIV_MODES:
        rec, pay, table = div_ev[div]
        for dl in DELIST_MODES:
            for cm in COST_MODES:
                for a in ACCOUNTS:
                    ev = ac.Events(reb[a], base_ev['post'], base_ev['pre'], rec, pay, table, marks=base_ev['marks'], log_fills=True)
                    key = f'{a}|{div}|{dl}|{cm}'
                    curves[key], logs[key] = ac.run_fractional(mk, ev, costs_for(cm, mk.full_calendar), dl == 'zero', pc.CASH0)
        print('combo div', div, 'done', flush=True)
    return curves, logs


def flatten_logs(logs, part):
    rows = []
    for key, lg in logs.items():
        if lg.get(part) is None: continue
        a, div, dl, cm = key.split('|')
        rows += [{'account': a, 'div': div, 'delist': dl, 'cost': cm, **x} for x in lg[part]]
    return pd.DataFrame(rows)


def fills_frame(fill_lists, dates, tickers, keys=None):
    """배열로 모은 체결(위치, 종목 위치, 매도, 매수)을 표로. keys가 있으면 계좌·조합 열을 붙인다."""
    parts = []
    for key, fl in (fill_lists.items() if keys is None else [(None, fill_lists)]):
        for i, j, sold, bought, price in fl:
            df = pd.DataFrame({'date': np.repeat(str(dates[i].date()), len(j)), 'ticker': np.array(tickers, dtype=object)[j],
                               'sell_qty': sold, 'buy_qty': bought, 'sell_value': sold * price, 'buy_value': bought * price})
            if key is not None:
                a, d, x, c = key.split('|'); df.insert(0, 'cost', c); df.insert(0, 'delist', x); df.insert(0, 'div', d); df.insert(0, 'account', a)
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def as_bool(s):
    return s.astype(str).str.lower().isin(['true', '1'])


def read_dated_csv(path, cap):
    """R1: 날짜 필드만 먼저 보고 상한 이하 줄만 해석한다. 상한 뒤 줄이 있으면 오류."""
    with open(path, newline='') as f:
        header = f.readline(); di = header.rstrip('\r\n').split(',').index('date'); keep, late = [header], 0
        for line in f:
            if line.split(',', di + 1)[di] <= str(cap.date()): keep.append(line)
            else: late += 1
    if late: raise RuntimeError(f'ETF input {os.path.basename(path)} has rows after cap')
    import io
    return pd.read_csv(io.StringIO(''.join(keep)), parse_dates=['date']).set_index('date')


def etf_frame(etf_dir, code, dates, cap):
    """069500은 069500_quality.csv가 반드시 있어야 한다(가격 파일에 ohlc_ok 열이 있어도, R1). 입력 최대일은 묶음 상한 이하."""
    cap = pc.cap_date(cap)                   # 단독 호출에도 절대 상한(END_CAP) 검사
    e = read_dated_csv(os.path.join(etf_dir, f'{code}.csv'), cap)
    if code == pc.KODEX200:
        qpath = os.path.join(etf_dir, f'{code}_quality.csv')
        if not os.path.exists(qpath): raise RuntimeError(f'{code} quality file missing')
        ok = as_bool(read_dated_csv(qpath, cap).ohlc_ok).reindex(e.index)
        if ok.isna().any(): raise RuntimeError(f'{code} quality file does not cover all dates')
    elif 'ohlc_ok' in e.columns: ok = as_bool(e.ohlc_ok)
    else: raise RuntimeError(f'{code} has no ohlc_ok column')
    e = e.assign(ok=ok.astype(bool)).reindex(dates)
    return e[['open', 'close']].assign(ok=e.ok.fillna(False).astype(bool).values)


def run_etfs(etf_dir, dates, first_reb, cap):
    cols, entry = {}, {}
    for code in ETF_CODES:
        e = etf_frame(etf_dir, code, dates, cap); key = f'E_{code}'
        cols[f'{key}|open'] = pd.Series(np.where(e.ok.values, e.open.values, np.nan), index=dates)
        for cm in COST_MODES:
            r = ac.run_etf(dates, e, first_reb, pc.COMMISSION if cm == 'net' else 0.0, pc.CASH0)
            cols[f'{key}|{cm}'], cols[f'{key}|{cm}_display'] = r['calc'], r['display']
            if cm == 'net':
                cols[f'{key}|ok'], cols[f'{key}|entered'] = r['ok'], r['entered']
                entry[code] = {'date': str(dates[r['entry']].date()), 'shares': r['shares']} if r['entry'] is not None else None
    return pd.DataFrame(cols)[pc.frame_columns('etf_curves.parquet')], entry


def run_integer(mk, reb, order, markets, base_ev, div_ev):
    rec, pay, table = div_ev['observed']
    ev = ac.Events(reb['P1'], base_ev['post'], base_ev['pre'], rec, pay, table, order=order, markets=markets)
    return ac.run_integer(mk, ev, costs_for('net', mk.full_calendar), False, pc.CASH0, pc.tick_up)


def dps_quality(mk, dates, full_cal, dps, tix, logs, cap, universe=None):
    """G4: 관측 창(이듬해 4/25~5/15)이 min(상한, 실행 마지막 날) 전에 끝난 회계연도만.
    대상 종목은 관측 창 끝까지의 교체 달 U0·P1로 고정한다(뒤에 편입된 종목은 넣지 않음). 관측된 모든 종목(권리 0주 포함)의 DPS/권리일 원 종가와
    P1·M(주 조합)의 권리 수량을 연결한다. 권리일 원 종가가 없으면 직전 원 종가 장부를 쓰고 price_basis에 표시한다."""
    rows = []
    main = lambda a: logs[f'{a}|observed|last|net']['rights']
    for y in sorted(set(dps.fy)):
        wend = pd.Timestamp(f'{y + 1}-{pc.DPS_WINDOW[1]}'); rd = pc.record_date(full_cal, y)
        if wend > min(cap, dates[-1]) or rd is None or not (dates[0] < rd <= dates[-1]): continue
        known = known_tickers(universe, wend)
        i = int(dates.searchsorted(rd))
        q = {a: next((r['qty_array'] for r in main(a) if r['fy'] == y), np.zeros(len(tix))) for a in ('P1', 'M')}
        for r in dps[dps.fy == y].itertuples():
            if r.ticker not in tix or (known is not None and r.ticker not in known): continue
            j = tix[r.ticker]; px = mk.raw_book[i, j]; yld = r.dps / px if np.isfinite(px) and px > 0 else np.nan
            rows.append({'fy': y, 'window_end': str(wend.date()), 'ticker': r.ticker, 'dps': r.dps, 'record_date': str(rd.date()),
                         'record_raw_close': px, 'price_basis': 'previous_close' if mk.raw_close_missing[i, j] else 'record_close',
                         'yield': yld, 'flag_gt30': bool(np.isfinite(yld) and yld > pc.DPS_FLAG_YIELD),
                         'P1_rights_qty': float(q['P1'][j]), 'M_rights_qty': float(q['M'][j])})
    return pc.table_frame(rows, 'dps_quality.csv')


def known_tickers(universe, until):
    if universe is None: return None
    u = universe[pd.to_datetime(universe.trade_date) <= until]
    return set(u.stock_code[u.in_U0.astype(bool) | (u.p1_rank > 0)])


def log_summary(logs, quality):
    rights, exits = flatten_logs(logs, 'rights'), flatten_logs(logs, 'exits')
    main = lambda df: df[(df['div'] == 'observed') & (df.delist == 'last') & (df.cost == 'net')] if len(df) else df
    r = main(rights); e = main(exits); fl = quality[quality.flag_gt30.astype(bool)] if len(quality) else quality
    return {'rights_ratio_fallback_main': {a: int(g.n_ratio_fallback.sum()) for a, g in r.groupby('account')} if len(r) else {},
            'rights_undetermined_main': {a: int(g.n_rights_undetermined.sum()) for a, g in r.groupby('account')} if len(r) else {},
            'rights_undetermined_qty_main': {a: float(g.undetermined_qty.sum()) for a, g in r.groupby('account')} if len(r) else {},
            'dps_flagged_gt30': int(len(fl)), 'dps_flagged_gt30_P1_qty': float(fl.P1_rights_qty.sum()) if len(fl) else 0.0,
            'dps_flagged_gt30_M_qty': float(fl.M_rights_qty.sum()) if len(fl) else 0.0,
            'exits_main': {f'{a}|{w}': int(len(g)) for (a, w), g in e.groupby(['account', 'when'])} if len(e) else {},
            'held_without_book_days_max': max(int(lg['marks'][-1]['held_without_book_days_cum']) if lg['marks'] else 0 for lg in logs.values())}


def sim_checks(b, mk, curves, logs, wchk, skipped, quality):
    orders = flatten_logs(logs, 'orders')
    return {'bundle_checks': b['checks'], 'skipped_months': skipped, 'base_date': str(mk.dates[0].date()), 'end_date': str(mk.dates[-1].date()),
            'n_days': len(mk.dates), 'n_tickers': len(mk.tickers), 'n_curves': len(curves), 'k': b['checks']['offset'], 'cap': b['checks']['cap'],
            'start': b['checks']['start'], 'n_months': int(b['months'].trade_date.nunique()),
            'weights_all_ok': bool(wchk.filter(like='_ok').all().all()) if len(wchk) else False,
            'min_cash_after_orders': float(orders.cash_after.min()) if len(orders) else None,
            'max_abs_conservation_err': float(orders.conservation_err.abs().max()) if len(orders) else None,
            'base_value_all_cash0': bool(all(abs(v.iloc[0] - pc.CASH0) < 1e-6 for v in curves.values())),
            'adj_ratio_missing_rows': int(b['px'].adj_ratio.isna().sum()), 'raw_open_missing_rows': int(b['px'].open_price.isna().sum()),
            'raw_open_nonpositive_rows': int((b['px'].open_price <= 0).sum()),
            'close_nonpositive_rows': int((b['px'].close_price <= 0).sum()) + int((b['px'].adj_close <= 0).sum()),
            'volume_zero_with_open_rows': int(((b['px'].volume == 0) & (b['px'].open_price > 0)).sum()),
            'absent_ticker_days': int(mk.absent.sum()), **log_summary(logs, quality)}


def write_table(out, name, df):
    df = pc.table_frame(df, name)
    df.to_parquet(os.path.join(out, name), index=False) if name.endswith('.parquet') else df.to_csv(os.path.join(out, name), index=False)


def write_outputs(out, curves, logs, etf, wchk, cells, b, mk, quality):
    p = lambda n: os.path.join(out, n)
    pd.DataFrame(curves)[pc.frame_columns('curves.parquet')].to_parquet(p('curves.parquet'))
    pd.DataFrame({k: lg['open_values'] for k, lg in logs.items()})[pc.frame_columns('open_values.parquet')].to_parquet(p('open_values.parquet'))
    etf.to_parquet(p('etf_curves.parquet'))
    for part in ('orders', 'marks', 'mark_detail', 'exits', 'pays', 'rights'): write_table(out, f'{part}.csv', flatten_logs(logs, part))
    detail = [{'account': k.split('|')[0], 'div': k.split('|')[1], 'delist': k.split('|')[2], 'cost': k.split('|')[3], 'fy': r['fy'], 'date': r['date'], **x}
              for k, lg in logs.items() for r in lg['rights'] for x in r['ratio_cases']]
    write_table(out, 'rights_detail.csv', detail)
    write_table(out, 'fills.parquet', fills_frame({k: lg['fills'] for k, lg in logs.items()}, mk.dates, mk.tickers))
    write_table(out, 'dps_quality.csv', quality)
    wchk.to_csv(p('weights_check.csv'), index=False); cells.to_csv(p('cells.csv'), index=False)
    b['cal'].to_csv(p('calendar.csv'), index=False); write_table(out, 'months.csv', b['months'])


def write_integer(out, iv, ilog, mk):
    iv.rename('P1_int').to_frame().to_parquet(os.path.join(out, 'integer_curve.parquet'))
    for part in ('orders', 'ca', 'exits', 'pays'): write_table(out, f'integer_{part}.csv', ilog[part])
    write_table(out, 'integer_fills.csv', fills_frame(ilog['fills'], mk.dates, mk.tickers, keys=False))


def main():
    a = parser().parse_args()
    if os.path.exists(a.outdir) and os.listdir(a.outdir): raise SystemExit('OUT_DIR must be new or empty')
    os.makedirs(a.outdir, exist_ok=True)
    b = load_bundle(a.bundle); ck = b['checks']; cap = pd.Timestamp(ck['cap'])
    full_cal = pd.DatetimeIndex(b['cal'].date); dates = account_dates(b['cal'], ck['end'], ck['start'])
    u = b['u']; tickers = sorted(set(u.stock_code[u.in_U0]) | set(u.stock_code[u.p1_rank > 0])); tix = {t: i for i, t in enumerate(tickers)}
    mk = build_market(b['px'], dates, tickers, full_cal)
    mk.absent = absence_matrix(b['exits'], dates, tix)
    reb, order, markets, wchk, cells, skipped = schedules(b, dates, tix)
    post, pre = exit_positions(b['exits'], dates, tix)
    base_ev = {'post': post, 'pre': pre, 'marks': mark_positions(dates)}
    div_ev = {m: dividend_events(dates, full_cal, b['dps'], tix, m) for m in DIV_MODES}
    curves, logs = run_accounts(mk, reb, base_ev, div_ev)
    first_reb = int(dates.searchsorted(b['months'].trade_date.min()))
    etf, entry = run_etfs(a.etf_dir, dates, first_reb, cap)
    quality = dps_quality(mk, dates, full_cal, b['dps'], tix, logs, cap, b['u'])
    write_outputs(a.outdir, curves, logs, etf, wchk, cells, b, mk, quality)
    chk = sim_checks(b, mk, curves, logs, wchk, skipped, quality); chk['etf_entry'] = entry; chk['etf_dir'] = os.path.realpath(a.etf_dir)
    if not a.skip_integer:
        iv, ilog = run_integer(mk, reb, order, markets, base_ev, div_ev)
        write_integer(a.outdir, iv, ilog, mk); chk['integer_combo'] = 'observed|last|net'
    json.dump(chk, open(os.path.join(a.outdir, 'sim_checks.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in chk.items() if k != 'bundle_checks'}, indent=1, default=str))


if __name__ == '__main__':
    main()
