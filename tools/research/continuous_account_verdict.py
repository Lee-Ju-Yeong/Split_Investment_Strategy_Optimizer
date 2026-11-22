"""연속 계좌 진단(a) 집계·필수 검사·판정. 규칙은 결과를 보기 전에 고정한 runs/continuous-account-r1/DESIGN-v2.2.md를 따른다.

usage: python -B tools/research/continuous_account_verdict.py OUT_DIR ETF_CSV --mode formal|smoke [--anchor-only] [--approved-dir DIR]
3월 실행 체크아웃을 작업 디렉터리로 실행(거래일·상장폐지·거래량 조회에 DB를 읽기 전용으로 쓴다).
- formal: 필수 검사(승인 작업표 행 대조·argv·성공·엔진 상태·날짜·결측·대사·ETF·재현·주판정 기준 확정) 중 하나라도 실패하면
  checks.json만 쓰고 종료 코드 2. 승인 해시·작업표는 --approved-dir(기본 OUT_DIR/../approved).
- 2단계 대조(stage2_reconcile.json, continuous_account_stage2_check.py가 먼저 만든다)를 읽어 2단계 해석 상태를 따로 저장한다.
- smoke: 작업 구성·3월 재현 검사는 건너뛰고 나머지는 같다. 판정은 verdict_smoke.json에 official=false로만 쓴다.
- --anchor-only: 3월 재현 작업만 검사한다(일괄 실행이 나머지 작업 전에 호출).
출력: checks.json, summary_accounts.csv, summary_segments.csv, boundaries.csv, mechanism.csv, dividend_sensitivity.csv,
      boundary_sensitivity.csv, direct_verdict.json(두 계좌 직접 비교), stage2_interpretation.json(2단계 연결 상태), verdict.json(둘을 합친 보고용)
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import continuous_account_rules as rules
import continuous_account_jobs as jobsgen

CASH0, ETF_COMM, SELL_COMM = 1e7, 0.00015, 0.00015
ANCHOR_FINAL = 16603008
PRIMARY_END = pd.Timestamp('2023-12-31')
PERIODS = {'to2023': PRIMARY_END, 'to2025': pd.Timestamp('2025-12-31')}
VARIANTS = ('stage2_style', 'recv_par', 'liqcost')


# ---------- 읽기 ----------
def read_run_safe(outdir, prefix):
    try: return read_run(outdir, prefix), None
    except Exception as exc: return None, f'{prefix}: cannot read outputs ({type(exc).__name__}: {exc})'


def read_run(outdir, prefix):
    base = os.path.join(outdir, 'runs', prefix)
    c = pd.read_csv(base + '_curve.csv', parse_dates=['date']).set_index('date')
    s = pd.read_csv(base + '_state.csv', parse_dates=['date']).set_index('date')
    return {'df': c.join(s, how='left'), 'state_dates': s.index, 'meta': json.load(open(base + '_meta.json')),
            'endpos': pd.read_csv(base + '_endpos.csv', dtype={'ticker': str}),
            'divs': pd.read_csv(base + '_divs.csv', dtype={'ticker': str}) if os.path.getsize(base + '_divs.csv') > 2 else pd.DataFrame(),
            'fills': pd.read_csv(base + '_fills.csv', dtype={'ticker': str}, parse_dates=['date'])}


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ---------- 필수 검사 ----------
def check_status(outdir, jobs):
    st = pd.read_csv(os.path.join(outdir, 'job_status.tsv'), sep='\t')
    errs = [f'job not recorded: {p}' for p in set(jobs.prefix) - set(st.prefix)]
    errs += [f'job recorded {n} times: {p}' for p, n in st.prefix.value_counts().items() if n != 1]
    return errs + [f'job failed: {r.prefix} exit={r.exit}' for r in st.itertuples() if r.exit != 0]


def check_meta(prefix, m):
    errs = []
    if not m.get('success') or m.get('degraded_run'): errs.append(f'{prefix}: success={m.get("success")} degraded_run={m.get("degraded_run")}')
    if m.get('universe_mode') != 'strict_pit' or m.get('env_universe_mode') != 'strict_pit': errs.append(f'{prefix}: universe_mode={m.get("universe_mode")}')
    if m.get('candidate_lookup_error_count', 1) or m.get('source_lookup_error_days', 1): errs.append(f'{prefix}: candidate lookup errors')
    return errs


def check_values(prefix, run, dates):
    df, errs = run['df'], []
    if not df.index.equals(dates): errs.append(f'{prefix}: curve dates != trading dates')
    if not run['state_dates'].equals(df.index): errs.append(f'{prefix}: state dates != curve dates')
    cols = ['value', 'value_engine', 'receivable', 'engine_value', 'cash', 'n_tickers', 'n_lots', 'n_full', 'max_lots']
    if df.index.has_duplicates: errs.append(f'{prefix}: duplicate dates')
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all(): errs.append(f'{prefix}: missing or infinite values')
    elif (df.value <= 0).any(): errs.append(f'{prefix}: non-positive value')
    elif not np.allclose(df.value, df.value_engine + df.receivable, rtol=1e-9, atol=1e-3): errs.append(f'{prefix}: value != engine + receivable')
    elif not np.allclose(df.value_engine, df.engine_value, rtol=1e-9, atol=1e-3): errs.append(f'{prefix}: state engine_value != curve')
    return errs + check_endpos(prefix, run)


def check_endpos(prefix, run):
    ep, last = run['endpos'], run['df'].iloc[-1]
    if len(ep) and not np.isfinite(ep[['quantity', 'price_used']].to_numpy(dtype=float)).all():
        return [f'{prefix}: end positions have missing or infinite price/quantity']
    stock = float((ep.quantity * ep.price_used).sum()) if len(ep) else 0.0
    if not np.isfinite(stock) or abs(stock - (last.engine_value - last.cash)) > max(5.0, 1e-6 * last.value):
        return [f'{prefix}: end positions do not reconcile with engine value']
    return []


def check_etf(etf, dates, etf_csv, approved):
    errs = []
    if etf.index.has_duplicates: errs.append('ETF file has duplicate dates')
    miss = dates.difference(etf.index)
    if len(miss): errs.append(f'ETF missing {len(miss)} trading dates (first {miss[0].date()})')
    sub = etf[~etf.index.duplicated()].reindex(dates)[['open', 'close']].to_numpy(dtype=float)
    if not np.isfinite(sub).all(): errs.append('ETF missing or infinite price')
    elif (sub <= 0).any(): errs.append('ETF non-positive price')
    if approved is not None and approved.get('069500.csv') != sha256(etf_csv): errs.append('ETF file hash differs from approved hash')
    return errs


def read_approved(d):
    """approved/SHA256SUMS: 'hash  경로' 행(jobs.tsv 사본, ETF 원본). 파일 이름 -> 해시."""
    p = os.path.join(d, 'SHA256SUMS')
    if not os.path.exists(p): return None
    return {os.path.basename(l.split(maxsplit=1)[1].strip()): l.split()[0] for l in open(p) if l.strip()}


def check_jobs_approved(outdir, jobs, cal, approved):
    if approved is None: return ['approved list missing']
    errs = [] if approved.get('jobs.tsv') == sha256(os.path.join(outdir, 'jobs.tsv')) else ['jobs.tsv hash differs from approved hash']
    a = jobsgen.parser().parse_args(['_']); a.boundaries = a.boundaries.split(',')
    return errs + rules.compare_job_rows(jobs.to_dict('records'), jobsgen.build_jobs(cal.dates, a).to_dict('records'))


def check_contiguity(jobs, runs):
    errs = []
    for (k, div), g in jobs[jobs.kind == 'restart'].groupby(['k', 'div']):
        u = pd.DatetimeIndex(np.concatenate([runs[p]['df'].index.values for p in g.sort_values('start').prefix]))
        c = jobs[(jobs.kind == 'cont') & (jobs.k == k) & (jobs['div'] == div)].prefix
        if len(c) != 1 or not u.equals(runs[c.iloc[0]]['df'].index): errs.append(f'restart windows k={k} div={div} do not tile cont dates')
    return errs


def anchor_errors(jobs, runs):
    a = jobs[jobs.kind == 'anchor'].prefix
    if len(a) != 1 or a.iloc[0] not in runs: return ['anchor run missing']
    f = float(runs[a.iloc[0]]['df'].value.iloc[-1])
    return [] if round(f) == ANCHOR_FINAL else [f'anchor final {round(f)} != {ANCHOR_FINAL}']


def run_checks(a, jobs, runs, read_errs, cal, etf, approved):
    formal = a.mode == 'formal'
    errs = list(read_errs)
    if formal:
        errs += rules.primary_errors() + rules.check_job_structure(jobs.to_dict('records')) + check_jobs_approved(a.outdir, jobs, cal, approved)
    errs += check_status(a.outdir, jobs)
    for j in jobs.to_dict('records'):
        if j['prefix'] not in runs: errs.append(f"{j['prefix']}: outputs missing"); continue
        r = runs[j['prefix']]; dates = cal.dates[(cal.dates >= pd.Timestamp(j['start'])) & (cal.dates <= pd.Timestamp(j['end']))]
        errs += check_meta(j['prefix'], r['meta']) + rules.check_argv(j, r['meta'].get('argv', [])) + check_values(j['prefix'], r, dates)
    span = cal.dates[(cal.dates >= pd.Timestamp(jobs.start.min())) & (cal.dates <= pd.Timestamp(jobs.end.max()))]
    if not errs: errs += check_contiguity(jobs, runs)
    errs += check_etf(etf, span, a.etf, approved if formal else None)
    return errs + (anchor_errors(jobs, runs) if formal else [])


# ---------- 지표 ----------
def perf(v, base, base_date):
    yrs = (v.index[-1] - base_date).days / 365.25
    path = np.concatenate([[base], v.values.astype(float)])
    mdd = float((path / np.maximum.accumulate(path) - 1).min()) * 100
    r = pd.Series(path).pct_change().dropna()
    cagr = ((path[-1] / base) ** (1 / yrs) - 1) * 100
    return {'cagr': cagr, 'mdd': mdd, 'calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
            'sharpe': float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan, 'final': float(path[-1]), 'base': float(base)}


def exposure(df, inv=None):
    """inv: 이어 붙인 곡선의 실제 분모로 잰 투자 비중(없으면 기본 구간 비중). 종목·차수 수는 항상 기본 구간 값."""
    inv = (df.value_engine - df.cash) / df.value if inv is None else inv
    return {'avg_invested': float(inv.mean()), 'days_invested_ge90': float((inv >= 0.9).mean()), 'avg_tickers': float(df.n_tickers.mean()),
            'avg_full': float(df.n_full.mean()), 'max_full': int(df.n_full.max()), 'end_full': int(df.n_full.iloc[-1]), 'avg_lots': float(df.n_lots.mean())}


def etf_curve(etf, idx):
    e = etf.reindex(idx)
    o = float(e.open.iloc[0]); n = np.floor(CASH0 / (o * (1 + ETF_COMM)))
    return pd.Series(CASH0 - (n * o + np.floor(n * o * ETF_COMM)) + n * e.close.values, index=idx)


def prior_day(cal, d):
    return cal.dates[cal.dates.searchsorted(pd.Timestamp(d)) - 1]


# ---------- 경계 구성요소 ----------
def receivable_pay_date(cal, run):
    d = run['divs']
    if d.empty or 'event' not in d or run['df'].receivable.iloc[-1] <= 0: return None
    ev = d[d.event.notna()]; paid = set(ev[ev.event == 'paid'].fy)
    open_fy = [fy for fy in ev[ev.event == 'receivable'].fy if fy not in paid]
    return cal.pay_date(int(open_fy[-1]), min_day=16) if open_fy else None


def boundary(conn, cal, run):
    df, ep = run['df'], run['endpos']; end = df.index[-1]; last = df.iloc[-1]
    vol = sc.q(conn, "SELECT stock_code, volume FROM DailyStockPrice WHERE date=%s AND stock_code IN (%s)" % ('%s', ','.join(['%s'] * len(ep))),
               [str(end.date())] + list(ep.ticker)) if len(ep) else pd.DataFrame(columns=['stock_code', 'volume'])
    tradable = set(vol[vol.volume.astype(float) > 0].stock_code)
    val = ep.quantity * ep.price_used; disp = float(val[ep.ticker.isin(tradable)].sum())
    return {'end': str(end.date()), 'stock_value': float(last.value_engine - last.cash), 'cash': float(last.cash), 'receivable': float(last.receivable),
            'est_sell_cost': disp * (SELL_COMM + cal.sell_tax(end)), 'nondisposable_value': float(val.sum()) - disp,
            'liquidable_value': float(last.cash) + disp * (1 - SELL_COMM - cal.sell_tax(end)),
            'n_full_end': int(last.n_full), 'first_day_invested': float((df.value_engine.iloc[0] - df.cash.iloc[0]) / df.value.iloc[0]),
            'recv_pay': receivable_pay_date(cal, run)}


# ---------- 이어 붙이기 ----------
NEVER = pd.Timestamp.max


def chain_simple(wins, liqcost=False):
    """stage2_style: 구간 끝 평가액(미수 배당 포함) 전체를 다음 구간에 굴림(2단계 계산).
    liqcost: 실제로 초기화하는 경계(마지막 구간 끝 제외)의 종가 평가액에서 그날 추정 매도 비용을 뺀다. 처분 불가분은 평가액으로 이월.
    투자 비중은 두 방식 모두 청산 직전 기준이다: 경계일에도 분자(매도 전 보유액)와 분모(비용 차감 전 평가액)를 같은 시점으로 쓴다."""
    out, inv, base = [], [], CASH0
    for i, w in enumerate(wins):
        df, s = w['df'], base / CASH0; vv = df.value * s
        inv.append((df.value_engine - df.cash) / df.value)          # 청산 직전 기준(분자·분모 모두 매도 전)
        if liqcost and i < len(wins) - 1: vv.iloc[-1] -= s * w['b']['est_sell_cost']
        out.append(vv); base = float(vv.iloc[-1])
    return pd.concat(out), pd.concat(inv)


def settle_pending(df, units, pending):
    """지급일별 미수 잔액(액면)을 이 구간에 반영. 지급일(이후 첫 거래일) 종가에 기본 계좌 지분을 산다."""
    face = pd.Series(0.0, index=df.index); units = units.copy(); left = {}
    for pay, amt in sorted(pending.items()):
        if pay > df.index[-1]: face += amt; left[pay] = amt; continue
        d = df.index[df.index >= pay][0]
        face[df.index < d] += amt; units[df.index >= d] += amt / float(df.value.loc[d])
    return face, units, left


def chain_recv_par(wins):
    """recv_par: 투자 지분과 지급일별 미수 잔액을 따로 이월. 미수금은 지급일까지 액면(위험 노출 없음),
    지급일 종가에 그 구간 기본 계좌 지분을 사며 산 지분도 이후 배당 권리를 똑같이 가진다.
    반환: (평가액, 투자 비중 = 지분 x (엔진 평가액 - 현금) / 평가액)."""
    out, inv, invested, pending = [], [], CASH0, {}
    for w in wins:
        df = w['df']; units0 = pd.Series(invested / CASH0, index=df.index)
        face, units, pending = settle_pending(df, units0, pending)
        vv = units * df.value + face; out.append(vv); inv.append(units * (df.value_engine - df.cash) / vv)
        u_end = float(units.iloc[-1]); invested = u_end * float(df.value_engine.iloc[-1]) + 0.0
        if w['b']['receivable'] > 0:
            pay = w['b']['recv_pay'] if w['b']['recv_pay'] is not None else NEVER
            pending[pay] = pending.get(pay, 0.0) + u_end * w['b']['receivable']
    return pd.concat(out), pd.concat(inv)


def chain(wins, variant):
    if variant == 'recv_par': return chain_recv_par(wins)
    return chain_simple(wins, liqcost=(variant == 'liqcost'))


def delist_zero(conn, run):
    v, fl = run['df'].value.copy(), run['fills']
    dl = sc.q(conn, "SELECT stock_code, delisted_date FROM TickerUniverseHistory WHERE delisted_date BETWEEN %s AND %s",
              [str(v.index[0].date()), str(v.index[-1].date())])
    for t, dd in zip(dl.stock_code, pd.to_datetime(dl.delisted_date)):
        qty = fl[(fl.ticker == t) & (fl.date < dd)].dq.sum() if t in set(fl.ticker) else 0
        if qty <= 0: continue
        px = sc.q(conn, "SELECT adj_close FROM DailyStockPrice WHERE stock_code=%s AND date < %s ORDER BY date DESC LIMIT 1", [t, str(dd.date())])
        v[v.index >= dd] -= qty * float(px.adj_close.iloc[0])
    return v


# ---------- 표 ----------
def account_row(name, s, etf, base_date, period, key):
    v, exp_df, inv = s; cut = v.index <= PERIODS[period]; vv = v[cut]
    ex = exposure(exp_df[exp_df.index <= PERIODS[period]], None if inv is None else inv[cut]) if exp_df is not None else {}
    p = perf(vv, CASH0, base_date); e = perf(etf_curve(etf, vv.index), CASH0, base_date)
    return {**key, 'account': name, 'period': period, 'end': str(vv.index[-1].date()), **p, **ex,
            'exposure_basis': ('pre_liquidation' if name.endswith('liqcost') else 'chain_curve') if inv is not None else ('own_curve' if exp_df is not None else ''),
            **{f'etf_{x}': e[x] for x in ('cagr', 'mdd', 'calmar', 'final')}, **rules.judge(p, e)}


def account_rows(conn, cal, etf, key, cont, wins):
    bd = prior_day(cal, cont['df'].index[0]); rdf = pd.concat([w['df'] for w in wins])
    series = {'cont': (cont['df'].value, cont['df'], None)}
    for x in VARIANTS:
        vals, inv = chain(wins, x); series[f'restart_{x}'] = (vals, rdf, inv)
    if key['div'] == 'approx':
        series['cont_delist0'] = (delist_zero(conn, cont), None, None)
        series['restart_delist0'] = (chain([{**w, 'df': w['df'].assign(value=delist_zero(conn, w['run']))} for w in wins], 'stage2_style')[0], None, None)
    return [account_row(n, s, etf, bd, p, key) for n, s in series.items() for p in PERIODS]


def segment_rows(cal, etf, key, cont, wins):
    rows, c = [], cont['df']
    for i, w in enumerate(wins):
        rs = w['df']; bd = prior_day(cal, rs.index[0]); prev = c.value[c.index <= bd]
        e = perf(etf_curve(etf, rs.index), CASH0, bd); rp = perf(rs.value, CASH0, bd)
        cp = perf(c.value.loc[rs.index], float(prev.iloc[-1]) if len(prev) else CASH0, bd)
        rows.append({**key, 'window': w['window'], 'idx': i, 'judged': i < rules.JUDGED, 'start': str(rs.index[0].date()), 'end': str(rs.index[-1].date()),
                     'etf_cagr': e['cagr'], 'etf_mdd': e['mdd'],
                     **{f'restart_{a}': b for a, b in {**rp, **exposure(rs), **rules.judge(rp, e)}.items()},
                     **{f'cont_{a}': b for a, b in {**cp, **exposure(c.loc[rs.index]), **rules.judge(cp, e)}.items()}})
    s = pd.DataFrame(rows)
    return s.assign(gap_cagr=s.restart_cagr - s.cont_cagr, d_invested=s.cont_avg_invested - s.restart_avg_invested, d_full=s.cont_avg_full - s.restart_avg_full)


def boundary_rows(key, cont, wins):
    c = cont['df']; out = []
    for w in wins:
        b = w['b']; d = pd.Timestamp(b['end']); cr = c.loc[d]
        out.append({**key, 'window': w['window'], **{x: y for x, y in b.items() if x != 'recv_pay'}, 'recv_pay': str(b['recv_pay'].date()) if b['recv_pay'] is not None else '',
                    'cont_invested_at_end': float((cr.value_engine - cr.cash) / cr.value), 'cont_n_full_at_end': int(cr.n_full), 'etf_first_day_invested': 1.0})
    return out


def mechanism_rows(segs):
    out = []
    for r in segs.itertuples():
        out.append({'k': r.k, 'div': segs.loc[r.Index, 'div'], 'window': r.window, 'control_first_window': r.idx == 0,
                    'gap_cagr': r.gap_cagr, 'd_invested': r.d_invested, 'd_full': r.d_full,
                    'rel_full': rules.relation(r.gap_cagr, r.d_full, rules.MECH_EPS['gap_cagr'], rules.MECH_EPS['d_full']),
                    'rel_invested': rules.relation(r.gap_cagr, r.d_invested, rules.MECH_EPS['gap_cagr'], rules.MECH_EPS['d_invested'])})
    return out


# ---------- 판정 ----------
def labels_for(acc, segs, k, div):
    s = segs[(segs.k == k) & (segs['div'] == div) & segs.judged]; a = acc[(acc.k == k) & (acc['div'] == div)].set_index(['account', 'period'])
    out = {}
    for crit in rules.CRITERIA:
        rc, cc = int(s[f'restart_{crit}'].sum()), int(s[f'cont_{crit}'].sum())
        lab, ch = rules.count_label(rc, cc)
        full = {f'{p}_{v}': rules.bool_label(bool(a.loc[(f'restart_{v}', p), crit]), bool(a.loc[('cont', p), crit])) for p in PERIODS for v in VARIANTS}
        out[crit] = {'five_windows': {'restart': rc, 'cont': cc, 'restart_ge3': rc >= rules.PASS_MIN, 'cont_ge3': cc >= rules.PASS_MIN, 'label': lab, 'count_change': ch},
                     'full_period': full,
                     'cont_full_pass': {p: bool(a.loc[('cont', p), crit]) for p in PERIODS},
                     'restart_full_pass': {f'{p}_{v}': bool(a.loc[(f'restart_{v}', p), crit]) for p in PERIODS for v in VARIANTS}}
    out['gap_cagr'] = {f'{p}_{v}': float(a.loc[(f'restart_{v}', p), 'cagr'] - a.loc[('cont', p), 'cagr']) for p in PERIODS for v in VARIANTS}
    return out


def boundary_table(per):
    return {(k, d, p, c): {x: per[(k, d)][c]['full_period'][f'{p}_{x}'] for x in VARIANTS}
            for (k, d) in per for p in PERIODS for c in rules.CRITERIA}


def direct_verdict(acc, segs, mech):
    """두 계좌 직접 비교(현재 계산법). 원래·칼마 기준 각각의 최종 라벨을 모두 낸다."""
    keys = sorted({(int(k), d) for k, d in zip(acc.k, acc['div'])})
    per = {(k, d): labels_for(acc, segs, k, d) for k, d in keys}
    ks = [k for k, d in keys if d == 'approx']; k0 = {d: per[(0, d)] for k, d in keys if k == 0}
    out = {'primary_criteria': rules.PRIMARY_CRITERIA, 'primary_period': 'to2023 + five windows'}
    for crit in rules.CRITERIA:
        out[crit] = {'five_windows_across_k': rules.across_offsets(per[(k, 'approx')][crit]['five_windows']['label'] for k in ks),
                     'full_period_across_k': {x: rules.across_offsets(per[(k, 'approx')][crit]['full_period'][x] for k in ks) for x in per[(ks[0], 'approx')][crit]['full_period']},
                     'final': rules.final_verdict({k: per[(k, 'approx')][crit]['five_windows']['label'] for k in ks},
                                                  {d: x[crit]['five_windows']['label'] for d, x in k0.items()}),
                     'k0_dividend_labels': {d: {'five': x[crit]['five_windows']['label'], 'full_to2023': x[crit]['full_period']['to2023_stage2_style']} for d, x in k0.items()}}
    sens = rules.boundary_sensitivity(boundary_table(per))
    out['boundary_treatment_sensitive'] = bool(sens); out['boundary_sensitive_positions'] = sens
    out['gap_class'] = {x: rules.gap_class([per[(k, 'approx')]['gap_cagr'][x] for k in ks]) for x in per[(ks[0], 'approx')]['gap_cagr']}
    m = pd.DataFrame(mech); m = m[(m['div'] == 'approx') & ~m.control_first_window]
    out['mechanism_counts'] = {c: m[c].value_counts().to_dict() for c in ('rel_full', 'rel_invested')}
    out['per_offset'] = {f'k{k}_{d}': x for (k, d), x in per.items()}
    out['final_original'], out['final_calmar'] = out['pass']['final'], out['pass_calmar']['final']
    out['final'] = rules.primary_final({c: out[c]['final'] for c in rules.CRITERIA})
    out['fixed_note'] = rules.STAGE2_NOTE
    return out, boundary_table(per)


SENTENCES = {
    'stage2_verdict_flipped': '같은 기준에서 재시작 계좌는 2단계 통과선을 넘고 연속 계좌는 못 넘었다. 2단계의 분할 계좌 통과는 구간 재시작 계산에 기댄 것이다.',
    'stage2_verdict_unchanged': '2단계 통과선 판정은 재시작 여부로 바뀌지 않았다(통과 수 변화는 따로 적음).',
    'stage2_verdict_reverse_flipped': '같은 기준에서 연속 계좌가 통과선을 넘고 재시작 계좌는 못 넘었다(재시작이 오히려 불리).'}


WITHHELD = '이 기준에서는 2단계 보관 결과와의 연결이 설명되지 않아(통과 구간·문턱 변화 또는 미설명 차이) 2단계 판정에 대한 해석을 유보한다. 직접 비교 표만 공개한다.'


S2_STATUSES = ('linked', 'withheld', 'smoke_not_applicable')


def withheld_all(reason):
    return {'by_criterion': {n: {'status': 'withheld', 'reason': reason} for n in rules.CRITERION_NAME.values()}, 'loader_note': reason}


def stage2_interpretation(outdir):
    """대조기 종료 코드(stage2_check.exit, 일괄 실행이 기록)와 JSON을 확인한다. 비0 종료·누락·파싱 실패·구조 불일치는
    두 기준 모두 withheld. 이 함수는 예외를 던지지 않는다(직접 판정 보존)."""
    ex, p = os.path.join(outdir, 'stage2_check.exit'), os.path.join(outdir, 'stage2_reconcile.json')
    try:
        if os.path.exists(ex) and open(ex).read().strip() != '0': return withheld_all(f'stage2 check exited {open(ex).read().strip()}')
        if not os.path.exists(p): return withheld_all('stage2_reconcile.json missing')
        s2 = json.load(open(p))
        bc = s2.get('by_criterion') if isinstance(s2, dict) else None
        if not isinstance(bc, dict) or any(not isinstance(bc.get(n), dict) or bc[n].get('status') not in S2_STATUSES for n in rules.CRITERION_NAME.values()):
            return withheld_all('stage2_reconcile.json has unexpected structure')
        return s2
    except Exception as exc:
        return withheld_all(f'stage2_reconcile.json unreadable ({type(exc).__name__})')


def combined(direct, s2):
    """보고용. 첫 문장 고정 -> 기준별 2단계 해석 상태 -> 기준별 최종 라벨. 두 기준을 OR로 합친 단일 결론은 만들지 않는다.
    2단계에 대한 해석 문장은 그 기준의 연결이 linked일 때만 붙인다."""
    out = {'first_sentence': rules.STAGE2_NOTE, 'primary_criteria': rules.PRIMARY_CRITERIA,
           'final_original': direct['final_original'], 'final_calmar': direct['final_calmar'], 'stage2_interpretation': s2, 'by_criterion': {}}
    for c in rules.CRITERIA:
        n = rules.CRITERION_NAME[c]; st = s2.get('by_criterion', {}).get(n, {}).get('status', 'withheld')
        out['by_criterion'][n] = {'final': direct[c]['final'], 'stage2_link': st,
                                  'stage2_sentence': SENTENCES.get(direct[c]['final']) if st == 'linked' else WITHHELD}
    return out


def dividend_table(acc, segs):
    rows = []
    for d in rules.DIVS_K0:
        if not ((acc.k == 0) & (acc['div'] == d)).any(): continue
        lab = labels_for(acc, segs, 0, d)
        for crit in rules.CRITERIA:
            fw = lab[crit]['five_windows']
            rows.append({'div': d, 'criterion': crit, 'five_restart': fw['restart'], 'five_cont': fw['cont'], 'five_label': fw['label'], 'count_change': fw['count_change'],
                         **{f'full_{x}': y for x, y in lab[crit]['full_period'].items()}, **{f'gap_{x}': y for x, y in lab['gap_cagr'].items()}})
    return rows


# ---------- 실행 ----------
def assemble(conn, cal, etf, jobs, runs):
    acc, segs, bnd = [], [], []
    for (k, div), g in jobs[jobs.kind == 'restart'].groupby(['k', 'div']):
        key = {'k': int(k), 'div': div}
        cont = runs[jobs[(jobs.kind == 'cont') & (jobs.k == k) & (jobs['div'] == div)].prefix.iloc[0]]
        wins = [{'window': w.window, 'run': runs[w.prefix], 'df': runs[w.prefix]['df'], 'b': boundary(conn, cal, runs[w.prefix])}
                for w in g.sort_values('start').itertuples()]
        acc += account_rows(conn, cal, etf, key, cont, wins); segs.append(segment_rows(cal, etf, key, cont, wins)); bnd += boundary_rows(key, cont, wins)
    return pd.DataFrame(acc), pd.concat(segs, ignore_index=True), pd.DataFrame(bnd)


def publish_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=str); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def publish(outdir, mode, direct):
    """직접 판정을 먼저 원자적으로 발행한 뒤 2단계 해석을 읽는다(대조 실패가 직접 판정을 막지 않도록)."""
    out = lambda n: os.path.join(outdir, n)
    if mode == 'smoke':
        s2 = stage2_interpretation(outdir)
        publish_json({'official': False, 'note': 'smoke: code-path check only, not a verdict', 'structure': direct, 'stage2': s2, 'final': None}, out('verdict_smoke.json'))
        return
    publish_json(direct, out('direct_verdict.json'))
    s2 = stage2_interpretation(outdir)
    publish_json(s2, out('stage2_interpretation.json')); publish_json(combined(direct, s2), out('verdict.json'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir'); ap.add_argument('etf')
    ap.add_argument('--mode', choices=['formal', 'smoke'], required=True); ap.add_argument('--anchor-only', action='store_true')
    ap.add_argument('--approved-dir')
    a = ap.parse_args(); out = lambda n: os.path.join(a.outdir, n)
    conn = sc.connect(); cal = sc.Calendar(sc.trading_dates(conn))
    jobs = pd.read_csv(out('jobs.tsv'), sep='\t')
    if a.anchor_only:
        jobs = jobs[jobs.kind == 'anchor']; got = {p: read_run_safe(a.outdir, p) for p in jobs.prefix}
        runs = {p: r for p, (r, e) in got.items() if r}; errs = [e for r, e in got.values() if e]
        errs += anchor_errors(jobs, runs) + [e for p, r in runs.items() for e in check_meta(p, r['meta'])]
        print(json.dumps({'anchor_ok': not errs, 'errors': errs})); sys.exit(2 if errs else 0)
    got = {p: read_run_safe(a.outdir, p) for p in jobs.prefix}
    runs = {p: r for p, (r, e) in got.items() if r}; read_errs = [e for r, e in got.values() if e]
    approved = read_approved(a.approved_dir or os.path.join(os.path.dirname(os.path.abspath(a.outdir)), 'approved'))
    try:
        etf = pd.read_csv(a.etf, parse_dates=['date']).set_index('date'); errs = run_checks(a, jobs, runs, read_errs, cal, etf, approved)
    except Exception as exc:
        errs = read_errs + [f'check stage crashed: {type(exc).__name__}: {exc}']
    json.dump({'mode': a.mode, 'passed': not errs, 'errors': errs}, open(out('checks.json'), 'w'), indent=1)
    if errs:
        print('\n'.join(['CHECKS FAILED — no verdict issued'] + errs), file=sys.stderr); sys.exit(2)
    jobs = jobs[jobs.kind != 'anchor']
    acc, segs, bnd = assemble(conn, cal, etf, jobs, runs); mech = mechanism_rows(segs)
    direct, btab = direct_verdict(acc, segs, mech)
    tables = [('summary_accounts', acc), ('summary_segments', segs), ('boundaries', bnd), ('mechanism', pd.DataFrame(mech)),
              ('dividend_sensitivity', pd.DataFrame(dividend_table(acc, segs))),
              ('boundary_sensitivity', pd.DataFrame([{'k': k, 'div': d, 'period': p, 'criterion': c, **labs} for (k, d, p, c), labs in btab.items()]))]
    for name, df in tables: df.to_csv(out(name + '.csv'), index=False)
    publish(a.outdir, a.mode, direct)
    print(json.dumps({'mode': a.mode, 'final_original': direct['final_original'], 'final_calmar': direct['final_calmar']} if a.mode == 'formal' else {'mode': 'smoke'}))


if __name__ == '__main__':
    main()
