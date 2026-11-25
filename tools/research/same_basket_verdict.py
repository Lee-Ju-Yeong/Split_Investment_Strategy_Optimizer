"""같은 종목 대조(same-basket-r1) 조립기: 모든 작업이 성공하고 검산이 통과했을 때만 설계 7·8절의 표와 문장을 만든다.

usage: python -B tools/research/same_basket_verdict.py OUT_DIR --mode formal|smoke [--etf CSV] [--fail REASON]
3월 실행 체크아웃을 작업 디렉터리로 실행한다(DB는 거래일 목록만 읽는다). 규칙은 결과를 보기 전에 고정한 runs/same-basket-r1/DESIGN-v1.1.md를 따른다.
- 거부 조건(하나라도 걸리면 checks.json만 쓰고 종료 코드 2, 표·문장 없음): 작업표가 다시 만든 기대 행과 다름, 기록에 없는 작업,
  마지막 종료 코드가 0이 아닌 작업, 재현 대조(10절 1) 미통과, 산출물 읽기 실패, 10절 2·3·5·6·7과 형식·대사 검사 실패.
- smoke: 작업표 대조·재현 대조는 건너뛴다. 표는 verdict/smoke_tables/에만 쓰고 verdict_smoke.json(official=false)에는 값 없이 구조만 남긴다.
표준 출력에는 성과 값을 쓰지 않는다.
--fail REASON: 검사·조립 없이 실패를 발행한다(일괄 실행기가 잠금을 잡은 뒤 재검사가 실패했을 때 부른다). verdict/에는 그 사유의
checks.json만 남고 예전 판정·표는 지워진다. 종료 코드 2. DB를 읽지 않는다.
발행은 한 묶음이다: 모든 산출물을 임시 폴더에 만든 뒤 OUT_DIR/verdict/를 통째로 바꾼다. 검사가 실패하면 verdict/에는 그때의
checks.json만 남고, 예전 판정·표(verdict/ 안과, r1 판본이 OUT_DIR 바로 아래 쓰던 같은 이름)는 지운다.
산출(formal, OUT_DIR/verdict/): checks.json, summary_jobs.csv, summary_baskets.csv, aux.csv, extension.csv, eligibility_by_year.csv, verdict.json, VERDICT.md
"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import same_basket_common as sb
import same_basket_rules as rules
import same_basket_checks as chk
import same_basket_aux as aux
import same_basket_ledger as ledger

REASON_INACTIVE, REASON_STOP = '매매 미발생 기간 초과', '손절매'
STATUS_COLS = ['id', 'prefix', 'exit', 'seconds', 'finished_utc']
MISSING_TITLE = '### X가 기간 안에 한 번도 사지 못한 명단 종목 (사유 3가지, 정의는 표 아래)'
NEVER_BOUGHT_NOTE = ('사유 정의: order_below_one_share = 시작일 주문액이 1주 가격보다 작음 / never_tier1_candidate = 기간 중 한 번도 1등급 후보에 '
                     '없었음(엔진이 후보를 조회한 날 기준) / other = 1등급 후보였던 날이 있었지만 사지 못함(빈자리 없음·재진입 대기 등). '
                     '첫날 뒤 등급을 이탈한 종목은 따로 세지 않는다.')
PUBLISHED = ('checks.json', 'summary_jobs.csv', 'summary_baskets.csv', 'aux.csv', 'extension.csv', 'eligibility_by_year.csv', 'verdict.json', 'VERDICT.md',
             'verdict_smoke.json', 'smoke_tables')


# ---------------- 거부 조건 ----------------
def status_map(outdir):
    st = pd.read_csv(os.path.join(outdir, 'job_status.tsv'), sep='\t')
    return {p: [int(x) for x in g.exit] for p, g in st.groupby('prefix', sort=False)}


def repro_errors(outdir):
    path = os.path.join(outdir, 'repro', 'repro_check.json')
    if not os.path.exists(path): return ['reproduction check (10.1) not recorded']
    return [] if json.load(open(path)).get('passed') is True else ['reproduction check (10.1) failed']


def gate_errors(outdir, jobs, dates, baskets, formal):
    """산출물을 읽기 전에 보는 조건."""
    try: errs = rules.refusal_reasons(status_map(outdir), list(jobs.prefix))
    except Exception as exc: errs = [f'job_status.tsv unreadable ({type(exc).__name__})']
    if not formal: return errs
    errs += sb.compare_jobs(jobs, sb.build_jobs(dates, baskets))
    errs += rules.check_job_structure(jobs.to_dict('records'), sb.N_JOBS)
    return errs + repro_errors(outdir)


def read_all(outdir, jobs):
    outs, errs = {}, []
    for job in jobs.itertuples():
        out, err = chk.read_job(os.path.join(outdir, 'runs'), job)
        if err: errs.append(err)
        else: outs[job.prefix] = out
    return outs, errs


def job_errors(job, out, dates, baskets, hashes, etf):
    basket = baskets[job.basket]
    errs = chk.x_errors(job, out['x'], dates, basket, hashes[job.basket])
    if job.kind == 'aux': return errs
    errs += chk.h_errors(job, out['h'], dates, basket, hashes[job.basket])
    archived = chk.archived_kodex(job.basket) if job.k == 0 else None
    return errs + chk.k_errors(job, out['k'], dates, etf, archived) + chk.first_day_errors(job, out)


def pick(jobs, kind, basket, k, div, delist):
    m = jobs[(jobs.kind == kind) & (jobs.basket == basket) & (jobs.k == k) & (jobs['div'] == div) & (jobs.delist == delist)]
    return next(m.itertuples()) if len(m) == 1 else None


def cross_errors(jobs, outs, baskets):
    """작업 사이 검사: 연장 절단(10.6), 폐지 없는 바구니의 두 상폐 가정 일치, 보조 합산(10.7), K가 배당·상폐 가정과 무관."""
    errs = []
    for y in sorted(jobs.basket.unique()):
        main, ext = pick(jobs, 'main', y, *sb.MAIN), pick(jobs, 'ext', y, *sb.MAIN)
        if main is None or ext is None: errs.append(f'basket {y}: main or extension job missing')
        else: errs += chk.cut_errors(main, outs[main.prefix], outs[ext.prefix])
        for (k, div), g in jobs[(jobs.kind == 'main') & (jobs.basket == y)].groupby(['k', 'div']):
            f, z = pick(jobs, 'main', y, k, div, 'freeze'), pick(jobs, 'main', y, k, div, 'zero')
            if f is not None and z is not None: errs += chk.no_delisting_errors(f, outs[f.prefix], outs[z.prefix])
        for k, g in jobs[(jobs.kind == 'main') & (jobs.basket == y)].groupby('k'):
            curves = [outs[p]['k']['curve'] for p in g.prefix]
            if any(not c.equals(curves[0]) for c in curves): errs.append(f'basket {y} k={k}: K differs across dividend/delisting assumptions')
        for ratio, g in jobs[(jobs.kind == 'aux') & (jobs.basket == y)].groupby('ratio'):
            errs += aux.group_errors(f'aux b{y} {ratio}', list(g.itertuples()), [outs[p] for p in g.prefix], baskets[y])
    return errs


# ---------------- 지표 ----------------
def metrics(values, base_date, base=sb.CASH0):
    """values: 시작일부터의 일별 평가액. 연복리는 시작일 직전 거래일 종가의 원금 기준."""
    final = float(values.iloc[-1])
    return {'cagr': sb.annualized(final, base_date, values.index[-1], base), 'mdd': sb.max_drawdown(values.values, base), 'final': final, 'pnl': final - base}


def x_side(run):
    c, st, f = run['curve'].iloc[1:], run['state'].iloc[1:], run['fills']
    sells = f[f.type == 'SELL'] if len(f) else f
    events = lambda word: int(sells[sells.reason.str.contains(word)][['date', 'ticker']].drop_duplicates().shape[0]) if len(sells) else 0
    return {'x_avg_invested': float(((c.value_engine - st.cash) / c.value).mean()), 'x_avg_cash': float((st.cash / c.value).mean()),
            'x_fills': int(len(f)), 'x_inactivity_exits': events(REASON_INACTIVE), 'x_stop_losses': events(REASON_STOP),
            'x_delisted_zeroed': len(run['sbmeta']['delisted_zeroed']), **never_bought_counts(run)}


def never_bought_counts(run):
    s, f = run['sbmeta'], run['fills']
    bought = set(f[f.type == 'BUY'].ticker) if len(f) else set()
    nb = chk.never_bought(s['allowed'], bought, set(s['start_unaffordable'] or []), set(s['ever_candidate']))
    return {'x_never_bought': len(nb), **{f'x_never_bought_{r}': sum(v == r for v in nb.values()) for r in chk.NEVER_BOUGHT_REASONS},
            'x_never_bought_list': ' '.join(f'{t}:{r}' for t, r in sorted(nb.items()))}


def job_row(job, out):
    base = out['x']['curve'].index[0]
    x, h, k = (metrics(v.iloc[1:], base) for v in (out['x']['curve'].value, out['h']['curve'].value, out['k']['curve']))
    hc = out['h']['curve'].iloc[1:]
    row = {c: getattr(job, c) for c in ('id', 'kind', 'basket', 'k', 'div', 'delist', 'start', 'end', 'prefix')}
    for name, m in (('x', x), ('h', h), ('k', k)): row.update({f'{name}_{key}': val for key, val in m.items()})
    row.update(a=x['cagr'] - h['cagr'], b=x['cagr'] - k['cagr'], **x_side(out['x']),
               h_avg_invested=float((hc.stock_value / hc.value).mean()), h_avg_cash=float((hc.cash / hc.value).mean()),
               h_not_bought=len(out['h']['meta']['skipped_at_start']), basket_delisted=len(out['h']['meta']['delisted']))
    return row


def aux_rows(jobs, outs, rows):
    """보조 a = 20개 계좌 합산 연복리 − 같은 바구니 H(주 값) 연복리."""
    out = []
    for (y, ratio), g in jobs[jobs.kind == 'aux'].groupby(['basket', 'ratio']):
        total, _ = aux.aggregate([outs[p] for p in g.prefix])
        m = metrics(total.iloc[1:], total.index[0])
        h = rows[(rows.kind == 'main') & (rows.basket == y) & (rows.k == sb.MAIN[0]) & (rows['div'] == sb.MAIN[1]) & (rows.delist == sb.MAIN[2])]
        out.append({'basket': int(y), 'ratio': ratio, **{f'aux_{k}': v for k, v in m.items()}, 'h_cagr': float(h.h_cagr.iloc[0]) if len(h) else np.nan,
                    'a_aux': m['cagr'] - float(h.h_cagr.iloc[0]) if len(h) else np.nan, 'accounts': len(g),
                    'accounts_never_entered': sum(len(outs[p]['x']['fills']) == 0 for p in g.prefix)})
    return pd.DataFrame(out)


def basket_values(rows, auxdf):
    values, auxv = {}, {}
    for y, g in rows[rows.kind == 'main'].groupby('basket'):
        m = g[(g.k == sb.MAIN[0]) & (g['div'] == sb.MAIN[1]) & (g.delist == sb.MAIN[2])]
        one = lambda col: float(m[col].iloc[0]) if len(m) == 1 else None
        values[int(y)] = {'a': one('a'), 'b': one('b'), 'a_all': [float(v) for v in g.a], 'b_all': [float(v) for v in g.b]}
        auxv[int(y)] = {r.ratio: float(r.a_aux) for r in auxdf[auxdf.basket == y].itertuples()} if len(auxdf) else {}
    return values, auxv


def extension_rows(jobs, outs, dates):
    """연장 계좌(판정 아님): 2023년 마지막 거래일(2024년 바구니 제외)과 마지막 날의 값·연복리."""
    out = []
    end2023 = dates[dates <= pd.Timestamp('2023-12-31')][-1]
    for job in jobs[jobs.kind == 'ext'].itertuples():
        o = outs[job.prefix]; base = o['x']['curve'].index[0]
        for label, day in (('end_2023', end2023), ('end', pd.Timestamp(job.end))):
            if label == 'end_2023' and (job.basket == 2024 or day > pd.Timestamp(job.end)): continue
            row = {'basket': job.basket, 'point': label, 'date': str(day.date())}
            for name, v in (('x', o['x']['curve'].value), ('h', o['h']['curve'].value), ('k', o['k']['curve'])):
                m = metrics(v.iloc[1:][v.index[1:] <= day], base); row.update({f'{name}_value': m['final'], f'{name}_cagr': m['cagr']})
            out.append(row)
    return pd.DataFrame(out)


def window_eligibility(jobs, outs):
    """기본 2년 구간의 진입 가능 종목 수 평균과 후보 없는 날 수. 매일 기록하는 연장 계좌의 기록을 2년 종료일에서 잘라 쓴다."""
    out = {}
    for job in jobs[jobs.kind == 'ext'].itertuples():
        main = pick(jobs, 'main', job.basket, *sb.MAIN)
        el = outs[job.prefix]['x']['eligible'].iloc[1:]
        el = el[el.index <= pd.Timestamp(main.end)] if main is not None else el
        out[int(job.basket)] = {'avg_eligible': float(el.n_eligible.mean()), 'days_no_eligible': int((el.n_eligible == 0).sum())}
    return out


def eligibility_rows(jobs, outs):
    """연도별 진입 가능 종목 수(명단 ∩ 그날 후보)와 후보 없는 날 수. 연장 계좌 기준(시작일 k=0)."""
    out = []
    for job in jobs[jobs.kind == 'ext'].itertuples():
        el = outs[job.prefix]['x']['eligible'].iloc[1:]
        for year, g in el.groupby(el.index.year):
            out.append({'basket': job.basket, 'year': int(year), 'days': len(g), 'avg_eligible': float(g.n_eligible.mean()),
                        'days_no_eligible': int((g.n_eligible == 0).sum()), 'days_tier2_blocked': int(g.tier_source.str.startswith('TIER_2_FALLBACK').sum())})
    return pd.DataFrame(out)


# ---------------- 발행 ----------------
def publish_text(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_checks(folder, mode, errs):
    publish_text(os.path.join(folder, 'checks.json'), json.dumps({'mode': mode, 'passed': not errs, 'errors': errs}, indent=1, ensure_ascii=False))


def swap_in(outdir, staged):
    """staged 폴더를 OUT_DIR/verdict로 바꿔 넣고, 예전 묶음과 r1 판본이 바로 아래 쓰던 같은 이름의 산출물을 지운다."""
    final, old = os.path.join(outdir, 'verdict'), os.path.join(outdir, f'.verdict.old.{os.getpid()}')
    for name in PUBLISHED:
        p = os.path.join(outdir, name)
        if os.path.isdir(p): shutil.rmtree(p)
        elif os.path.exists(p): os.remove(p)
    if os.path.exists(final): os.replace(final, old)
    os.replace(staged, final)
    if os.path.exists(old): shutil.rmtree(old)


def fmt(x, nd=2):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def md_table(df, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples():
        lines.append('| ' + ' | '.join(fmt(v) if isinstance(v, float) else str(v) for v in (getattr(r, c) for c in cols)) + ' |')
    return '\n'.join(lines)


def verdict_markdown(summary, baskets_df, auxdf, ext, elig):
    t = summary['table']; cell = lambda key: f"{len(t['cells'][key])} {t['cells'][key] or ''}"
    parts = ['# 같은 종목 대조 (same-basket-r1) 결과', '', '규칙: DESIGN-v1.1.md 7·8절. a = X − H, b = X − K (연복리 %p, 반올림 전 부호).', '',
             '## 문장', '', summary['sentence']['text'], '', summary['note'], '',
             '## 2×2 표 (기본값: k=0, 5월 지급, 동결, 2년. 칸 = 바구니 수와 개설 연도)', '',
             '| 분할 계좌의 연복리 | KODEX 보유보다 낮음 | KODEX 보유보다 높음 |', '|---|---|---|',
             f"| 같은 종목 보유보다 낮음 | {cell('neg/neg')} | {cell('neg/pos')} |", f"| 같은 종목 보유보다 높음 | {cell('pos/neg')} | {cell('pos/pos')} |", '',
             f"동률 {t['tie']}, 누락 {t['missing']}", '',
             '## 바구니별 (주 값과 가정 의존)', '', md_table(baskets_df, ['basket', 'x_cagr', 'h_cagr', 'k_cagr', 'a', 'b', 'a_sign', 'a_dependence', 'b_sign', 'b_dependence', 'aux_label']), '',
             '## 함께 표기 (판정 아님, 주 값 계좌)', '',
             md_table(baskets_df, ['basket', 'x_mdd', 'h_mdd', 'k_mdd', 'x_avg_invested', 'x_avg_cash', 'h_avg_invested', 'x_fills', 'x_inactivity_exits',
                                   'x_stop_losses', 'avg_eligible', 'days_no_eligible', 'basket_delisted', 'x_pnl', 'h_pnl', 'k_pnl']), '',
             MISSING_TITLE, '',
             md_table(baskets_df, ['basket', 'x_never_bought', 'x_never_bought_order_below_one_share', 'x_never_bought_never_tier1_candidate', 'x_never_bought_other', 'x_never_bought_list']), '',
             NEVER_BOUGHT_NOTE, '',
             '## 보조 시험 (배정 방식)', '', md_table(auxdf, ['basket', 'ratio', 'aux_cagr', 'h_cagr', 'a_aux', 'aux_mdd', 'accounts_never_entered']) if len(auxdf) else '(없음)', '',
             '## 연장 계좌 (판정 아님)', '', '12년 열위에는 판 뒤 다시 살 후보가 사라진 효과가 들어가므로 이 값으로 "분할 자체가 깎았다"고 쓰지 않는다.', '',
             md_table(ext, ['basket', 'point', 'date', 'x_value', 'h_value', 'k_value', 'x_cagr', 'h_cagr', 'k_cagr']) if len(ext) else '(없음)', '',
             '## 연도별 진입 가능 종목 수 (명단 ∩ 그날 후보)', '', md_table(elig, ['basket', 'year', 'days', 'avg_eligible', 'days_no_eligible', 'days_tier2_blocked']) if len(elig) else '(없음)', '']
    return '\n'.join(parts)


def basket_table(rows, summary, elig):
    m = rows[(rows.kind == 'main') & (rows.k == sb.MAIN[0]) & (rows['div'] == sb.MAIN[1]) & (rows.delist == sb.MAIN[2])].copy()
    lab = summary['labels']
    for key in ('avg_eligible', 'days_no_eligible'): m[key] = [elig.get(int(y), {}).get(key, np.nan) for y in m.basket]
    for key in ('a', 'b'):
        m[f'{key}_sign'] = [lab[int(y)][key]['sign'] for y in m.basket]; m[f'{key}_dependence'] = [lab[int(y)][key]['dependence'] for y in m.basket]
    m['aux_label'] = [summary['aux'][int(y)]['label'] for y in m.basket]
    return m.sort_values('basket').reset_index(drop=True)


def assemble(jobs, outs, dates):
    rows = pd.DataFrame([job_row(j, outs[j.prefix]) for j in jobs[jobs.kind != 'aux'].itertuples()])
    auxdf = aux_rows(jobs, outs, rows)
    values, auxv = basket_values(rows, auxdf)
    summary = rules.summarize(values, auxv)
    return {'summary_jobs': rows, 'summary_baskets': basket_table(rows, summary, window_eligibility(jobs, outs)), 'aux': auxdf, 'extension': extension_rows(jobs, outs, dates),
            'eligibility_by_year': eligibility_rows(jobs, outs)}, summary


def publish(outdir, mode, tables, summary):
    """outdir: 임시 발행 폴더."""
    if mode == 'smoke':
        folder = os.path.join(outdir, 'smoke_tables'); os.makedirs(folder, exist_ok=True)
        for name, df in tables.items(): df.to_csv(os.path.join(folder, name + '.csv'), index=False)
        shape = {name: {'rows': len(df), 'columns': list(df.columns)} for name, df in tables.items()}
        publish_text(os.path.join(outdir, 'verdict_smoke.json'), json.dumps({'official': False, 'note': 'smoke: code-path check only, not a verdict',
                     'tables': shape, 'summary_keys': sorted(summary)}, indent=1))
        return
    for name, df in tables.items(): df.to_csv(os.path.join(outdir, name + '.csv'), index=False)
    publish_text(os.path.join(outdir, 'verdict.json'), json.dumps({'official': True, 'design': 'DESIGN-v1.1.md', **summary}, indent=1, ensure_ascii=False, default=str))
    publish_text(os.path.join(outdir, 'VERDICT.md'), verdict_markdown(summary, tables['summary_baskets'], tables['aux'], tables['extension'], tables['eligibility_by_year']))


def collect_errors(a, jobs, dates, baskets, hashes):
    errs = gate_errors(a.outdir, jobs, dates, baskets, a.mode == 'formal')
    if errs: return None, errs
    outs, errs = read_all(a.outdir, jobs)
    if errs: return None, errs
    etf = ledger.read_etf_capped(a.etf)
    for job in jobs.itertuples(): errs += job_errors(job, outs[job.prefix], dates, baskets, hashes, etf)
    return outs, errs if errs else cross_errors(jobs, outs, baskets)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir'); ap.add_argument('--mode', choices=['formal', 'smoke'], required=True)
    ap.add_argument('--etf', default=sb.ETF_CSV); ap.add_argument('--fail')
    a = ap.parse_args()
    if a.fail is not None:
        staged = os.path.join(a.outdir, f'.verdict.tmp.{os.getpid()}'); os.makedirs(staged)
        write_checks(staged, a.mode, [f'run_all: {a.fail}']); swap_in(a.outdir, staged)
        print('failure published; no verdict', file=sys.stderr); sys.exit(2)
    folder = os.path.join(a.outdir, 'baskets')
    try:
        dates = sc.trading_dates(sc.connect()); jobs = sb.read_jobs(os.path.join(a.outdir, 'jobs.tsv'))
        baskets = sb.load_baskets(folder); hashes = {y: sb.sha256(sb.basket_path(y, folder)) for y in sb.YEARS}
        outs, errs = collect_errors(a, jobs, dates, baskets, hashes)
    except Exception as exc:
        outs, errs = None, [f'check stage crashed: {type(exc).__name__}: {exc}']
    staged = os.path.join(a.outdir, f'.verdict.tmp.{os.getpid()}'); os.makedirs(staged)
    if not errs:
        try:
            tables, summary = assemble(jobs, outs, dates); publish(staged, a.mode, tables, summary)
        except Exception as exc:
            shutil.rmtree(staged); os.makedirs(staged); errs = [f'assembly crashed: {type(exc).__name__}: {exc}']
    write_checks(staged, a.mode, errs)
    swap_in(a.outdir, staged)
    if errs:
        print(f'CHECKS FAILED ({len(errs)} errors) — no verdict issued; see verdict/checks.json', file=sys.stderr); sys.exit(2)
    print(json.dumps({'mode': a.mode, 'published': True}))


if __name__ == '__main__':
    main()
