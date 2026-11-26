"""same-basket-r2 조립기: 모든 작업이 성공하고 검산이 통과했을 때만 설계 4·5절의 표와 문장을 만든다.

usage: python -B tools/research/same_basket_core_verdict.py OUT_DIR --mode formal|smoke [--etf CSV] [--fail REASON]
3월 실행 체크아웃을 작업 디렉터리로 실행한다(DB는 거래일 목록만 읽는다). 규칙은 결과를 보기 전에 고정한 runs/same-basket-r2/DESIGN-v1.md를 따른다.
- 거부 조건(하나라도 걸리면 checks.json만 쓰고 종료 코드 2, 표·문장 없음): 작업표가 다시 만든 기대 행과 다름(formal), 기록에 없는 작업,
  마지막 종료 코드가 0이 아닌 작업, X 재현 대조(r1 10절 1) 미통과(formal), 산출물 읽기 실패, 작업별 검사(분할 몫 X·H·K·보유 몫·S·작은 몫·C,
  formal은 H·K가 r1 본실행과 바이트 일치) 실패, 작업 사이 검사 실패.
- smoke: 작업표 대조·재현 대조·r1 바이트 대조는 건너뛴다. 표는 verdict/smoke_tables/에만 쓰고 verdict_smoke.json(official=false)에는 구조만 남긴다.
표준 출력에는 성과 값을 쓰지 않는다. --fail REASON: 검사·조립 없이 그 사유의 checks.json만 발행하고 예전 판정·표를 지운다(종료 코드 2).
발행은 한 묶음이다(임시 폴더에 만든 뒤 OUT_DIR/verdict/를 통째로 바꾼다).
산출(formal, OUT_DIR/verdict/): checks.json, summary_jobs.csv(108 조건), summary_baskets.csv, control.csv, weight_daily.csv(기본 조건), verdict.json, VERDICT.md
"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import same_basket_common as sb
import same_basket_rules as r1rules
import same_basket_checks as chk
import same_basket_ledger as ledger
import same_basket_verdict as v1
import same_basket_core_common as cc
import same_basket_core_rules as rules
import same_basket_core_checks as cchk

PUBLISHED = ('checks.json', 'summary_jobs.csv', 'summary_baskets.csv', 'control.csv', 'weight_daily.csv', 'verdict.json', 'VERDICT.md',
             'verdict_smoke.json', 'smoke_tables')
NEVER_NOTE = ('분할 몫이 기간 안에 한 번도 사지 못한 명단 종목: order_below_one_share = 시작일 주문액(그 몫 순자산의 2.2%)이 1주 가격보다 작음 / '
              'never_tier1_candidate = 엔진이 후보를 조회한 날 한 번도 1등급 후보에 없었음 / other = 후보였던 날이 있었지만 사지 못함. '
              '보유 몫·작은 몫의 못 산 종목은 시작일 시가에 한 주도 사지 못한 자리(끝까지 현금)다.')


# ---------------- 거부 조건 ----------------
def gate_errors(outdir, jobs, dates, formal):
    try: errs = r1rules.refusal_reasons(v1.status_map(outdir), list(jobs.prefix))
    except Exception as exc: errs = [f'job_status.tsv unreadable ({type(exc).__name__})']
    if not formal: return errs
    errs += sb.compare_jobs(jobs, cc.build_jobs(dates)) + r1rules.check_job_structure(jobs.to_dict('records'), cc.N_JOBS)
    return errs + v1.repro_errors(outdir)


def collect_errors(a, jobs, dates, baskets, hashes):
    errs = gate_errors(a.outdir, jobs, dates, a.mode == 'formal')
    if errs: return None, errs
    run_dir = os.path.join(a.outdir, 'runs'); outs = {}
    for job in jobs.itertuples():
        out, err = cchk.read_job(run_dir, job)
        if err: errs.append(err)
        else: outs[job.prefix] = out
    if errs: return None, errs
    ctx = {'dates': dates, 'baskets': baskets, 'hashes': hashes, 'etf': ledger.read_etf_capped(a.etf), 'run_dir': run_dir,
           'r1_ref': cchk.r1_reference() if a.mode == 'formal' else None}
    for job in jobs.itertuples(): errs += cchk.job_errors(job, outs[job.prefix], ctx)
    return outs, errs if errs else cchk.cross_errors(jobs, outs)


# ---------------- 지표 ----------------
def cagr(values, base_date):
    """values: 시작 직전 거래일부터의 일별 평가액. 연복리는 시작 직전 거래일 종가의 원금 1,000만 원 기준(r1과 같은 식)."""
    return v1.metrics(values.iloc[1:], base_date)


def main_row(job, out):
    base = out['s'].index[0]
    s, h, k = cagr(out['s'].value, base), cagr(out['h']['curve'].value, base), cagr(out['k']['curve'], base)
    sv, hv = out['s'].iloc[1:], out['h']['curve'].iloc[1:]
    w = cc.weight_stats(sv.stock_value, sv.value, hv.stock_value, hv.value)
    trades = cc.split_trades(out['x']['fills'])
    nb = v1.never_bought_counts(out['x'])
    row = {c: getattr(job, c) for c in ('id', 'basket', 'k', 'div', 'delist', 'start', 'end', 'prefix')}
    for name, m in (('s', s), ('h', h), ('k', k)): row.update({f'{name}_cagr': m['cagr'], f'{name}_mdd': m['mdd']})
    row.update(a=s['cagr'] - h['cagr'], b=s['cagr'] - k['cagr'], **w, weight_ok=cc.weight_ok(w['s_weight'], w['h_weight']), **trades,
               split_held=cc.split_effect_held(trades), split_never_bought=nb['x_never_bought'],
               **{k_.replace('x_never_bought_', 'split_never_'): v for k_, v in nb.items() if k_ != 'x_never_bought' and k_ != 'x_never_bought_list'},
               split_never_list=nb['x_never_bought_list'], core_not_bought=len(out['core']['meta']['skipped_at_start']),
               h_not_bought=len(out['h']['meta']['skipped_at_start']), basket_delisted=len(out['h']['meta']['delisted']))
    return row


def control_row(job, out, main):
    base = out['c'].index[0]
    c = cagr(out['c'].value, base)['cagr']
    return {'prefix': job.prefix, 'c_cagr': c, 'c_minus_h': c - main['h_cagr'], 's_minus_c': main['s_cagr'] - c,
            'small_not_bought': len(out['small']['meta']['skipped_at_start'])}


def basket_values(rows):
    values = {}
    for y, g in rows.groupby('basket'):
        m = g[(g.k == sb.MAIN[0]) & (g['div'] == sb.MAIN[1]) & (g.delist == sb.MAIN[2])]
        one = lambda col: float(m[col].iloc[0]) if len(m) == 1 else None
        cond = lambda r: {'weight_ok': bool(r.weight_ok), 'split_held': bool(r.split_held)}
        values[int(y)] = {'a': one('a'), 'b': one('b'), 'a_all': [float(v) for v in g.a], 'b_all': [float(v) for v in g.b],
                          'main': cond(next(m.itertuples())) if len(m) == 1 else {'weight_ok': False, 'split_held': True},
                          'all': [cond(r) for r in g.itertuples()]}
    return values


def basket_table(rows, summary):
    m = rows[(rows.k == sb.MAIN[0]) & (rows['div'] == sb.MAIN[1]) & (rows.delist == sb.MAIN[2])].copy().sort_values('basket')
    lab = summary['labels']
    for key in ('a', 'b'):
        m[f'{key}_sign'] = [lab[int(y)][key]['sign'] for y in m.basket]; m[f'{key}_dependence'] = [lab[int(y)][key]['dependence'] for y in m.basket]
    m['judged'] = [lab[int(y)]['judged'] for y in m.basket]
    m['gate_reasons'] = [' '.join(lab[int(y)]['reasons']) for y in m.basket]
    for key in ('n_conditions_failed', 'n_cash_unresolved', 'n_split_held'): m[key] = [lab[int(y)][key] for y in m.basket]
    flips = rows.groupby('basket').c_flips.sum()
    m['c_flips_of_18'] = [int(flips[y]) for y in m.basket]
    return m.reset_index(drop=True)


def weight_daily(jobs, outs):
    parts = []
    for job in jobs[(jobs.kind == 'main') & (jobs.k == sb.MAIN[0]) & (jobs['div'] == sb.MAIN[1]) & (jobs.delist == sb.MAIN[2])].itertuples():
        s, h = outs[job.prefix]['s'].iloc[1:], outs[job.prefix]['h']['curve'].iloc[1:]
        sw, hw = cc.daily_weight(s.stock_value, s.value), cc.daily_weight(h.stock_value, h.value)
        parts.append(pd.DataFrame({'basket': job.basket, 'date': s.index.strftime('%Y-%m-%d'), 's_weight': sw, 'h_weight': hw, 'gap': sw - hw}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['basket', 'date', 's_weight', 'h_weight', 'gap'])


def assemble(jobs, outs):
    mains = {j.prefix: main_row(j, outs[j.prefix]) for j in jobs[jobs.kind == 'main'].itertuples()}
    ctrl = []
    for j in jobs[jobs.kind == 'ctrl'].itertuples():
        mp = cc.prefix('main', j.basket, j.k, j.div, j.delist)
        ctrl.append({'main_prefix': mp, **control_row(j, outs[j.prefix], mains[mp])})
    ctrl = pd.DataFrame(ctrl)
    flags = rules.control_flags([{'main_prefix': r.main_prefix, 'a': mains[r.main_prefix]['a'], 's_minus_c': r.s_minus_c} for r in ctrl.itertuples()])
    ctrl['c_flips'] = [f['flips'] for f in flags]
    rows = pd.DataFrame(list(mains.values())).merge(ctrl.rename(columns={'prefix': 'ctrl_prefix'}), left_on='prefix', right_on='main_prefix', how='left').drop(columns='main_prefix')
    summary = rules.summarize(basket_values(rows))
    control = rows[['basket', 'k', 'div', 'delist', 'a', 's_minus_c', 'c_minus_h', 'c_flips', 'small_not_bought', 'core_not_bought']]
    return {'summary_jobs': rows, 'summary_baskets': basket_table(rows, summary), 'control': control, 'weight_daily': weight_daily(jobs, outs)}, summary


# ---------------- 발행 ----------------
def swap_in(outdir, staged):
    final, old = os.path.join(outdir, 'verdict'), os.path.join(outdir, f'.verdict.old.{os.getpid()}')
    for name in PUBLISHED:
        p = os.path.join(outdir, name)
        if os.path.isdir(p): shutil.rmtree(p)
        elif os.path.exists(p): os.remove(p)
    if os.path.exists(final): os.replace(final, old)
    os.replace(staged, final)
    if os.path.exists(old): shutil.rmtree(old)


def verdict_markdown(summary, tables):
    t, bt, c = summary['table'], tables['summary_baskets'], summary['counts']
    cell = lambda key: f"{len(t['cells'][key])} {t['cells'][key] or ''}"
    parts = ['# 80% 보유 + 20% 분할 (same-basket-r2) 결과', '',
             '규칙: runs/same-basket-r2/DESIGN-v1.md 4·5절. a = S − H, b = S − K (연복리 %p, 반올림 전 부호). 기본 조건: k=0, 5월 지급, 동결, 2년.', '',
             '## 문장', '', summary['sentence']['text'], '', summary['note'], '',
             '## 2×2 표 (기본 조건이 비중·거래 기준을 통과한 바구니. 칸 = 바구니 수와 개설 연도)', '',
             '| | S < K | S > K |', '|---|---|---|',
             f"| S < H | {cell('S<H·S<K')} | {cell('S<H·S>K')} |", f"| S > H | {cell('S>H·S<K')} | {cell('S>H·S>K')} |", '',
             f"따로 센 것: {rules.LABEL_CASH} {t['cash_unresolved']}, {rules.LABEL_SPLIT} {t['split_held']}, 동률 {t['tie']}, 누락 {t['missing']}. "
             f"a 가정 의존 {c['dependent']}(비중 미달 조건에서만 {c['dependent_only_weight_fail']}), 미확정 {c['undetermined']}. "
             f"b 가정 의존 {c['b_dependent']}(비중 미달 조건에서만 {c['b_dependent_only_weight_fail']}), 미확정 {c['b_undetermined']}. "
             f"{rules.NOTE_A_ONLY}.", '',
             '## 바구니별 (기본 조건 값, 18가지 조건의 가정 의존과 기준 미달 수)', '',
             v1.md_table(bt, ['basket', 's_cagr', 'h_cagr', 'k_cagr', 'a', 'b', 'a_sign', 'a_dependence', 'b_sign', 'b_dependence', 'judged', 'gate_reasons',
                              'n_conditions_failed', 'n_cash_unresolved', 'n_split_held']), '',
             '## 비중과 분할 몫 거래 (기본 조건, 판정 전제)', '',
             v1.md_table(bt, ['basket', 's_weight', 'h_weight', 'weight_gap', 'daily_gap_min', 'daily_gap_max', 'weight_ok', 'split_addbuys', 'split_takeprofits',
                              'split_held', 'split_tickers_bought', 'split_never_bought', 'core_not_bought', 'h_not_bought']), '',
             NEVER_NOTE, '', '## 대조 C (판정 아님: 보유 몫 800만 원 + 종목당 10만 원 단순 보유)', '',
             'c_flips_of_18 = 18가지 조건 가운데 sign(S − C)가 sign(S − H)와 다른 조건 수(정수 주 잔돈이 승패를 바꾼 조건).', '',
             v1.md_table(bt, ['basket', 'c_cagr', 'c_minus_h', 's_minus_c', 'c_flips_of_18', 'small_not_bought']), '',
             '## 함께 표기 (판정 아님)', '', v1.md_table(bt, ['basket', 's_mdd', 'h_mdd', 'k_mdd', 'basket_delisted']), '']
    return '\n'.join(parts)


def publish(outdir, mode, tables, summary):
    if mode == 'smoke':
        folder = os.path.join(outdir, 'smoke_tables'); os.makedirs(folder, exist_ok=True)
        for name, df in tables.items(): df.to_csv(os.path.join(folder, name + '.csv'), index=False)
        shape = {name: {'rows': len(df), 'columns': list(df.columns)} for name, df in tables.items()}
        v1.publish_text(os.path.join(outdir, 'verdict_smoke.json'), json.dumps({'official': False, 'note': 'smoke: code-path check only, not a verdict',
                        'tables': shape, 'summary_keys': sorted(summary)}, indent=1))
        return
    for name, df in tables.items(): df.to_csv(os.path.join(outdir, name + '.csv'), index=False)
    v1.publish_text(os.path.join(outdir, 'verdict.json'), json.dumps({'official': True, 'design': 'same-basket-r2/DESIGN-v1.md', **summary}, indent=1, ensure_ascii=False, default=str))
    v1.publish_text(os.path.join(outdir, 'VERDICT.md'), verdict_markdown(summary, tables))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir'); ap.add_argument('--mode', choices=['formal', 'smoke'], required=True)
    ap.add_argument('--etf', default=sb.ETF_CSV); ap.add_argument('--fail')
    a = ap.parse_args()
    staged = os.path.join(a.outdir, f'.verdict.tmp.{os.getpid()}'); os.makedirs(staged)
    if a.fail is not None:
        v1.write_checks(staged, a.mode, [f'run_all: {a.fail}']); swap_in(a.outdir, staged)
        print('failure published; no verdict', file=sys.stderr); sys.exit(2)
    folder = os.path.join(a.outdir, 'baskets')
    try:
        dates = sc.trading_dates(sc.connect()); jobs = sb.read_jobs(os.path.join(a.outdir, 'jobs.tsv'))
        baskets = sb.load_baskets(folder); hashes = {y: sb.sha256(sb.basket_path(y, folder)) for y in sb.YEARS}
        outs, errs = collect_errors(a, jobs, dates, baskets, hashes)
    except Exception as exc:
        outs, errs = None, [f'check stage crashed: {type(exc).__name__}: {exc}']
    if not errs:
        try:
            tables, summary = assemble(jobs, outs); publish(staged, a.mode, tables, summary)
        except Exception as exc:
            shutil.rmtree(staged); os.makedirs(staged); errs = [f'assembly crashed: {type(exc).__name__}: {exc}']
    v1.write_checks(staged, a.mode, errs)
    swap_in(a.outdir, staged)
    if errs:
        print(f'CHECKS FAILED ({len(errs)} errors) — no verdict issued; see verdict/checks.json', file=sys.stderr); sys.exit(2)
    print(json.dumps({'mode': a.mode, 'published': True}))


if __name__ == '__main__':
    main()
