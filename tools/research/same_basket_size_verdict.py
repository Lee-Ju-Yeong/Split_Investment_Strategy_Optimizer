"""주문율 격자(same-basket-size-r1) 조립기: 모든 작업이 성공하고 검산이 통과했을 때만 설계 3·4절의 표와 문장을 만든다.

usage: python -B tools/research/same_basket_size_verdict.py OUT_DIR --mode formal|smoke [--fail REASON]
3월 실행 체크아웃을 작업 디렉터리로 실행한다(DB는 거래일 목록만 읽는다). 규칙은 결과를 보기 전에 고정한 DESIGN-v1.md를 따른다.
- 거부 조건(하나라도 걸리면 checks.json만 쓰고 종료 코드 2, 표·문장 없음): 작업표가 기대 행(756)과 다름(formal), 기록에 없는 작업,
  마지막 종료 코드가 0이 아닌 작업, r1 정식 검사 미통과·재사용 파일 해시 불일치, r1x 작업의 r1 대조 미통과(formal은 바이트 대조),
  산출물 읽기 실패, X 검사(r1 검사 재사용)·주문액 기록 검사·H/K 날짜 검사 실패.
- 판정 값: 2.2%는 r1 X(바이트 일치를 확인한 파일), 그 밖은 이번 실행 X. H·K는 r1 주 작업 산출물. 자금 지표의 주문액은 이번 실행의 기록.
- smoke: 작업표 대조를 건너뛰고, r1x는 이번 실행 X(종료일까지 자른 r1 대조 통과분)를 쓴다. 표는 verdict/smoke_tables/에만 쓰고
  verdict_smoke.json(official=false)에는 값 없이 구조만 남긴다.
--fail REASON: 검사·조립 없이 실패를 발행한다(일괄 실행기가 잠금 뒤 실패했을 때). 종료 코드 2. DB를 읽지 않는다.
발행은 한 묶음: 임시 폴더에 만든 뒤 OUT_DIR/verdict/를 통째로 바꾼다. 실패하면 verdict/에는 checks.json만 남는다.
표준 출력에는 성과 값을 쓰지 않는다.
산출(formal, OUT_DIR/verdict/): checks.json, summary_jobs.csv, baskets_by_ratio.csv, ratios.csv, verdict.json, VERDICT.md
"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import same_basket_common as sb
import same_basket_rules as r1rules
import same_basket_checks as chk
import same_basket_size_common as sz
import same_basket_size_checks as szc
import same_basket_size_funds as funds
import same_basket_size_rules as rules

PUBLISHED = ('checks.json', 'summary_jobs.csv', 'baskets_by_ratio.csv', 'ratios.csv', 'verdict.json', 'VERDICT.md', 'verdict_smoke.json', 'smoke_tables')
MAIN = sb.MAIN


# ---------------- 거부 조건 ----------------
def status_map(outdir):
    st = pd.read_csv(os.path.join(outdir, 'job_status.tsv'), sep='\t')
    return {p: [int(x) for x in g.exit] for p, g in st.groupby('prefix', sort=False)}


def gate_errors(outdir, jobs, dates, formal):
    try: errs = r1rules.refusal_reasons(status_map(outdir), list(jobs.prefix))
    except Exception as exc: errs = [f'job_status.tsv unreadable ({type(exc).__name__})']
    if formal: errs += sz.compare_jobs(jobs, sz.build_jobs(dates)) + sz.check_job_structure(jobs)
    if errs: return errs
    run_dir = os.path.join(outdir, 'runs')
    errs = [e for j in jobs[jobs.kind == 'r1x'].itertuples() for e in szc.r1cmp_errors(j, run_dir, formal)]
    return errs + szc.r1_reuse_errors(jobs, x_from_r1=formal)


def read_job(run_dir, job, formal):
    """X(판정 값), 주문액 기록, r1 H·K(종료일에서 자름)."""
    try:
        own = os.path.join(run_dir, job.prefix)
        xsrc = os.path.join(sz.R1_RUNS, job.r1prefix) if (job.kind == 'r1x' and formal) else own
        h, k = szc.truncate_hk(chk.read_h(os.path.join(sz.R1_RUNS, job.r1prefix)), chk.read_k(os.path.join(sz.R1_RUNS, job.r1prefix)), job.end)
        x = {**chk.read_x(xsrc), 'buyprices': szc.read_buyprices(own)}
        return {'x': x, 'log': szc.read_orderlog(own), 'h': h, 'k': k}, None
    except Exception as exc:
        return None, f'{job.prefix}: cannot read outputs ({type(exc).__name__}: {exc})'


def job_errors(job, out, dates, baskets, hashes):
    basket = baskets[job.basket]
    errs = szc.size_x_errors(job, out['x'], dates, basket, hashes[job.basket])
    if errs: return errs
    return szc.order_log_errors(job, out['x'], out['log'], dates) + szc.hk_date_errors(job, out['x'], out['h'], out['k'])


# ---------------- 지표 ----------------
def cagr(values):
    """values: 시작 전날 행을 포함한 일별 평가액. 연복리는 시작 전날 종가의 원금 기준(r1과 같은 식)."""
    return sb.annualized(float(values.iloc[-1]), values.index[0], values.index[-1], sb.CASH0)


def job_row(job, out, basket):
    x = out['x']
    m = funds.fund_metrics(x['curve'], x['state'], x['fills'], out['log'], basket)
    xc, hc, kc = cagr(x['curve'].value), cagr(out['h']['curve'].value), cagr(out['k']['curve'])
    row = {c: getattr(job, c) for c in ('id', 'kind', 'basket', 'k', 'div', 'delist', 'start', 'end', 'prefix')}
    row.update(ratio=float(job.ratio), x_cagr=xc, h_cagr=hc, k_cagr=kc, a=xc - hc, b=xc - kc,
               x_mdd=sb.max_drawdown(x['curve'].value.iloc[1:].values), **m, **funds.criteria(m))
    return row


def per_ratio_data(rows):
    """{주문율: {'base': {바구니: {a,b,ok}}, 'variants': {바구니: [..]}}}"""
    data = {}
    for r, g in rows.groupby('ratio', sort=False):
        base, variants = {}, {}
        for y, gy in g.groupby('basket'):
            pack = lambda df: [{'a': float(v.a), 'b': float(v.b), 'ok': bool(v.all_ok)} for v in df.itertuples()]
            m = gy[(gy.k == MAIN[0]) & (gy['div'] == MAIN[1]) & (gy.delist == MAIN[2])]
            base[int(y)] = pack(m)[0] if len(m) == 1 else {'a': None, 'b': None, 'ok': None}
            variants[int(y)] = pack(gy)
        data[float(r)] = {'base': base, 'variants': variants}
        data[float(r)]['summary'] = rules.ratio_summary(base, variants)
    return data


def assemble(jobs, outs, baskets):
    rows = pd.DataFrame([job_row(j, outs[j.prefix], baskets[j.basket]) for j in jobs.itertuples()])
    data = per_ratio_data(rows)
    ratios = [r for r in sz.RATIOS if r in data]
    base_ok = {(r, y): v['ok'] for r in ratios for y, v in data[r]['base'].items()}
    passes = funds.ratio_passes(base_ok, ratios, sorted({int(y) for y in rows.basket}))
    runs = funds.boundary_runs(passes)
    summary = {'grid': [sz.ratio_text(r) for r in ratios], 'passes': {sz.ratio_text(r): ok for r, ok in passes},
               'boundary_runs': [[sz.ratio_text(r) for r in run] for run in runs], 'boundary_text': rules.runs_text(runs),
               'sentence': rules.sentence(runs, data), 'notes': [rules.NOTE_PARTIAL, rules.NOTE_NO_P],
               'per_ratio': {sz.ratio_text(r): {k: data[r]['summary'][k] for k in ('table', 'dependent', 'undetermined')} for r in ratios}}
    return {'summary_jobs': rows, 'baskets_by_ratio': basket_table(rows, data), 'ratios': ratio_table(data, passes)}, summary


def basket_table(rows, data):
    m = rows[(rows.k == MAIN[0]) & (rows['div'] == MAIN[1]) & (rows.delist == MAIN[2])].copy()
    for key in ('a_sign', 'b_sign', 'a_dependence', 'b_dependence'):
        m[key] = [data[float(r)]['summary']['labels'][int(y)][key] for r, y in zip(m.ratio, m.basket)]
    m['ratio_text'] = [sz.ratio_text(r) for r in m.ratio]
    return m.sort_values(['ratio', 'basket']).reset_index(drop=True)


def ratio_table(data, passes):
    out = []
    for r, ok in passes:
        s = data[r]['summary']; t = s['table']; labels = s['labels']
        out.append({'ratio': r, 'ratio_text': sz.ratio_text(r), 'six_pass': ok, 'n_all_ok': sum(l['ok'] for l in labels.values()),
                    **{f'cell_{k.replace("/", "_")}': len(v) for k, v in t['cells'].items()},
                    'below_criteria': len(t['below_criteria']), 'tie': len(t['tie']), 'missing': len(t['missing']),
                    'a_dependent': len(s['dependent']['a']), 'b_dependent': len(s['dependent']['b']),
                    'a_undetermined': len(s['undetermined']['a']), 'b_undetermined': len(s['undetermined']['b'])})
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
    final, old = os.path.join(outdir, 'verdict'), os.path.join(outdir, f'.verdict.old.{os.getpid()}')
    for name in PUBLISHED:
        p = os.path.join(outdir, name)
        if os.path.isdir(p): shutil.rmtree(p)
        elif os.path.exists(p): os.remove(p)
    if os.path.exists(final): os.replace(final, old)
    os.replace(staged, final)
    if os.path.exists(old): shutil.rmtree(old)


def fmt(x, nd=2):
    if isinstance(x, (bool, np.bool_)): return 'O' if x else 'X'
    if isinstance(x, (float, np.floating)): return '' if not np.isfinite(x) else f'{x:.{nd}f}'
    return str(x)


def md_table(df, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(): lines.append('| ' + ' | '.join(fmt(getattr(r, c)) for c in cols) + ' |')
    return '\n'.join(lines)


def table_md(name, t):
    cell = lambda key: f"{len(t['cells'][key])} {t['cells'][key] or ''}"
    return [f'### 주문율 {name}', '', '| 분할 계좌의 연복리 | KODEX 보유보다 낮음 | KODEX 보유보다 높음 |', '|---|---|---|',
            f"| 같은 종목 보유보다 낮음 | {cell('neg/neg')} | {cell('neg/pos')} |", f"| 같은 종목 보유보다 높음 | {cell('pos/neg')} | {cell('pos/pos')} |", '',
            f"기준 미달 {len(t['below_criteria'])} {t['below_criteria'] or ''}, 동률 {len(t['tie'])}, 누락 {len(t['missing'])}", '']


def verdict_markdown(summary, tables):
    b = tables['baskets_by_ratio']
    parts = ['# 전액 분할 계좌 주문율 격자 (same-basket-size-r1) 결과', '',
             '규칙: DESIGN-v1.md 3·4절. a = X_r − H, b = X_r − K (연복리 %p, 반올림 전 부호). 기본 조건 = k=0, 5월 지급, 동결, 2년.', '',
             '## 경계 구간 (설계 3절)', '', f"격자: {' · '.join(summary['grid'])}. 여섯 바구니 세 기준 통과: "
             + ', '.join(f'{k} {"O" if v else "X"}' for k, v in summary['passes'].items()), '', f"경계 구간: {summary['boundary_text']}", '',
             '## 문장', '', summary['sentence']['text'], ''] + [n for note in summary['notes'] for n in (note, '')]
    parts += ['## 주문율별 요약', '', md_table(tables['ratios'], ['ratio_text', 'six_pass', 'n_all_ok', 'cell_neg_neg', 'cell_neg_pos', 'cell_pos_neg', 'cell_pos_pos',
                                                              'below_criteria', 'tie', 'missing', 'a_dependent', 'b_dependent', 'a_undetermined']), '',
              '## 2×2 표 (주문율별, 기본 조건. 기준 미달·동률·누락은 칸 밖에서 따로 셈)', '']
    for name, s in summary['per_ratio'].items(): parts += table_md(name, s['table'])
    parts += ['## 바구니별 수익 비교 (기본 조건)', '', md_table(b, ['ratio_text', 'basket', 'x_cagr', 'h_cagr', 'k_cagr', 'a', 'b', 'a_sign', 'a_dependence', 'b_sign', 'b_dependence']), '',
              '## 바구니별 자금 운용 (기본 조건, 비율은 0~1)', '',
              md_table(b, ['ratio_text', 'basket', 'avg_invested', 'short_days_ratio', 'add_buy_share', 'n_profit_sells', 'first_day_invested', 'min_cash_share',
                           'n_bought', 'n_never_bought', 'invest_ok', 'capacity_ok', 'split_ok', 'label']), '',
              '기준: 평균 투자 비중 ≥ 0.90, 다음 주문 여력 부족일 비율 ≤ 0.20, 추가매수 체결액 비중 ≥ 0.20 이고 익절 ≥ 1건. 18가정 전체 값은 summary_jobs.csv.', '']
    return '\n'.join(parts)


def publish(staged, mode, tables, summary):
    if mode == 'smoke':
        folder = os.path.join(staged, 'smoke_tables'); os.makedirs(folder)
        for name, df in tables.items(): df.to_csv(os.path.join(folder, name + '.csv'), index=False)
        shape = {name: {'rows': len(df), 'columns': list(df.columns)} for name, df in tables.items()}
        publish_text(os.path.join(staged, 'verdict_smoke.json'), json.dumps({'official': False, 'note': 'smoke: code-path check only, not a verdict',
                     'tables': shape, 'summary_keys': sorted(summary), 'sentence_kind': summary['sentence']['kind']}, indent=1))
        return
    for name, df in tables.items(): df.to_csv(os.path.join(staged, name + '.csv'), index=False)
    publish_text(os.path.join(staged, 'verdict.json'), json.dumps({'official': True, 'design': 'DESIGN-v1.md', **summary}, indent=1, ensure_ascii=False, default=str))
    publish_text(os.path.join(staged, 'VERDICT.md'), verdict_markdown(summary, tables))


def collect(outdir, mode, dates):
    formal = mode == 'formal'
    jobs = sz.read_jobs(os.path.join(outdir, 'jobs.tsv'))
    folder = os.path.join(outdir, 'baskets')
    baskets = sb.load_baskets(folder); hashes = {y: sb.sha256(sb.basket_path(y, folder)) for y in sb.YEARS}
    errs = gate_errors(outdir, jobs, dates, formal)
    if errs: return jobs, None, baskets, errs
    outs, errs = {}, []
    for job in jobs.itertuples():
        out, err = read_job(os.path.join(outdir, 'runs'), job, formal)
        if err: errs.append(err); continue
        outs[job.prefix] = out; errs += job_errors(job, out, dates, baskets, hashes)
    return jobs, outs, baskets, errs


def fail_only(outdir, mode, reason):
    staged = os.path.join(outdir, f'.verdict.tmp.{os.getpid()}'); os.makedirs(staged)
    write_checks(staged, mode, [f'run_all: {reason}']); swap_in(outdir, staged)
    print('failure published; no verdict', file=sys.stderr); sys.exit(2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir'); ap.add_argument('--mode', choices=['formal', 'smoke'], required=True); ap.add_argument('--fail')
    a = ap.parse_args()
    if a.fail is not None: fail_only(a.outdir, a.mode, a.fail)
    try:
        jobs, outs, baskets, errs = collect(a.outdir, a.mode, sc.trading_dates(sc.connect()))
    except Exception as exc:
        errs = [f'check stage crashed: {type(exc).__name__}: {exc}']
    staged = os.path.join(a.outdir, f'.verdict.tmp.{os.getpid()}'); os.makedirs(staged)
    if not errs:
        try:
            tables, summary = assemble(jobs, outs, baskets); publish(staged, a.mode, tables, summary)
        except Exception as exc:
            shutil.rmtree(staged); os.makedirs(staged); errs = [f'assembly crashed: {type(exc).__name__}: {exc}']
    write_checks(staged, a.mode, errs); swap_in(a.outdir, staged)
    if errs:
        print(f'CHECKS FAILED ({len(errs)} errors) — no verdict issued; see verdict/checks.json', file=sys.stderr); sys.exit(2)
    print(json.dumps({'mode': a.mode, 'published': True}))


if __name__ == '__main__':
    main()
