"""같은 종목 대조(same-basket-r1) smoke 기술 검사: 실행 폴더의 산출물을 실자료와 대조한다. 성과 값(수익률·평가액)은 출력하지 않는다.

usage: python -B tools/research/same_basket_smoke_check.py OUT_DIR
3월 실행 체크아웃을 작업 디렉터리로 실행한다(DB는 SELECT만, 주가는 2025-12-30 이하). 조립기의 필수 검사(checks.json)에 더해:
- 파일 형식: 작업별 산출물과 열 이름
- 첫 주문 가능일: 주·연장 작업의 X 첫 체결일, H 최초 매수일, K 매수일이 등록한 시작일과 같은지(X는 체결이 있을 때)
- 명단 안 진입: X의 모든 매수 종목이 허용 명단 안인지
- H 손계산 대조(10절 4): 시작일의 수정 시가를 DailyStockPrice에서 엔진과 다른 경로로 직접 읽어 호가 올림·수량·수수료를 다시 계산
  (수수료 = 매수금액 × 3 // 20000, 수수료 절사로 한 주가 더 들어가는 경우까지 정수로 확인)
- 로그: 실행 폴더 안에 작업별 하위 출력 파일이 없고, 어떤 파일에도 하위 실행기의 최종 평가액 출력("final":)이 없다.
  --run-log FILE을 주면 일괄 실행 로그의 모든 줄이 번호·시간·상태 형식인지 본다.
- 시간: 작업 종류별 걸린 시간(초)
결과는 OUT_DIR/smoke_check.json(통과 여부·개수·날짜·시간만)과 표준 출력의 한 줄 요약.
"""
import json, os, re, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import same_basket_common as sb
import same_basket_checks as chk
import same_basket_ledger as ledger

X_COLUMNS = {'curve': ['date', 'value_engine', 'receivable', 'value'], 'fills': ['date', 'ticker', 'type', 'dq', 'dcash', 'order', 'reason'],
             'state': ['date', 'engine_value', 'cash', 'n_tickers', 'n_lots', 'n_full', 'max_lots'], 'endpos': ['ticker', 'quantity', 'lots', 'price_used'],
             'eligible': ['date', 'signal_date', 'tier_source', 'n_candidates', 'n_eligible', 'n_held', 'n_entry_signals']}
H_COLUMNS = {'H_curve': ['date', 'value', 'cash', 'receivable', 'stock_value', 'n_held'], 'H_pos': ['date', 'ticker', 'quantity', 'price'],
             'H_buys': ['date', 'ticker', 'kind', 'open', 'price', 'quantity', 'gross', 'commission', 'cost', 'budget'], 'K_curve': ['date', 'value']}


def header(path):
    with open(path) as f: return f.readline().strip().split(',')


def format_errors(run_dir, job):
    want = dict(X_COLUMNS, **(H_COLUMNS if job.kind != 'aux' else {}))
    errs = []
    for name, cols in want.items():
        path = os.path.join(run_dir, f'{job.prefix}_{name}.csv')
        if not os.path.exists(path): errs.append(f'{job.prefix}: {name} missing')
        elif os.path.getsize(path) > 2 and header(path) != cols: errs.append(f'{job.prefix}: {name} columns differ')
    return errs


def adjusted_open(conn, tickers, day):
    """엔진 조회식과 같은 정의(저장된 수정 시가, 없거나 어긋나면 시가 × 수정계수)를 엔진과 다른 경로로 직접 읽는다."""
    if pd.Timestamp(day) > pd.Timestamp(sb.END_CAP): raise SystemExit('price query after cap')
    ph = ','.join(['%s'] * len(tickers))
    d = sc.q(conn, f"SELECT stock_code, open_price, adj_open, adj_ratio FROM DailyStockPrice WHERE date = %s AND stock_code IN ({ph})", [day] + list(tickers))
    out = {}
    for r in d.itertuples():
        raw = float(r.open_price) * float(r.adj_ratio) if r.adj_ratio is not None else np.nan
        out[r.stock_code] = raw if r.adj_open is None or abs(float(r.adj_open) - raw) > 1e-5 else float(r.adj_open)
    return out


def hand_buy(open_px, budget):
    """손계산(정수만): 호가 올림 가격, 수수료 포함 budget 이하 최대 주식 수, 수수료(매수금액 × 3 // 20000)."""
    price = ledger.tick_up(open_px); budget = int(budget)
    n = budget // price
    while n > 0 and n * price + n * price * 3 // 20000 > budget: n -= 1
    return price, n, n * price * 3 // 20000


def hand_errors(job, h, opens, basket):
    """시작일 최초 매수를 손계산 식으로 다시 계산해 대조한다."""
    init = h['buys'][h['buys'].kind == 'initial'].set_index('ticker') if len(h['buys']) else pd.DataFrame()
    errs = []
    for t in basket:
        o = opens.get(t)
        can = o is not None and np.isfinite(o) and o > 0
        price, n, fee = hand_buy(o, sb.SLOT_CASH) if can else (None, 0, 0)
        if n == 0:
            if t in init.index and init.loc[t, 'quantity'] != 0: errs.append(f'{job.prefix}: {t} bought although not buyable by hand calculation')
            continue
        if t not in init.index or (init.loc[t, 'price'], init.loc[t, 'quantity'], init.loc[t, 'commission'], init.loc[t, 'cost']) != (price, n, fee, n * price + fee):
            errs.append(f'{job.prefix}: {t} initial buy differs from hand calculation')
    return errs


def start_info(job, out):
    f, start = out['x']['fills'], pd.Timestamp(job.start)
    first = f.date.min() if len(f) else None
    info = {'x_first_fill': None if first is None else str(first.date()), 'x_buys_on_start': int(((f.date == start) & (f.type == 'BUY')).sum()) if len(f) else 0}
    errs = [f'{job.prefix}: X first fill is not the start day'] if first is not None and first != start and job.kind != 'aux' else []
    if job.kind != 'aux':
        b = out['h']['buys']; info.update(h_initial_buys=int((b.kind == 'initial').sum()) if len(b) else 0, h_first_buy=str(b.date.min().date()) if len(b) else None,
                                          k_start=str(out['k']['curve'].index[1].date()))
    return info, errs


LOG_LINE = re.compile(r'^(job \d+ \d+s (ok|FAILED\(\d+\)|skipped)|jobs_total \d+ pending \d+|repro ok|precheck ok \(no run started\)|done: \S+|'
                      r'SAME_BASKET_EXIT=\d+|ERROR: .*)$')


def log_errors(out_dir, run_log=None):
    errs = []
    for root, _, files in os.walk(out_dir):
        if os.path.basename(root) == 'code': continue
        for name in files:
            path = os.path.join(root, name)
            if name.endswith(('.log', '.out')) and name != 'verdict.log': errs.append(f'child output file kept: {os.path.relpath(path, out_dir)}')
            if name.endswith(('.log', '.out', '.err', '.json', '.txt')) and '"final":' in open(path, errors='replace').read():
                errs.append(f'final-value field found in {os.path.relpath(path, out_dir)}')
    if run_log:
        bad = [l for l in open(run_log).read().splitlines() if not LOG_LINE.match(l)]
        if bad: errs.append(f'run log has {len(bad)} lines outside the number/time/status format')
    return errs


def main():
    out_dir = sys.argv[1]; run_dir = os.path.join(out_dir, 'runs')
    run_log = sys.argv[sys.argv.index('--run-log') + 1] if '--run-log' in sys.argv else None
    conn = sc.connect(); jobs = sb.read_jobs(os.path.join(out_dir, 'jobs.tsv')); baskets = sb.load_baskets(os.path.join(out_dir, 'baskets'))
    errs, starts, opens = [], {}, {}
    for job in jobs.itertuples():
        errs += format_errors(run_dir, job)
        out, err = chk.read_job(run_dir, job)
        if err: errs.append(err); continue
        info, e = start_info(job, out); errs += e
        allowed = [job.ticker] if job.kind == 'aux' else baskets[job.basket]
        errs += chk.x_basket_errors(job, out['x'], allowed)
        if job.kind == 'aux': continue
        starts[job.prefix] = info
        key = (job.basket, job.start)
        if key not in opens: opens[key] = adjusted_open(conn, baskets[job.basket], job.start)
        errs += hand_errors(job, out['h'], opens[key], baskets[job.basket])
    st = pd.read_csv(os.path.join(out_dir, 'job_status.tsv'), sep='\t').merge(jobs[['prefix', 'kind']], on='prefix')
    secs = {k: {'jobs': int(len(g)), 'mean': round(float(g.seconds.mean()), 1), 'max': int(g.seconds.max())} for k, g in st.groupby('kind')}
    errs += log_errors(out_dir, run_log)
    cpath = os.path.join(out_dir, 'verdict', 'checks.json')
    checks = json.load(open(cpath)) if os.path.exists(cpath) else {'passed': False, 'errors': ['verdict/checks.json missing']}
    aux_fill = {r: int(sum(os.path.getsize(os.path.join(run_dir, p + '_fills.csv')) > 2 for p in g.prefix)) for r, g in jobs[jobs.kind == 'aux'].groupby('ratio')}
    result = {'passed': not errs and checks['passed'], 'errors': errs, 'verdict_checks_passed': checks['passed'], 'verdict_check_errors': checks['errors'],
              'jobs': int(len(jobs)), 'starts': starts, 'aux_accounts_with_fills': aux_fill, 'seconds': secs}
    json.dump(result, open(os.path.join(out_dir, 'smoke_check.json'), 'w'), indent=1)
    print(json.dumps({'passed': result['passed'], 'errors': len(errs), 'verdict_checks_passed': checks['passed'], 'seconds': secs}))
    sys.exit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
