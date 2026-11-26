"""same-basket-r2 smoke 기술 검사: 실행 폴더의 산출물을 실자료와 대조한다. 성과 값(수익률·평가액·비중 값)은 출력하지 않는다.

usage: python -B tools/research/same_basket_core_smoke_check.py OUT_DIR [--run-log FILE]
3월 실행 체크아웃을 작업 디렉터리로 실행한다(DB는 SELECT만, 주가는 2025-12-30 이하). 조립기의 필수 검사(checks.json)에 더해:
- 파일 형식: 작업별 산출물과 열 이름(X·H·K·보유 몫·S, 대조는 보유 몫·작은 몫·C)
- 첫 주문 가능일: 분할 몫 첫 체결일(있을 때), H·보유 몫·작은 몫 최초 매수일, K 매수일이 등록한 시작일과 같은지
- 명단 안 진입: 분할 몫의 모든 매수 종목이 명단 안인지
- 손계산(설계 6절): 시작일 수정 시가를 DailyStockPrice에서 엔진과 다른 경로로 직접 읽어 H(50만 원)·보유 몫(40만 원)·작은 몫(10만 원)의
  호가 올림·수량·수수료를 정수 식으로 다시 계산(r1 smoke 손계산 함수)
- 비중 손계산: 보유 몫 주식평가액을 일별 보유 수량 × 가격(_core_H_pos)에서, 분할 몫 주식평가액을 엔진 평가액 − 현금에서 다시 더해
  S 곡선의 주식평가액·순자산과 같은지(값은 출력하지 않음)
- 로그: 하위 출력 파일이 없고 어떤 파일에도 "final": 필드가 없음. --run-log면 실행 로그가 번호·시간·상태 형식만인지
- 시간: 작업 종류별 걸린 시간(초)
결과는 OUT_DIR/smoke_check.json(통과 여부·개수·날짜·시간만)과 표준 출력의 한 줄 요약.
"""
import json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import same_basket_common as sb
import same_basket_checks as chk
import same_basket_smoke_check as s1
import same_basket_core_common as cc
import same_basket_core_checks as cchk

H_COLS = s1.H_COLUMNS['H_curve']
MAIN_FILES = dict(s1.X_COLUMNS, **s1.H_COLUMNS, **{'core_H_curve': H_COLS, 'core_H_pos': s1.H_COLUMNS['H_pos'], 'core_H_buys': s1.H_COLUMNS['H_buys'], 'S_curve': cc.S_COLS})
CTRL_FILES = {'core_H_curve': H_COLS, 'small_H_curve': H_COLS, 'core_H_buys': s1.H_COLUMNS['H_buys'], 'small_H_buys': s1.H_COLUMNS['H_buys'],
              'C_curve': ['date', 'core_value', 'small_value', 'value', 'stock_value', 'cash', 'receivable']}


def format_errors(run_dir, job):
    errs = []
    for name, cols in (MAIN_FILES if job.kind == 'main' else CTRL_FILES).items():
        path = os.path.join(run_dir, f'{job.prefix}_{name}.csv')
        if not os.path.exists(path): errs.append(f'{job.prefix}: {name} missing')
        elif os.path.getsize(path) > 2 and s1.header(path) != cols: errs.append(f'{job.prefix}: {name} columns differ')
    return errs


def hand_errors(job, part, h, opens, basket, slot):
    """r1 손계산(정수 식)을 그 몫의 종목당 한도로."""
    with cc.account(sb.CASH0, slot):
        return [e.replace(job.prefix, f'{job.prefix} [{part}]', 1) for e in s1.hand_errors(job, h, opens, basket)]


def first_buy_errors(job, part, h):
    b = h['buys']
    return [f'{job.prefix}: {part} first buy is not the start day'] if len(b) and b.date.min() != pd.Timestamp(job.start) else []


def weight_hand_errors(job, out):
    """보유 몫 주식평가액 = Σ 수량 × 가격(일별 보유 기록), 분할 몫 = 엔진 평가액 − 현금. 두 합이 S 곡선과 같고 비중이 0~1인지."""
    s, pos = out['s'], out['core']['pos']
    core = (pos.quantity * pos.price).groupby(pos.date).sum().reindex(s.index).fillna(0.0) if len(pos) else pd.Series(0.0, index=s.index)
    split = out['x']['curve'].value_engine - out['x']['state'].cash
    errs = []
    if not np.allclose(s.stock_value, core + split, rtol=0, atol=cchk.VALUE_TOL): errs.append(f'{job.prefix}: S stock value differs from hand sum')
    nav = out['core']['curve'].value + out['x']['curve'].value
    if not np.allclose(s.value, nav, rtol=0, atol=cchk.VALUE_TOL): errs.append(f'{job.prefix}: S net asset value differs from hand sum')
    w = cc.daily_weight((core + split).iloc[1:], nav.iloc[1:])
    if not ((w >= 0) & (w <= 1 + 1e-12)).all(): errs.append(f'{job.prefix}: weight outside [0, 1]')
    return errs


def job_checks(job, out, opens, basket):
    errs = hand_errors(job, 'core', out['core'], opens, basket, cc.CORE['slot']) + first_buy_errors(job, 'core', out['core'])
    if job.kind == 'ctrl': return errs + hand_errors(job, 'small', out['small'], opens, basket, cc.SMALL['slot']) + first_buy_errors(job, 'small', out['small'])
    errs += hand_errors(job, 'H', out['h'], opens, basket, sb.SLOT_CASH) + first_buy_errors(job, 'H', out['h'])
    f = out['x']['fills']
    if len(f) and f.date.min() != pd.Timestamp(job.start): errs.append(f'{job.prefix}: split first fill is not the start day')
    if out['k']['curve'].index[1] != pd.Timestamp(job.start): errs.append(f'{job.prefix}: K start differs')
    return errs + chk.x_basket_errors(job, out['x'], basket) + weight_hand_errors(job, out)


def main():
    out_dir = sys.argv[1]; run_dir = os.path.join(out_dir, 'runs')
    run_log = sys.argv[sys.argv.index('--run-log') + 1] if '--run-log' in sys.argv else None
    conn = sc.connect(); jobs = sb.read_jobs(os.path.join(out_dir, 'jobs.tsv')); baskets = sb.load_baskets(os.path.join(out_dir, 'baskets'))
    errs, opens, starts = [], {}, {}
    for job in jobs.itertuples():
        errs += format_errors(run_dir, job)
        out, err = cchk.read_job(run_dir, job)
        if err: errs.append(err); continue
        key = (job.basket, job.start)
        if key not in opens: opens[key] = s1.adjusted_open(conn, baskets[job.basket], job.start)
        errs += job_checks(job, out, opens[key], baskets[job.basket])
        if job.kind == 'main':
            f = out['x']['fills']
            starts[job.prefix] = {'split_first_fill': str(f.date.min().date()) if len(f) else None, 'h_first_buy': str(out['h']['buys'].date.min().date()),
                                  'core_first_buy': str(out['core']['buys'].date.min().date()), 'k_start': str(out['k']['curve'].index[1].date()),
                                  'split_n_fills': int(len(f)), 'core_initial_buys': int((out['core']['buys'].kind == 'initial').sum())}
    st = pd.read_csv(os.path.join(out_dir, 'job_status.tsv'), sep='\t').merge(jobs[['prefix', 'kind']], on='prefix')
    secs = {k: {'jobs': int(len(g)), 'mean': round(float(g.seconds.mean()), 1), 'max': int(g.seconds.max())} for k, g in st.groupby('kind')}
    errs += s1.log_errors(out_dir, run_log)
    cpath = os.path.join(out_dir, 'verdict', 'checks.json')
    checks = json.load(open(cpath)) if os.path.exists(cpath) else {'passed': False, 'errors': ['verdict/checks.json missing']}
    result = {'passed': not errs and checks['passed'], 'errors': errs, 'verdict_checks_passed': checks['passed'], 'verdict_check_errors': checks['errors'],
              'jobs': int(len(jobs)), 'starts': starts, 'seconds': secs}
    json.dump(result, open(os.path.join(out_dir, 'smoke_check.json'), 'w'), indent=1)
    print(json.dumps({'passed': result['passed'], 'errors': len(errs), 'verdict_checks_passed': checks['passed'], 'seconds': secs}))
    sys.exit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
