"""same-basket-r2 산출물 읽기와 기술 검사(설계 6절). 성과 값을 출력하지 않는다. 모든 검사는 오류 문자열 목록을 돌려준다.

r1 검사(same_basket_checks)를 불러 쓴다. r1 검사는 X 계좌 금액을 expect_account()에서, H 계좌 금액을 same_basket_common.CASH0·SLOT_CASH에서
읽으므로 분할 몫(200만 원)·보유 몫(40만 원 × 20)·작은 몫(10만 원 × 20)은 split_account()·same_basket_core_common.account() 안에서 검사한다.
"""
import contextlib, hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import same_basket_common as sb
import same_basket_checks as chk
import same_basket_core_common as cc

VALUE_TOL = chk.VALUE_TOL


@contextlib.contextmanager
def split_account():
    """r1 X 검사가 기대하는 계좌를 분할 몫(200만 원, 주문율 2.2%, 명단 전체)으로 잠시 바꾼다."""
    old = chk.expect_account
    chk.expect_account = lambda job: {'initial_cash': cc.SPLIT_CASH, 'order_investment_ratio': cc.SPLIT_RATIO, 'only': None}
    try: yield
    finally: chk.expect_account = old


# ---------------- 읽기 ----------------
def read_frame(path):
    return pd.read_csv(path, parse_dates=['date']).set_index('date')


def read_job(run_dir, job):
    """main: X(분할 몫)·H·K·core·S, ctrl: core·small·C. 읽기 실패는 (None, 오류)."""
    p = os.path.join(run_dir, job.prefix)
    try:
        if job.kind == 'main':
            out = {'x': chk.read_x(p), 'h': chk.read_h(p), 'k': chk.read_k(p), 'core': chk.read_h(p + '_core'), 's': read_frame(p + '_S_curve.csv')}
        else:
            out = {'core': chk.read_h(p + '_core'), 'small': chk.read_h(p + '_small'), 'c': read_frame(p + '_C_curve.csv')}
        return out, None
    except Exception as exc:
        return None, f'{job.prefix}: cannot read outputs ({type(exc).__name__}: {exc})'


# ---------------- 부분 검사 ----------------
def x_errors(job, x, dates, basket, basket_sha):
    """분할 몫: r1 X 검사 전부(메타·값·첫 주문일·전일 신호·명단 안 진입·현금 대사·폐지 뒤 체결 없음)를 200만 원 계좌로."""
    with split_account():
        return chk.x_errors(job, x, dates, basket, basket_sha)


def slot_errors(job, part, h, dates, basket, basket_sha, amounts):
    """보유 몫·작은 몫: r1 H 검사 전부(매수 규칙·수수료·현금 대사·수량 불감소)를 그 계좌 금액으로. 메타의 계좌 금액도 본다."""
    with cc.account(amounts['cash'], amounts['slot']):
        errs = [e.replace(job.prefix, f'{job.prefix} [{part}]', 1) for e in chk.h_errors(job, h, dates, basket, basket_sha)]
    m = h['meta']
    if m.get('part') != part or m.get('cash0') != amounts['cash'] or m.get('slot_cash') != amounts['slot']: errs.append(f'{job.prefix}: {part} meta amounts differ')
    return errs


def s_errors(job, out):
    """S = 보유 몫 + 분할 몫(일별 합), 시작 전날 1,000만 원 현금, 두 몫 사이 이체 없음(각 몫 현금은 r1 대사 검사가 따로 본다)."""
    s, p = out['s'], job.prefix
    try: want = cc.combine(out['core']['curve'], out['x']['curve'], out['x']['state'])
    except ValueError as exc: return [f'{p}: S cannot be rebuilt ({exc})']
    if not s.index.equals(want.index) or list(s.columns) != list(want.columns): return [f'{p}: S layout differs']
    errs = [] if np.allclose(s.to_numpy(float), want.to_numpy(float), rtol=0, atol=VALUE_TOL) else [f'{p}: S differs from core + split']
    if not np.isfinite(s.to_numpy(float)).all(): errs.append(f'{p}: S has missing values')
    first = s.iloc[0]
    if first.value != sb.CASH0 or first.cash != sb.CASH0 or first.core_cash != cc.CORE['cash'] or first.split_cash != cc.SPLIT_CASH:
        errs.append(f'{p}: S is not 8,000,000 + 2,000,000 in cash on the day before start')
    if not np.allclose(s.value, s.cash + s.stock_value + s.receivable, rtol=0, atol=VALUE_TOL): errs.append(f'{p}: S value != cash + stock + receivable')
    return errs


def c_errors(job, out):
    c, p = out['c'], job.prefix
    core, small = out['core']['curve'], out['small']['curve']
    if not (c.index.equals(core.index) and c.index.equals(small.index)): return [f'{p}: C dates differ']
    ok = (np.allclose(c.value, core.value + small.value, rtol=0, atol=VALUE_TOL) and np.allclose(c.cash, core.cash + small.cash, rtol=0, atol=VALUE_TOL)
          and np.allclose(c.stock_value, core.stock_value + small.stock_value, rtol=0, atol=VALUE_TOL))
    errs = [] if ok else [f'{p}: C differs from core + small']
    if c.value.iloc[0] != sb.CASH0: errs.append(f'{p}: C does not start with 10,000,000 in cash')
    return errs


def file_sha(path):
    h = hashlib.sha256(); h.update(open(path, 'rb').read()); return h.hexdigest()


def r1_reference(sums_path=os.path.join(cc.R1_FULL, 'SHA256SUMS')):
    """r1 본실행 SHA256SUMS: {'runs/파일 이름': 해시}. r1 산출물 자체는 열지 않는다."""
    out = {}
    for line in open(sums_path).read().splitlines():
        h, name = line.split(maxsplit=1)
        if name.startswith('./runs/'): out[name[2:]] = h
    return out


def hk_repro_errors(job, run_dir, ref):
    """H·K가 r1 본실행의 같은 작업 산출물과 바이트 단위로 같아야 한다(설계 6절: r1 H·K 재현)."""
    bad = [f for f in cc.H_FILES if ref.get(f'runs/{job.prefix}_{f}') != file_sha(os.path.join(run_dir, f'{job.prefix}_{f}'))]
    return [f'{job.prefix}: H/K differ from the r1 run ({", ".join(bad)})'] if bad else []


def job_errors(job, out, ctx):
    """ctx: {'dates','baskets','hashes','etf','run_dir','r1_ref'(정식만)}."""
    basket, sha, dates = ctx['baskets'][job.basket], ctx['hashes'][job.basket], ctx['dates']
    errs = slot_errors(job, 'core', out['core'], dates, basket, sha, cc.CORE)
    if job.kind == 'ctrl': return errs + slot_errors(job, 'small', out['small'], dates, basket, sha, cc.SMALL) + c_errors(job, out)
    errs += x_errors(job, out['x'], dates, basket, sha) + chk.h_errors(job, out['h'], dates, basket, sha)
    errs += chk.k_errors(job, out['k'], dates, ctx['etf'], chk.archived_kodex(job.basket) if job.k == 0 else None) + chk.first_day_errors(job, out)
    if ctx.get('r1_ref') is not None: errs += hk_repro_errors(job, ctx['run_dir'], ctx['r1_ref'])
    return errs + s_errors(job, out) + core_first_day_errors(job, out)


def core_first_day_errors(job, out):
    b = out['core']['buys']
    return [f'{job.prefix}: core first buy is not the start day'] if len(b) and b.date.min() != pd.Timestamp(job.start) else []


# ---------------- 작업 사이 ----------------
def pick(jobs, kind, basket, k, div, delist):
    m = jobs[(jobs.kind == kind) & (jobs.basket == basket) & (jobs.k == k) & (jobs['div'] == div) & (jobs.delist == delist)]
    return next(m.itertuples()) if len(m) == 1 else None


def cross_errors(jobs, outs):
    """주·대조 작업의 보유 몫이 같은지, 폐지 없는 바구니의 두 상폐 가정이 같은지, K가 배당·상폐 가정과 무관한지."""
    errs = [f'{j.prefix}: main job missing' for j in jobs[jobs.kind == 'ctrl'].itertuples() if pick(jobs, 'main', j.basket, j.k, j.div, j.delist) is None]
    for job in jobs[jobs.kind == 'main'].itertuples():
        ctrl = pick(jobs, 'ctrl', job.basket, job.k, job.div, job.delist)
        if ctrl is None: errs.append(f'{job.prefix}: control job missing'); continue
        a, b = outs[job.prefix]['core'], outs[ctrl.prefix]['core']
        if not (a['curve'].equals(b['curve']) and a['buys'].equals(b['buys'])): errs.append(f'{job.prefix}: core part differs between main and control jobs')
        if job.delist == 'zero':
            f = pick(jobs, 'main', job.basket, job.k, job.div, 'freeze')
            if f is not None and not outs[job.prefix]['h']['meta']['delisted'] and not outs[job.prefix]['s'].equals(outs[f.prefix]['s']):
                errs.append(f'{job.prefix}: zero-delisting S differs from freeze without any delisting')
    for (y, k), g in jobs[jobs.kind == 'main'].groupby(['basket', 'k']):
        curves = [outs[p]['k']['curve'] for p in g.prefix]
        if any(not c.equals(curves[0]) for c in curves): errs.append(f'basket {y} k={k}: K differs across dividend/delisting assumptions')
    return errs
