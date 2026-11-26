"""주문율 격자(same-basket-size-r1) 산출물 읽기와 기술 검사(설계 5절). 성과 값을 출력하지 않는다. 모든 검사는 오류 문자열 목록을 돌려준다.

- X: r1 검사(same_basket_checks)의 메타·값·전일 신호·명단·현금 보존·상폐 검사를 그대로 쓰고, 주문율만 그 작업의 값으로 본다.
- 주문액 기록(엔진 기록): 매 거래일 기록, 엔진 시작일과 매달 첫 거래일에만 새로 정함, 새로 정한 값 = float32(basis) × float32(주문율),
  basis = 전일 마감 엔진 평가액 + 그날 아침 입금 배당(float32 오차 허용), 그 밖의 날은 직전 값 유지, 모든 매수 체결의 수량이
  기록된 체결가로 그날 주문액으로 살 수 있는 최대 정수 주인지 확인.
- r1 재사용: r1 정식 실행의 조립 검사 통과, 재사용 파일이 r1 SHA256SUMS와 같음, r1x 작업의 바이트 대조 기록.
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import same_basket_common as sb
import same_basket_checks as chk
import same_basket_size_common as sz

BASIS_ULPS = 4                     # float32 합산 순서 차이 허용(ulp)
X_FILES = ('curve.csv', 'state.csv', 'fills.csv', 'divs.csv', 'eligible.csv', 'endpos.csv', 'meta.json', 'sbmeta.json')
HK_FILES = ('H_curve.csv', 'H_pos.csv', 'H_buys.csv', 'H_divs.csv', 'H_meta.json', 'K_curve.csv', 'K_meta.json')


def read_orderlog(prefix):
    return pd.read_csv(prefix + '_orderamt.csv', parse_dates=['date'])


def read_buyprices(prefix):
    return pd.read_csv(prefix + '_buyprices.csv', dtype={'ticker': str}, parse_dates=['date'])


def size_x_errors(job, run, dates, basket, basket_sha):
    """r1 X 검사를 그대로 쓰되 주문율 기대값만 이 작업의 주문율로 바꾼다."""
    p, want = job.prefix, float(job.ratio)
    errs = [e for e in chk.x_meta_errors(job, run, dates, basket_sha) if not e.startswith(f'{p}: order_investment_ratio=')]
    got = run['sbmeta'].get('order_investment_ratio', np.nan)
    if not np.isclose(got, want, rtol=0, atol=1e-12): errs.append(f'{p}: order_investment_ratio={got} (expected {want})')
    errs += chk.x_value_errors(job, run, dates)
    if errs: return errs
    return chk.x_signal_errors(job, run, dates) + chk.x_basket_errors(job, run, basket) + chk.x_cash_errors(job, run) + chk.x_delist_errors(job, run)


def expected_recompute_days(days):
    """엔진 시작일(첫 행)과, 그 뒤 달이 바뀐 첫 거래일."""
    months = days.month
    return days[[0] + [i for i in range(1, len(days)) if months[i] != months[i - 1]]]


def f32_order(basis, ratio):
    return float(np.float32(np.float32(basis) * np.float32(ratio)))


def paid_on(divs, day):
    if divs.empty or 'event' not in divs: return 0.0
    p = divs[(divs.event == 'paid') & (pd.to_datetime(divs.date) == day)]
    return float(p.amount.sum())


def order_log_errors(job, run, log, dates):
    """엔진 기록으로 주문율이 실제 주문액에 반영됐는지 본다."""
    p, ratio = job.prefix, float(job.ratio)
    days = dates[(dates >= sb.prev_day(dates, job.start)) & (dates <= pd.Timestamp(job.end))]
    if not pd.DatetimeIndex(log.date).equals(days): return [f'{p}: order log dates differ from engine trading days']
    log = log.set_index('date'); errs = []
    if not log.index[log.recomputed.astype(bool)].equals(expected_recompute_days(days)): errs.append(f'{p}: order amount not recomputed exactly on month starts')
    curve = run['curve']
    for i, (d, r) in enumerate(log.iterrows()):
        if not r.recomputed:
            if r.investment_per_order != log.investment_per_order.iloc[i - 1]: errs.append(f'{p}: order amount changed mid-month on {d.date()}'); break
            continue
        if r.investment_per_order != f32_order(r.basis, ratio): errs.append(f'{p}: order amount != float32(basis) x ratio on {d.date()}'); break
        want = sz.CASH0 if i == 0 else float(curve.value_engine.iloc[i - 1]) + paid_on(run['divs'], d)
        if not abs(r.basis - want) <= BASIS_ULPS * float(np.spacing(np.float32(max(abs(want), 1.0)))):
            errs.append(f'{p}: order basis differs from prior close value + same-day dividend on {d.date()}'); break
    return errs + fill_size_errors(job, run, log.investment_per_order)


def fill_size_errors(job, run, ipo):
    """체결 장부와 체결가 기록을 대조하고 엔진과 같은 방식으로 정수 주 수를 계산한다."""
    fills, prices = run['fills'], run['buyprices']
    keys = ['date', 'ticker', 'dq']
    buys = fills.loc[fills.type == 'BUY', keys] if len(fills) else pd.DataFrame(columns=keys)
    if list(buys.itertuples(index=False, name=None)) != list(prices[keys].itertuples(index=False, name=None)):
        return [f'{job.prefix}: buy price log does not match buy fills']
    for r in prices.itertuples():
        cap = np.float32(ipo.loc[r.date])
        price = r.buy_price
        if not (np.isfinite(price) and price > 0 and price == int(price)
                and r.dq > 0 and r.dq == np.floor(cap / int(price))):
            return [f'{job.prefix}: buy of {r.ticker} on {pd.Timestamp(r.date).date()} is not sized by the order amount']
    return []


def r1cmp_errors(job, run_dir, formal):
    path = os.path.join(run_dir, job.prefix + '_r1cmp.json')
    if not os.path.exists(path): return [f'{job.prefix}: r1 comparison not recorded']
    c = json.load(open(path))
    errs = [] if c.get('passed') is True and c.get('r1_prefix') == job.r1prefix else [f'{job.prefix}: differs from r1 X {c.get("files")}']
    if formal and c.get('mode') != 'bytes': errs.append(f'{job.prefix}: r1 comparison is not a full byte comparison')
    return errs


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def r1_reuse_errors(jobs, r1_full=sz.R1_FULL, x_from_r1=True):
    """r1 정식 실행의 조립 검사가 통과했고, 재사용하는 r1 파일(H·K 전부, r1x의 X)이 r1 SHA256SUMS와 같은지."""
    cpath = os.path.join(r1_full, 'verdict', 'checks.json')
    if not os.path.exists(cpath) or json.load(open(cpath)).get('passed') is not True: return ['r1 formal checks did not pass']
    sums = {}
    for line in open(os.path.join(r1_full, 'SHA256SUMS')).read().splitlines():
        if line.strip(): h, n = line.split(None, 1); sums[n.strip()] = h
    want = set()
    for j in jobs.itertuples():
        want |= {f'{j.r1prefix}_{f}' for f in HK_FILES}
        if x_from_r1 and j.kind == 'r1x': want |= {f'{j.r1prefix}_{f}' for f in X_FILES}
    bad = [n for n in sorted(want) if sums.get(f'./runs/{n}') != sha_file(os.path.join(r1_full, 'runs', n))]
    return [f'r1 reused files differ from r1 SHA256SUMS: {len(bad)} (e.g. {bad[:3]})'] if bad else []


def truncate_hk(h, k, end):
    """smoke: r1 H·K를 작업 종료일에서 자른다(정식에서는 종료일이 같아 그대로)."""
    end = pd.Timestamp(end)
    return {**h, 'curve': h['curve'][h['curve'].index <= end]}, {**k, 'curve': k['curve'][k['curve'].index <= end]}


def hk_date_errors(job, x, h, k):
    idx = x['curve'].index
    ok = h['curve'].index.equals(idx) and k['curve'].index.equals(idx)
    return [] if ok else [f'{job.prefix}: reused r1 H/K dates differ from X dates']
