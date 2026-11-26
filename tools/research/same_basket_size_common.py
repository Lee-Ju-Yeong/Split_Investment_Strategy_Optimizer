"""전액 분할 계좌 주문율 격자(same-basket-size-r1) 공통: 상수·작업표·주문율 이름. DB·엔진을 읽지 않는 순수 모듈이다.

설계: runs/same-basket-size-r1/DESIGN-v1.md. 명단·기간·시작일·가정은 r1(same_basket_common)을 그대로 쓴다.
- 주문율 격자(설계 2절): 2.2%(r1 X 재사용), 3·4·5·6·8·10%. 결과를 보고 바꾸지 않는다.
- 2.2% 108작업(kind r1x)은 주문액 기록기만 붙여 다시 돌리고 r1 X와 바이트 일치를 강제한다(조정자 결정, 2026-11-25). 판정 값은 r1 X를 읽는다.
- 신규 6개 주문율 × 108 = 648작업(kind size). H·K는 r1 산출물(runs/same-basket-r1/full/runs/main_*)을 재사용한다.
"""
import os
import pandas as pd
import same_basket_common as sb

R = sb.R
RUN_DIR = f'{R}/runs/same-basket-size-r1'
R1_FULL = f'{R}/runs/same-basket-r1/full'
R1_RUNS = f'{R1_FULL}/runs'
RATIOS = (0.022, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)      # 격자 순서(경계 구간의 '연속'은 이 순서)
R1_RATIO = 0.022
CASH0 = sb.CASH0
JOB_COLS = ['id', 'kind', 'basket', 'k', 'div', 'delist', 'ratio', 'start', 'end', 'prefix', 'r1prefix']
N_JOBS = {'r1x': 108, 'size': 648}
SMOKE_RATIOS = (0.022, 0.03, 0.06, 0.10)


def ratio_name(r):
    """0.022 → r022, 0.1 → r100 (천분율 세 자리)."""
    return f'r{int(round(float(r) * 1000)):03d}'


def ratio_text(r):
    return f'{float(r) * 100:g}%'


def job_kind(r):
    return 'r1x' if abs(float(r) - R1_RATIO) < 1e-12 else 'size'


def rows_for(ratio, main_rows):
    """r1 주 작업 행(sb.main_rows 형식)을 주문율 작업 행으로 바꾼다. r1prefix는 같은 가정의 r1 작업 이름."""
    out = []
    for kind, y, k, d, z, _t, _r, s, e, pre in main_rows:
        out.append((job_kind(ratio), y, k, d, z, f'{float(ratio):g}', s, e, f'{ratio_name(ratio)}_b{y}_k{k}_{d}_{z}', pre))
    return out


def build_jobs(dates):
    """정식 작업표 756행: 2.2%(r1x) 108행이 먼저(불일치를 일찍 잡는다), 그다음 3%→10% 순. 주문율 안에서는 r1 작업표 순서."""
    rows = [r for ratio in RATIOS for y in sb.YEARS for r in rows_for(ratio, sb.main_rows(dates, y))]
    return jobs_frame(rows)


def build_smoke_jobs(dates, year, end, ratios=SMOKE_RATIOS):
    """smoke: 한 바구니 k=0의 6가정 × 주문율 몇 개, END에서 끝남. r1x 작업은 r1 X를 END에서 자른 값과 대조한다."""
    end = dates[dates <= pd.Timestamp(end)][-1]
    base = [r for r in sb.main_rows(dates, year) if r[2] == 0]
    base = [r[:8] + (end,) + r[9:] for r in base]
    return jobs_frame([r for ratio in ratios for r in rows_for(ratio, base)])


def jobs_frame(rows):
    df = pd.DataFrame(rows, columns=JOB_COLS[1:])
    for c in ('start', 'end'): df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    df.insert(0, 'id', range(1, len(df) + 1))
    return df


def read_jobs(path):
    return pd.read_csv(path, sep='\t', dtype=str).astype({'id': int, 'basket': int, 'k': int})


def compare_jobs(actual, expected):
    norm = lambda df: [tuple(str(v) for v in r) for r in df[JOB_COLS].itertuples(index=False)]
    a, e = norm(actual), norm(expected)
    if a == e: return []
    return ([f'job rows differ: {len(a)} actual vs {len(e)} expected'] + [f'unexpected job row {x}' for x in a if x not in e][:5]
            + [f'missing job row {x}' for x in e if x not in a][:5])


def check_job_structure(jobs):
    errs = [] if jobs.prefix.is_unique else ['duplicate prefix']
    for kind, n in N_JOBS.items():
        got = int((jobs.kind == kind).sum())
        if got != n: errs.append(f'{kind}: {got} jobs (need {n})')
    return errs


def x_source(job, run_dir):
    """판정에 쓰는 X 산출물 접두어: 2.2%는 r1 X, 그 밖은 이번 실행."""
    return os.path.join(R1_RUNS, job.r1prefix) if job.kind == 'r1x' else os.path.join(run_dir, job.prefix)
