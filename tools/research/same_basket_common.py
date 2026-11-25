"""같은 종목 대조(same-basket-r1) 공통: 상수·달력·명단·작업표·지표. DB·엔진을 읽지 않는 순수 모듈이다.

설계: runs/same-basket-r1/DESIGN-v1.1.md. 날짜 계산은 호출자가 넘긴 거래일 목록(DatetimeIndex)만 쓴다.
"""
import hashlib, math, os
import numpy as np, pandas as pd

R = '/mnt/volume/workspace/split-research'
RUN_DIR = f'{R}/runs/same-basket-r1'
BASKET_DIR = f'{RUN_DIR}/baskets'
PANEL_POOL = f'{R}/runs/tier1-pool-vs-index-r1/formal/panel/panel_k0.csv'      # 후보군 패널
PANEL_STAGE2 = f'{R}/runs/selection-vs-etf-r1/panel_v2.csv'                    # 2단계 패널
STAGE2_CURVES = f'{R}/runs/selection-vs-etf-r1/sim/curves_{{year}}.csv'          # 보관된 KODEX 곡선({year}_ETF 열)
ETF_CSV = f'{R}/data/etf/069500.csv'
END_CAP = '2025-12-30'                 # 주가 조회 상한(설계 9절)
YEARS = (2014, 2016, 2018, 2020, 2022, 2024)
EXPECTED_CANDIDATES = {2014: 40, 2016: 46, 2018: 53, 2020: 77, 2022: 24, 2024: 77}
N_BASKET = 20
OFFSETS = (0, 5, 20)
DIVS = ('approx', 'none', 'upper')     # 5월 지급 / 없음 / 권리 직후 지급
DELISTS = ('freeze', 'zero')           # 마지막 가격 동결 / 폐지일부터 0원
MAIN = (0, 'approx', 'freeze')         # 바구니별 주 값
CASH0 = 1e7
SLOT_CASH = 5e5                        # H의 종목당 한도, 보조 계좌 원금
COMMISSION = 0.00015
AUX_RATIOS = {'r44': 0.44, 'r10': 0.10}
BASKET_COLS = ['rank', 'ticker', 'basis_date', 'open_date', 'panel_sha256', 'stage2_panel_sha256']
JOB_COLS = ['id', 'kind', 'basket', 'k', 'div', 'delist', 'ticker', 'ratio', 'start', 'end', 'prefix']
N_JOBS = {'main': 108, 'ext': 6, 'aux': 240}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


# ---------------- 달력 ----------------
def cap_day(dates, cap=END_CAP):
    return dates[dates <= pd.Timestamp(cap)][-1]


def open_day(dates, year):
    """개설일 D0 = 그해 첫 거래일."""
    return dates[dates.searchsorted(pd.Timestamp(f'{year}-01-01'))]


def prev_day(dates, day):
    i = dates.searchsorted(pd.Timestamp(day))
    if i <= 0 or dates[i] != pd.Timestamp(day): raise ValueError(f'{day} is not a trading day with a predecessor')
    return dates[i - 1]


def start_day(dates, year, k):
    """시작일 = D0에서 k거래일 뒤."""
    return dates[dates.searchsorted(open_day(dates, year)) + k]


def main_end(dates, year, cap=END_CAP):
    """주 판정 종료일 = 개설 이듬해 마지막 거래일(상한 이하)."""
    return dates[dates <= min(pd.Timestamp(f'{year + 1}-12-31'), pd.Timestamp(cap))][-1]


# ---------------- 명단 ----------------
def basket_path(year, folder=BASKET_DIR):
    return os.path.join(folder, f'basket_{year}.csv')


def read_basket(path):
    b = pd.read_csv(path, dtype={'ticker': str, 'basis_date': str, 'open_date': str})
    if list(b.columns) != BASKET_COLS: raise ValueError(f'{path}: unexpected columns {list(b.columns)}')
    if list(b['rank']) != list(range(1, N_BASKET + 1)): raise ValueError(f'{path}: rank must be 1..{N_BASKET}')
    if b.ticker.duplicated().any() or not b.ticker.str.fullmatch(r'[0-9A-Z]{6}').all(): raise ValueError(f'{path}: bad or duplicated ticker')
    if b.basis_date.nunique() != 1 or b.open_date.nunique() != 1: raise ValueError(f'{path}: basis/open date not unique')
    return b


def load_baskets(folder=BASKET_DIR, years=YEARS):
    """{연도: 순위순 종목 목록}. 폴더의 SHA256SUMS와 파일 해시가 다르면 오류."""
    sums = dict(line.split()[::-1] for line in open(os.path.join(folder, 'SHA256SUMS')).read().splitlines() if line.strip())
    out = {}
    for y in years:
        p = basket_path(y, folder)
        if sums.get(os.path.basename(p)) != sha256(p): raise ValueError(f'{p}: hash differs from SHA256SUMS')
        out[y] = list(read_basket(p).ticker)
    return out


# ---------------- 작업표 ----------------
def job_prefix(kind, year, k, div, delist, rank=None, ticker=None, ratio=None):
    if kind == 'aux': return f'aux_b{year}_{ratio}_{rank:02d}_{ticker}'
    return f'{kind}_b{year}_k{k}_{div}_{delist}'


def main_rows(dates, year):
    """한 바구니의 18가지. 주 값(k=0·approx·freeze)이 맨 앞에 오도록 정렬한다."""
    combos = sorted(((k, d, z) for k in OFFSETS for d in DIVS for z in DELISTS), key=lambda c: c != MAIN)
    return [('main', year, k, d, z, '-', '-', start_day(dates, year, k), main_end(dates, year), job_prefix('main', year, k, d, z)) for k, d, z in combos]


def ext_row(dates, year):
    k, d, z = MAIN
    return ('ext', year, k, d, z, '-', '-', start_day(dates, year, k), cap_day(dates), job_prefix('ext', year, k, d, z))


def aux_rows(dates, year, tickers):
    k, d, z = MAIN
    return [('aux', year, k, d, z, t, name, start_day(dates, year, k), main_end(dates, year), job_prefix('aux', year, k, d, z, rank=i + 1, ticker=t, ratio=name))
            for name in AUX_RATIOS for i, t in enumerate(tickers)]


def build_jobs(dates, baskets):
    """정식 작업표(주 108 + 연장 6 + 보조 240). 조립기가 같은 함수로 기대 행을 다시 만들어 대조한다."""
    rows = [r for y in YEARS for r in main_rows(dates, y)] + [ext_row(dates, y) for y in YEARS]
    rows += [r for y in YEARS for r in aux_rows(dates, y, baskets[y])]
    return jobs_frame(rows)


def build_smoke_jobs(dates, baskets, year, cut, end):
    """smoke: 한 바구니 k=0만. 주 작업은 cut에서 끝나고(2년 종료일 역할) 연장 작업은 end까지 간다."""
    cut, end = dates[dates <= pd.Timestamp(cut)][-1], dates[dates <= pd.Timestamp(end)][-1]
    s = start_day(dates, year, 0); k, d0, z0 = MAIN
    rows = [('main', year, 0, d, z, '-', '-', s, cut, job_prefix('main', year, 0, d, z)) for d in DIVS for z in DELISTS]
    rows.sort(key=lambda r: (r[3], r[4]) != (d0, z0))
    rows.append(('ext', year, k, d0, z0, '-', '-', s, end, job_prefix('ext', year, k, d0, z0)))
    rows += [r[:8] + (cut,) + r[9:] for r in aux_rows(dates, year, baskets[year])]
    return jobs_frame(rows)


def jobs_frame(rows):
    df = pd.DataFrame(rows, columns=JOB_COLS[1:])
    for c in ('start', 'end'): df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    df.insert(0, 'id', range(1, len(df) + 1))
    return df


def read_jobs(path):
    return pd.read_csv(path, sep='\t', dtype=str).astype({'id': int, 'basket': int, 'k': int})


def compare_jobs(actual, expected):
    """작업표를 행 단위로 정확히 대조(순서 포함). 오류 문자열 목록."""
    norm = lambda df: [tuple(str(v) for v in r) for r in df[JOB_COLS].itertuples(index=False)]
    a, e = norm(actual), norm(expected)
    if a == e: return []
    errs = [f'job rows differ: {len(a)} actual vs {len(e)} expected']
    return errs + [f'unexpected job row {x}' for x in a if x not in e][:5] + [f'missing job row {x}' for x in e if x not in a][:5]


# ---------------- 지표 ----------------
def annualized(v_end, base_date, end_date, base=CASH0):
    """연복리(%): 시작일 직전 거래일 종가의 원금 → 종료일 종가, 연 365.25일."""
    yrs = (pd.Timestamp(end_date) - pd.Timestamp(base_date)).days / 365.25
    if not (yrs > 0 and base > 0 and v_end is not None and math.isfinite(v_end) and v_end > 0): return float('nan')
    return ((v_end / base) ** (1 / yrs) - 1) * 100


def max_drawdown(values, base=CASH0):
    """최대낙폭(%): 원금을 포함한 일별 평가액."""
    path = np.concatenate([[base], np.asarray(values, dtype=float)])
    return float((path / np.maximum.accumulate(path) - 1).min() * 100)
