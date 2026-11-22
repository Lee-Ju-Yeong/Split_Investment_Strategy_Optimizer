"""연속 계좌 진단(a)의 실행 목록(jobs.tsv)을 만든다. DB는 거래일 목록만 읽는다.

usage: python -B tools/research/continuous_account_jobs.py OUT_DIR [--offsets 0,5,20] [--end 2025-12-31]
       [--boundaries 2014-01-01,2016-01-01,...] [--no-bounds] [--no-anchor]
3월 실행 체크아웃을 작업 디렉터리로 실행(config/config.yaml의 DB 접속을 쓴다).
- 시작일 k: 첫 경계(2014-01-01) 이후 첫 거래일에서 k거래일 뒤.
- 연속 계좌(cont): 시작일 k ~ END를 한 번에 실행.
- 재시작 계좌(restart): 경계 b_j 이후 첫 거래일 + k거래일을 구간 시작으로, 다음 구간 시작 전 거래일(마지막은 END)을 구간 끝으로.
- 배당 처리: 주 분석 approx(모든 k), 하한 none·상한 upper(k=0만, --no-bounds로 끔). 세금은 모두 period.
- anchor: 3월 연속 실행 재현(2014-01-01~2023-12-31, tax fixed, div none). --no-anchor로 끔.
열: kind k window start end prefix tax div   (prefix는 OUT_DIR/runs/ 아래 상대 이름)
"""
import argparse, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import stage2_common as sc

BOUNDARIES = '2014-01-01,2016-01-01,2018-01-01,2020-01-01,2022-01-01,2024-01-01'
ANCHOR = ('anchor', 0, 'a2014', '2014-01-01', '2023-12-31', 'fixed', 'none')


def window_starts(dates, boundaries, k):
    return [dates[dates.searchsorted(pd.Timestamp(b)) + k] for b in boundaries]


def restart_windows(dates, boundaries, k, end):
    starts = window_starts(dates, boundaries, k)
    ends = [dates[dates.searchsorted(s) - 1] for s in starts[1:]] + [dates[dates <= pd.Timestamp(end)][-1]]
    return list(zip(boundaries, starts, ends))


def rows_for(dates, a, k):
    divs = ['approx'] + (['none', 'upper'] if k == 0 and not a.no_bounds else [])
    wins = restart_windows(dates, a.boundaries, k, a.end); out = []
    for div in divs:
        out.append(('cont', k, 'all', wins[0][1], wins[-1][2], 'period', div))
        out += [('restart', k, b.replace('-', ''), s, e, 'period', div) for b, s, e in wins]
    return out


def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir')
    ap.add_argument('--offsets', default='0,5,20'); ap.add_argument('--end', default='2025-12-31')
    ap.add_argument('--boundaries', default=BOUNDARIES); ap.add_argument('--no-bounds', action='store_true')
    ap.add_argument('--no-anchor', action='store_true')
    return ap


def build_jobs(dates, a):
    """정식 기대 작업표도 이 함수로 만든다(continuous_account_verdict.py가 기본 인자로 다시 생성해 대조)."""
    rows = [] if a.no_anchor else [ANCHOR]
    for k in map(int, a.offsets.split(',')):
        rows += rows_for(dates, a, k)
    df = pd.DataFrame(rows, columns=['kind', 'k', 'window', 'start', 'end', 'tax', 'div'])
    for c in ('start', 'end'):
        df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%d')
    df['prefix'] = [f'{r.kind}_k{r.k}_{r.window}_{r.tax}_{r.div}' for r in df.itertuples()]
    return df


def main():
    a = parser().parse_args(); a.boundaries = a.boundaries.split(',')
    df = build_jobs(sc.trading_dates(sc.connect()), a)
    os.makedirs(a.outdir, exist_ok=True)
    df.to_csv(os.path.join(a.outdir, 'jobs.tsv'), sep='\t', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
