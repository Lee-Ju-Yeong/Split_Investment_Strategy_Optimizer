"""주문율 격자(same-basket-size-r1) 작업표(jobs.tsv)를 만든다. DB는 거래일 목록만 읽는다.

usage: python -B tools/research/same_basket_size_jobs.py OUT_DIR [--smoke YEAR END] [--verify]
3월 실행 체크아웃을 작업 디렉터리로 실행한다.
- 정식: 2.2%(r1x) 108 + 신규 6개 주문율 × 108 = 756행. 가정·기간은 r1 주 작업과 같다.
- --smoke YEAR END: 그 바구니 k=0의 6가정 × 주문율 2.2·3·6·10%, END에서 끝남(기술 검사용).
- --verify: 쓰지 않고 OUT_DIR/jobs.tsv가 다시 만든 작업표와 행 단위로 같은지만 본다(다르면 종료 코드 2).
열: id kind basket k div delist ratio start end prefix r1prefix
"""
import argparse, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_common as sc
import same_basket_common as sb
import same_basket_size_common as sz


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir')
    ap.add_argument('--smoke', nargs=2, metavar=('YEAR', 'END')); ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    dates = sc.trading_dates(sc.connect())
    df = sz.build_smoke_jobs(dates, int(a.smoke[0]), a.smoke[1]) if a.smoke else sz.build_jobs(dates)
    path = os.path.join(a.outdir, 'jobs.tsv')
    if a.verify:
        errs = sz.compare_jobs(sz.read_jobs(path), df)
        print('\n'.join(errs) if errs else f'jobs.tsv matches the expected table ({len(df)} rows)'); sys.exit(2 if errs else 0)
    os.makedirs(a.outdir, exist_ok=True)
    if os.path.exists(path): raise SystemExit(f'{path} already exists; not overwriting')
    df.to_csv(path, sep='\t', index=False)
    print(df.groupby('kind').size().to_string()); print('rows', len(df), 'sha256', sb.sha256(path))


if __name__ == '__main__':
    main()
