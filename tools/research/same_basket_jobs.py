"""같은 종목 대조(same-basket-r1) 작업표(jobs.tsv)를 만든다. DB는 거래일 목록만 읽는다.

usage: python -B tools/research/same_basket_jobs.py OUT_DIR [--baskets DIR] [--smoke YEAR CUT END] [--verify]
3월 실행 체크아웃을 작업 디렉터리로 실행한다.
- 정식: 주 108(바구니 6 × 시작일 3 × 배당 3 × 상폐 2, 개설 이듬해 마지막 거래일까지) + 연장 6(k=0·approx·freeze, 2025-12-30까지)
  + 보조 240(바구니 6 × 주문율 2 × 종목 20, 기본 2년).
- --smoke YEAR CUT END: 그 바구니의 k=0만. 주·보조 작업은 CUT에서, 연장 작업은 END에서 끝난다(코드 경로 확인용).
- --verify: 쓰지 않고, 이미 있는 OUT_DIR/jobs.tsv가 다시 만든 작업표와 행 단위로 같은지만 본다(다르면 종료 코드 2).
열: id kind basket k div delist ticker ratio start end prefix (start는 등록한 시작일, prefix는 OUT_DIR/runs/ 아래 이름)
"""
import argparse, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_common as sc
import same_basket_common as sb


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('outdir'); ap.add_argument('--baskets', default=sb.BASKET_DIR)
    ap.add_argument('--smoke', nargs=3, metavar=('YEAR', 'CUT', 'END')); ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    dates = sc.trading_dates(sc.connect()); baskets = sb.load_baskets(a.baskets)
    df = sb.build_smoke_jobs(dates, baskets, int(a.smoke[0]), a.smoke[1], a.smoke[2]) if a.smoke else sb.build_jobs(dates, baskets)
    path = os.path.join(a.outdir, 'jobs.tsv')
    if a.verify:
        errs = sb.compare_jobs(sb.read_jobs(path), df)
        print('\n'.join(errs) if errs else f'jobs.tsv matches the expected table ({len(df)} rows)'); sys.exit(2 if errs else 0)
    os.makedirs(a.outdir, exist_ok=True)
    if os.path.exists(path): raise SystemExit(f'{path} already exists; not overwriting')
    df.to_csv(path, sep='\t', index=False)
    print(df.groupby('kind').size().to_string()); print('rows', len(df), 'sha256', sb.sha256(path))


if __name__ == '__main__':
    main()
