"""실행별 ETF 입력: data/etf의 069500·069500_quality·스타일 4종을 날짜 필드만 보고 CAP 이하 줄만 OUT_DIR로 복사한다(F4).

usage: python -I -B tools/research/pool_vs_index_etf_cap.py IN_DIR OUT_DIR CAP
CAP 뒤 줄의 가격 필드는 해석하지 않는다(문자열 날짜 비교만). OUT_DIR은 새 폴더여야 하고 etf_cap_manifest.json에 원본·결과 해시를 남긴다.
"""
import hashlib, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_vs_index_common import STYLE_ETFS, KODEX200, cap_date

FILES = [f'{KODEX200}.csv', f'{KODEX200}_quality.csv'] + [f'{c}.csv' for c in STYLE_ETFS]
ISO = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def copy_capped(src, dst, cap):
    kept = dropped = 0
    with open(src, newline='') as f, open(dst, 'x', newline='') as g:
        header = f.readline(); g.write(header); di = header.rstrip('\r\n').split(',').index('date')
        for line in f:
            day = line.split(',', di + 1)[di]
            if not ISO.match(day): raise SystemExit(f'bad date field in {src}')
            if day <= cap: g.write(line); kept += 1
            else: dropped += 1
    return {'source': src, 'source_sha256': sha256(src), 'file': dst, 'sha256': sha256(dst), 'rows': kept, 'rows_after_cap_not_parsed': dropped}


def main():
    src, out, cap = sys.argv[1], sys.argv[2], str(cap_date(sys.argv[3]).date())
    os.makedirs(out, exist_ok=False)
    man = {'cap': cap, 'files': {n: copy_capped(os.path.join(src, n), os.path.join(out, n), cap) for n in FILES}}
    json.dump(man, open(os.path.join(out, 'etf_cap_manifest.json'), 'x'), indent=1)
    print(json.dumps({n: {k: v for k, v in e.items() if k in ('rows', 'rows_after_cap_not_parsed')} for n, e in man['files'].items()}))


if __name__ == '__main__':
    main()
