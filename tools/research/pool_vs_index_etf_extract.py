"""스타일 ETF 4종을 etf-m5-r1 원본에서 END_CAP(2025-12-30) 이하 행만 뽑아 data/etf/<code>.csv로 쓴다(원본은 읽기만).

usage: python -I -B tools/research/pool_vs_index_etf_extract.py SRC_kr_daily.csv.gz OUT_DIR [--kodex-quality-only]
--verify-dir DIR: 추출 결과를 DIR의 기존 파일과 SHA-256으로 대조만 한다(쓰지 않음; F4 재추출 확인용).
--kodex-quality-only: 069500의 날짜별 ohlc_ok만 OUT_DIR/069500_quality.csv로 쓴다(v2.1 B: 가격은 기존 069500.csv, 품질 표시는 원본 연결).
END_CAP 이후 행은 걸러서 버리고 출력하지 않는다. 같은 이름 파일이 있으면 덮어쓰지 않고 멈춘다.
OUT_DIR/etf_extract_manifest.json에 원본·결과 SHA-256, 행 수, 기간, ohlc_ok=False 일수(월말 포함)를 남긴다.
"""
import csv, gzip, hashlib, io, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_vs_index_common import END_CAP, STYLE_ETFS, KODEX200

KEEP = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'dividend', 'split_ratio', 'ohlc_ok',
        'corp_action_ok', 'price_basis', 'listing_status']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load_capped(src, codes):
    """F4: 줄마다 코드·날짜 두 필드만 먼저 보고, 허용(코드 일치·END_CAP 이하) 줄만 모아 해석한다.
    허용되지 않은 줄의 가격 필드는 숫자로 해석하지 않는다."""
    keep, codes = [], set(codes)
    with gzip.open(src, 'rt', newline='') as f:
        r = csv.reader(f); header = next(r); si, di = header.index('symbol'), header.index('date')
        for row in r:
            if row[si] in codes and len(row[di]) == 10 and row[di] <= END_CAP: keep.append(row)
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(header); w.writerows(keep); buf.seek(0)
    return pd.read_csv(buf, dtype={'symbol': str, 'delisting_date': str})


def month_end_flags(g):
    d = pd.to_datetime(g.date)
    last = g.groupby([d.dt.year, d.dt.month]).tail(1)
    return int((~last.ohlc_ok.astype(bool)).sum())


def write_one(df, code, out_dir):
    path = os.path.join(out_dir, f'{code}.csv')
    if os.path.exists(path): raise SystemExit(f'refuse to overwrite {path}')
    g = df[df.symbol == code].sort_values('date')[KEEP]
    g.to_csv(path, index=False)
    return {'file': path, 'sha256': sha256(path), 'rows': len(g), 'first': g.date.iloc[0], 'last': g.date.iloc[-1],
            'ohlc_false_days': int((~g.ohlc_ok.astype(bool)).sum()), 'ohlc_false_month_ends': month_end_flags(g),
            'price_basis': sorted(g.price_basis.unique())}


def write_kodex_quality(src, out_dir):
    path = os.path.join(out_dir, f'{KODEX200}_quality.csv')
    if os.path.exists(path): raise SystemExit(f'refuse to overwrite {path}')
    g = load_capped(src, [KODEX200]).sort_values('date')[['date', 'ohlc_ok']]
    g.to_csv(path, index=False)
    man = {'source': src, 'source_sha256': sha256(src), 'end_cap': END_CAP, 'file': path, 'sha256': sha256(path), 'rows': len(g),
           'first': g.date.iloc[0], 'last': g.date.iloc[-1], 'ohlc_false_days': int((~g.ohlc_ok.astype(bool)).sum())}
    json.dump(man, open(os.path.join(out_dir, f'{KODEX200}_quality_manifest.json'), 'x'), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != 'file'}, indent=1))


def verify_against(src, scratch, ref_dir):
    df = load_capped(src, list(STYLE_ETFS) + [KODEX200])
    res = {}
    for c in STYLE_ETFS:
        res[c] = write_one(df, c, scratch)['sha256'] == sha256(os.path.join(ref_dir, f'{c}.csv'))
    q = df[df.symbol == KODEX200].sort_values('date')[['date', 'ohlc_ok']]
    qp = os.path.join(scratch, f'{KODEX200}_quality.csv'); q.to_csv(qp, index=False)
    res[f'{KODEX200}_quality'] = sha256(qp) == sha256(os.path.join(ref_dir, f'{KODEX200}_quality.csv'))
    print(json.dumps(res)); sys.exit(0 if all(res.values()) else 1)


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if '--kodex-quality-only' in sys.argv[3:]: return write_kodex_quality(src, out_dir)
    if '--verify-dir' in sys.argv[3:]: return verify_against(src, out_dir, sys.argv[sys.argv.index('--verify-dir') + 1])
    df = load_capped(src, list(STYLE_ETFS) + [KODEX200])
    assert pd.to_datetime(df.date).max() <= pd.Timestamp(END_CAP)
    k200 = df[df.symbol == KODEX200]
    man = {'source': src, 'source_sha256': sha256(src), 'end_cap': END_CAP, 'label': 'naver_adjusted_price_approximation',
           'etfs': {c: write_one(df, c, out_dir) for c in STYLE_ETFS},
           'kodex200_ohlc_false_days_in_source': int((~k200.ohlc_ok.astype(bool)).sum())}
    mpath = os.path.join(out_dir, 'etf_extract_manifest.json')
    if os.path.exists(mpath): raise SystemExit(f'refuse to overwrite {mpath}')
    json.dump(man, open(mpath, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({c: {k: v for k, v in e.items() if k != 'file'} for c, e in man['etfs'].items()}, indent=1))


if __name__ == '__main__':
    main()
