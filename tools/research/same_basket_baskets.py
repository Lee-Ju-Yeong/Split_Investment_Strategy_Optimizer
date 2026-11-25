"""같은 종목 대조(same-basket-r1) 명단 고정: 여섯 개설일의 상위 20종목을 기존 패널 두 개에서 뽑아 대조한 뒤 basket_YYYY.csv로 쓴다.

usage: python -B tools/research/same_basket_baskets.py [--out BASKET_DIR]
3월 실행 체크아웃을 작업 디렉터리로 실행(DB는 거래일 목록만 읽는다). 설계 2절:
- 개설일 D0 = 그해 첫 거래일, 기준일 = 직전 거래일(DB 달력).
- 후보군 패널(panel_k0.csv)과 2단계 패널(panel_v2.csv)에서 D0 행의 직전 신호일과 1~20위 종목코드를 순서대로 대조한다.
- 기준일 후보 수는 40·46·53·77·24·77이어야 한다(panel_k0.months.csv와 두 패널의 raw_candidates).
하나라도 다르면 아무 파일도 쓰지 않고 종료 코드 2로 끝난다. 이미 있는 명단 파일은 내용이 같을 때만 통과한다(덮어쓰지 않음).
"""
import argparse, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import stage2_common as sc
import same_basket_common as sb


def read_panel(path):
    return pd.read_csv(path, dtype={'ticker': str}, parse_dates=['trade_date', 'signal_date'])


def top_rows(panel, d0, name, errs):
    """D0 행의 (신호일, 후보 수, 순위순 상위 20 종목)."""
    g = panel[panel.trade_date == d0].sort_values('rank')
    if len(g) < sb.N_BASKET or list(g['rank'].iloc[:sb.N_BASKET]) != list(range(1, sb.N_BASKET + 1)):
        errs.append(f'{name} {d0.date()}: ranks 1..{sb.N_BASKET} not present'); return None, None, []
    if g.signal_date.nunique() != 1 or g.raw_candidates.nunique() != 1:
        errs.append(f'{name} {d0.date()}: signal date or candidate count not unique'); return None, None, []
    return g.signal_date.iloc[0], int(g.raw_candidates.iloc[0]), list(g.ticker.iloc[:sb.N_BASKET])


def month_count(months, d0, basis, errs):
    m = months[months.trade_date == d0]
    if len(m) != 1 or m.signal_date.iloc[0] != basis or m.status.iloc[0] != 'ok':
        errs.append(f'months {d0.date()}: row missing, not ok, or signal date differs'); return None
    return int(m.raw_candidates.iloc[0])


def basket_for(year, dates, pool, stage2, months, errs):
    d0 = sb.open_day(dates, year); basis = sb.prev_day(dates, d0)
    sig_p, n_p, top_p = top_rows(pool, d0, 'pool panel', errs)
    sig_s, n_s, top_s = top_rows(stage2, d0, 'stage2 panel', errs)
    n_m = month_count(months, d0, basis, errs)
    if (sig_p, sig_s) != (basis, basis): errs.append(f'{year}: panel signal dates {sig_p} {sig_s} != basis {basis.date()}')
    if {n_p, n_s, n_m} != {sb.EXPECTED_CANDIDATES[year]}: errs.append(f'{year}: candidate counts {n_p} {n_s} {n_m} != {sb.EXPECTED_CANDIDATES[year]}')
    if top_p != top_s: errs.append(f'{year}: top-{sb.N_BASKET} tickers differ between the two panels')
    return d0, basis, top_p


def basket_text(d0, basis, tickers, hashes):
    df = pd.DataFrame({'rank': range(1, len(tickers) + 1), 'ticker': tickers, 'basis_date': str(basis.date()), 'open_date': str(d0.date()),
                       'panel_sha256': hashes[0], 'stage2_panel_sha256': hashes[1]})
    return df[sb.BASKET_COLS].to_csv(index=False)


def write_once(path, text):
    """새 파일만 만든다. 이미 있으면 내용이 같아야 한다."""
    if os.path.exists(path):
        if open(path).read() != text: raise SystemExit(f'{path} exists with different content; not overwriting')
        return
    with open(path, 'x') as f: f.write(text)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=sb.BASKET_DIR); a = ap.parse_args()
    dates = sc.trading_dates(sc.connect())
    pool, stage2 = read_panel(sb.PANEL_POOL), read_panel(sb.PANEL_STAGE2)
    months = pd.read_csv(sb.PANEL_POOL.replace('.csv', '.months.csv'), parse_dates=['trade_date', 'signal_date'])
    hashes = (sb.sha256(sb.PANEL_POOL), sb.sha256(sb.PANEL_STAGE2)); errs = []
    found = {y: basket_for(y, dates, pool, stage2, months, errs) for y in sb.YEARS}
    if errs:
        print('\n'.join(['BASKET CHECK FAILED — nothing written'] + errs), file=sys.stderr); sys.exit(2)
    os.makedirs(a.out, exist_ok=True)
    for y, (d0, basis, tickers) in found.items():
        write_once(sb.basket_path(y, a.out), basket_text(d0, basis, tickers, hashes))
        print(y, 'basis', basis.date(), 'open', d0.date(), 'candidates', sb.EXPECTED_CANDIDATES[y], 'top20 match: yes')
    names = [os.path.basename(sb.basket_path(y, a.out)) for y in sb.YEARS]
    write_once(os.path.join(a.out, 'SHA256SUMS'), ''.join(f'{sb.sha256(os.path.join(a.out, n))}  {n}\n' for n in names))
    print('panel_sha256', hashes[0]); print('stage2_panel_sha256', hashes[1])


if __name__ == '__main__':
    main()
