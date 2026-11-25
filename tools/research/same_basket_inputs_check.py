"""같은 종목 대조(same-basket-r1) 본실행 전 입력 점검: 계좌를 돌리지 않고 여섯 바구니의 입력만 확인한다.

usage: python -B tools/research/same_basket_inputs_check.py OUT_DIR
3월 실행 체크아웃을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행한다. DB는 SELECT만, 주가는 2025-12-30 이하만 읽는다.
계좌 계산·수익률·평가액·가격 수준을 만들거나 출력하지 않는다. 출력은 개수와 참/거짓, 문제가 있으면 종목코드·날짜·사유뿐이다.
- KODEX 파일: 각 시작일(k=0·5·20)부터 2025-12-30까지 모든 거래일에 유한한 양수 시가·종가가 있는지
- 보관 KODEX 곡선(curves_YYYY.csv): 날짜가 기본 2년 구간(k=0)의 거래일과 같은지
- 가격 행: 명단 종목을 엔진 DataHandler.load_stock_data로 직전 거래일~2025-12-30 적재(수정가격 NULL이면 엔진이 오류를 낸다),
  시작일별로 시가가 있는 종목 수, 기간 안 상장폐지 종목 수
- 배당 입력: 명단 20종목 모두를 권리일마다 ok / no_dividend / delisted / ratio_missing으로 분류한다. ratio_missing(배당이 있는데
  원주로 환산할 수 없음)은 막는 문제다(보유 중이면 X·H가 중단한다). 권리 확정 다음 거래일과 지급일의 위치, 직전 연도 대체 수
- 상장폐지: 폐지일(마지막 상장일) 뒤 가격 행이 있으면 막는 문제다(X 엔진이 그 행으로 거래·평가할 수 있으므로). 폐지일 당일 행은 허용한다
결과: OUT_DIR/inputs_check.json
"""
import json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import stage2_common as sc
import same_basket_common as sb
import same_basket_checks as chk
import same_basket_ledger as ledger


def etf_problems(etf, dates, year):
    out = []
    for k in sb.OFFSETS:
        days = dates[(dates >= sb.start_day(dates, year, k)) & (dates <= sb.cap_day(dates))]
        px = etf.reindex(days)[['open', 'close']].to_numpy(dtype=float)
        bad = days[~(np.isfinite(px).all(axis=1) & (px > 0).all(axis=1))]
        out += [{'basket': year, 'k': k, 'date': str(d.date()), 'reason': 'KODEX price missing or non-positive'} for d in bad[:5]]
    if etf.index.has_duplicates: out.append({'basket': year, 'reason': 'KODEX file has duplicate dates'})
    return out


def archived_problems(dates, year):
    days = dates[(dates >= sb.open_day(dates, year)) & (dates <= sb.main_end(dates, year))]
    idx = chk.archived_kodex(year).dropna().index
    return [] if idx.equals(days) else [{'basket': year, 'reason': f'archived KODEX curve dates differ from the 2-year trading days ({len(idx)} vs {len(days)})'}]


def rows_after_delisting(dh, t, dates, delisted):
    if t not in delisted: return 0
    df = dh.load_stock_data(t, str(dates[0].date()), str(dates[-1].date()))
    return 0 if df is None or df.empty else int((df.index > pd.Timestamp(delisted[t])).sum())


def price_check(dh, tickers, dates, year, delisted):
    """종목별 적재 오류와 시작일별 거래 가능 종목 수. 가격 값은 남기지 않는다."""
    problems, tradable, after = [], {k: 0 for k in sb.OFFSETS}, 0
    pos = {k: int(dates.searchsorted(sb.start_day(dates, year, k))) for k in sb.OFFSETS}
    for t in tickers:
        try: opens, closes = ledger.price_matrices(dh, [t], dates)
        except Exception as exc:
            problems.append({'basket': year, 'ticker': t, 'reason': f'price load failed: {type(exc).__name__}: {str(exc)[:160]}'}); continue
        if not np.isfinite(closes).any(): problems.append({'basket': year, 'ticker': t, 'reason': 'no price rows in the window'}); continue
        n_after = rows_after_delisting(dh, t, dates, delisted)
        if n_after: after += 1; problems.append({'basket': year, 'ticker': t, 'date': delisted[t], 'reason': f'{n_after} price rows after the delisting date'})
        for k, i in pos.items(): tradable[k] += int(ledger.tradable(opens[i, 0]))
        if not ledger.tradable(opens[pos[0], 0]): problems.append({'basket': year, 'ticker': t, 'date': str(dates[pos[0]].date()), 'reason': 'no open price on the opening day (H keeps this slot in cash)'})
    return problems, tradable, after


def dividend_check(conn, cal, tickers, dates, year, delisted):
    events = ledger.dividend_events(conn, cal, tickers, dates, 1, delisted)
    problems, rows = [], []
    for ev in events:
        per = ev['per_ticker']
        if sorted(per) != sorted(tickers): problems.append({'basket': year, 'date': ev['record_date'], 'reason': 'dividend inputs do not cover the basket'})
        for t, v in per.items():
            if v['status'] == 'ratio_missing': problems.append({'basket': year, 'ticker': t, 'date': ev['record_date'], 'reason': 'dividend with no adj_ratio (cannot convert to raw shares)'})
        status = {s: sum(v['status'] == s for v in per.values()) for s in ('ok', 'no_dividend', 'delisted', 'ratio_missing')}
        rows.append({'fy': ev['fy'], 'record_date': ev['record_date'], 'rights_day': str(dates[ev['rights_i']].date()),
                     'pay_day': None if ev['pay_i'] is None else str(dates[ev['pay_i']].date()), 'tickers_classified': len(per), **status,
                     'fallback_from_prior_year': sum(v['status'] == 'ok' and v['fallback'] for v in per.values())})
    return problems, rows


def basket_check(conn, cal, etf, baskets, year):
    dates = cal.dates; tickers = baskets[year]
    base = sb.prev_day(dates, sb.open_day(dates, year))
    days = dates[(dates >= base) & (dates <= sb.cap_day(dates))]
    dh, basis = ledger.engine_data_handler(str(base.date()), str(days[-1].date()))
    delist_i, delisted = ledger.delist_positions(conn, tickers, days)
    price_problems, tradable, after = price_check(dh, tickers, days, year, delisted)
    div_problems, events = dividend_check(conn, cal, tickers, days, year, delisted)
    main_end = sb.main_end(dates, year)
    summary = {'basis_day': str(base.date()), 'open_day': str(days[1].date()), 'main_end': str(main_end.date()), 'days_to_cap': len(days) - 1, 'price_basis': basis,
               'tradable_on_start': {f'k{k}': n for k, n in tradable.items()}, 'delisted_by_main_end': sum(pd.Timestamp(d) <= main_end for d in delisted.values()),
               'delisted_by_cap': len(delisted), 'delisted_with_rows_after': after, 'dividend_events': events}
    return summary, etf_problems(etf, dates, year) + archived_problems(dates, year) + price_problems + div_problems


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    conn = sc.connect(); cal = sc.Calendar(sc.trading_dates(conn)); baskets = sb.load_baskets(); etf = ledger.read_etf_capped(sb.ETF_CSV)
    result = {'cap': sb.END_CAP, 'baskets': {}, 'problems': []}
    for year in sb.YEARS:
        summary, problems = basket_check(conn, cal, etf, baskets, year)
        result['baskets'][year] = summary; result['problems'] += problems
        print(year, 'problems', len(problems), flush=True)
    blocking = [p for p in result['problems'] if 'keeps this slot in cash' not in p['reason']]
    result['passed'] = not blocking
    json.dump(result, open(os.path.join(out, 'inputs_check.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({'passed': result['passed'], 'problems': len(result['problems']), 'blocking': len(blocking)}))


if __name__ == '__main__':
    main()
