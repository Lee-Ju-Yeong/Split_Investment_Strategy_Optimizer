"""주문율 격자(same-basket-size-r1) X 실행기: r1 X 실행기(same_basket_run.py)를 --cash 10000000 --ratio R로 부르고
엔진의 그달 주문액(investment_per_order)을 기록한다. r1 도구·제품 코드는 바꾸지 않는다(읽기만 하는 감싸기).

usage: python -B tools/research/same_basket_size_run.py START END OUT_PREFIX --basket FILE --ratio R --div approx|none|upper
       --delist freeze|zero [--r1 R1_PREFIX]
3월 실행 체크아웃을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행한다.
- same_basket_run.py 인자: START END OUT_PREFIX --basket FILE --warmup --delist Z --cash 10000000 --ratio R --tax period --div D --hold none
  (r1 주 작업과 같은 옵션에 시작 현금·주문율만 명시한다).
- OUT_PREFIX_orderamt.csv: 엔진이 _calculate_monthly_investment를 부른 날마다 date, recomputed(그날 새로 정했는지),
  basis(새로 정할 때 엔진이 쓴 평가액: 첫날은 시작 현금, 그 밖에는 Portfolio.get_total_value(전일)), investment_per_order.
  엔진은 매도 신호 생성 첫머리(배당 입금 뒤, 매도 전)에서 정하므로 basis = 전일 마감 평가액 + 그날 아침 입금 배당이다.
- OUT_PREFIX_buyprices.csv: 체결 장부와 일치하는 매수별 체결가. 수수료가 포함된 현금 변화에서 체결가를 역산할 때 생기는 오차를 피한다.
- --r1 R1_PREFIX: 2.2% 재현(r1x). 실행 뒤 X 산출물을 r1 산출물과 대조해 OUT_PREFIX_r1cmp.json을 쓴다. 종료일이 r1과 같으면
  파일 11개 바이트 일치(_meta.json은 argv만 빼고 같음), 짧으면(smoke) 곡선·상태·후보 기록·체결을 종료일까지 잘라 문자 그대로 대조한다.
  다르면 종료 코드 5.
표준 출력에 성과 값을 쓰지 않는다. 하위 실행기의 종료 코드(3: 전일 신호 미준비, 4: 명단 밖 매수)는 그대로 돌려준다.
"""
import argparse, filecmp, json, os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from src.backtest.cpu.portfolio import Portfolio
from src.backtest.cpu.strategy import MagicSplitStrategy
import same_basket_run as sbr

EXIT_R1_MISMATCH = 5
ORDER_COLS = ['date', 'recomputed', 'basis', 'investment_per_order']
BUY_COLS = ['date', 'ticker', 'dq', 'buy_price']
CMP_FILES = ('curve', 'fills', 'divs', 'state', 'endpos', 'unfilled', 'addsignals', 'holds', 'eligible', 'divinputs')
CUT_FILES = ('curve', 'state', 'eligible', 'fills')


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('prefix')
    ap.add_argument('--basket', required=True); ap.add_argument('--ratio', type=float, required=True)
    ap.add_argument('--div', choices=('approx', 'none', 'upper'), required=True); ap.add_argument('--delist', choices=('freeze', 'zero'), required=True)
    ap.add_argument('--r1')
    return ap


def run_argv(a):
    return [a.start, a.end, a.prefix, '--basket', a.basket, '--warmup', '--delist', a.delist, '--cash', '10000000', '--ratio', repr(float(a.ratio)),
            '--tax', 'period', '--div', a.div, '--hold', 'none']


def install_order_log(rows):
    """_calculate_monthly_investment를 감싸 그날의 주문액과, 새로 정한 날에는 엔진이 쓴 평가액을 남긴다(값을 바꾸지 않는다)."""
    calc0, total0 = MagicSplitStrategy._calculate_monthly_investment, Portfolio.get_total_value
    box = {'active': False, 'basis': None}

    def total(self, *args, **kwargs):
        v = total0(self, *args, **kwargs)
        if box['active']: box['basis'] = float(v)
        return v

    def calc(self, current_date, current_day_idx, trading_dates, portfolio, data_handler):
        before = self.previous_month
        box.update(active=True, basis=None)
        try: calc0(self, current_date, current_day_idx, trading_dates, portfolio, data_handler)
        finally: box['active'] = False
        changed = before != self.previous_month
        basis = (float(portfolio.initial_cash) if current_day_idx == 0 else box['basis']) if changed else None
        day = sbr.day_key(current_date)
        row = {'date': day, 'recomputed': changed, 'basis': basis, 'investment_per_order': float(self.investment_per_order)}
        if rows and rows[-1]['date'] == day:
            if changed: rows[-1].update(recomputed=True, basis=basis)
            rows[-1]['investment_per_order'] = row['investment_per_order']
        else: rows.append(row)
    MagicSplitStrategy._calculate_monthly_investment = calc
    Portfolio.get_total_value = total


def install_buy_log(rows):
    record = Portfolio.record_trade

    def record_trade(self, trade):
        record(self, trade)
        if trade.trade_type == 'BUY':
            rows.append((sbr.day_key(trade.date), trade.code, trade.quantity, trade.buy_price))

    Portfolio.record_trade = record_trade


def cut_frame(path, end):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[df.date <= end].reset_index(drop=True) if 'date' in df else df


def compare_r1(prefix, r1, end):
    """r1 X 산출물과 대조. 종료일이 같으면 바이트, 짧으면 종료일까지 자른 문자 대조."""
    r1_end = json.load(open(r1 + '_sbmeta.json'))['end']
    files = {}
    if end == r1_end:
        mode = 'bytes'
        for s in CMP_FILES: files[s] = filecmp.cmp(f'{prefix}_{s}.csv', f'{r1}_{s}.csv', shallow=False)
        files['sbmeta'] = filecmp.cmp(prefix + '_sbmeta.json', r1 + '_sbmeta.json', shallow=False)
        strip = lambda p: {k: v for k, v in json.load(open(p)).items() if k != 'argv'}
        files['meta_without_argv'] = strip(prefix + '_meta.json') == strip(r1 + '_meta.json')
    else:
        mode = 'truncated'
        for s in CUT_FILES: files[s] = cut_frame(f'{prefix}_{s}.csv', end).equals(cut_frame(f'{r1}_{s}.csv', end))
    return {'passed': all(files.values()), 'mode': mode, 'end': end, 'r1_end': r1_end, 'r1_prefix': os.path.basename(r1), 'files': files}


def main():
    a = parser().parse_args()
    rows, buys = [], []
    install_order_log(rows)
    install_buy_log(buys)
    sys.argv = [sbr.__file__] + run_argv(a)
    code = 0
    try: sbr.main()
    except SystemExit as exc: code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    pd.DataFrame(rows, columns=ORDER_COLS).to_csv(a.prefix + '_orderamt.csv', index=False)
    pd.DataFrame(buys, columns=BUY_COLS).to_csv(a.prefix + '_buyprices.csv', index=False)
    if code == 0 and a.r1:
        res = compare_r1(a.prefix, a.r1, a.end)
        json.dump(res, open(a.prefix + '_r1cmp.json', 'w'), indent=1)
        if not res['passed']:
            print(f'r1 mismatch: {[k for k, v in res["files"].items() if not v]}', file=sys.stderr); code = EXIT_R1_MISMATCH
    sys.exit(code)


if __name__ == '__main__':
    main()
