"""same-basket-r2 장부 실행기: r1 H 장부(same_basket_ledger)를 그대로 써서 H·K(r1과 같은 규칙·같은 파일), 보유 몫(40만 원 × 20),
대조 C의 작은 몫(10만 원 × 20)을 만들고, 주 작업이면 분할 몫(X 산출물)과 합친 S 곡선을 쓴다.

usage: python -B tools/research/same_basket_core_ledger.py START END OUT_PREFIX --basket FILE --div approx|none|upper --delist freeze|zero
       --kind main|ctrl [--etf CSV] [--hk-only]
3월 실행 체크아웃을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행한다. DB는 읽기만 하고 주가는 END(≤ 2025-12-30)까지만 읽는다.
입력(가격·배당 사건·폐지일)은 r1 same_basket_ledger.load_inputs로 한 번만 읽고 세 장부가 함께 쓴다(장부는 입력을 바꾸지 않는다).
- main: OUT_PREFIX_H_*·_K_*(r1 same_basket_ledger.py main()과 같은 계산·같은 메타 → r1 본실행 산출물과 바이트가 같아야 한다),
        OUT_PREFIX_core_H_*(보유 몫: 시작일 시가, 종목당 40만 원 한도, 수수료 포함 최대 정수 주, 못 산 자리는 현금, 매도 없음, 배당 재투자 = r1 H 규칙),
        OUT_PREFIX_S_curve.csv(보유 몫 + 분할 몫 일별 합. 분할 몫은 같은 OUT_PREFIX의 X 산출물 _curve·_state를 읽는다. --hk-only면 H·K만 쓴다).
- ctrl: OUT_PREFIX_core_H_*(보유 몫, 주 작업과 같은 계산), OUT_PREFIX_small_H_*(종목당 10만 원 단순 보유), OUT_PREFIX_C_curve.csv(두 몫의 일별 합).
표준 출력에는 아무것도 쓰지 않는다(성과 값 없음).
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import same_basket_common as sb
import same_basket_ledger as ledger
import same_basket_core_common as cc

C_COLS = ['date', 'core_value', 'small_value', 'value', 'stock_value', 'cash', 'receivable']


def base_meta(a, inp, info):
    """r1 same_basket_ledger.main()이 H 메타에 넣는 값과 같은 순서·같은 값."""
    return {**info, 'account_start': inp['dates'][1], 'end': inp['dates'][-1], 'div': a.div, 'delist': a.delist,
            'basket_file': os.path.basename(a.basket), 'basket_sha256': sb.sha256(a.basket)}


def write_h_and_k(a, inp, dates, info):
    """r1 same_basket_ledger.main()의 H·K 부분과 같은 계산·같은 쓰기."""
    state = ledger.run_hold(inp, a.div, a.delist)
    ledger.write_hold(a.prefix, state, base_meta(a, inp, info))
    k, shares = ledger.kodex_curve(ledger.read_etf_capped(a.etf), dates[1:])
    out = pd.concat([pd.Series([sb.CASH0], index=dates[:1]), k]).rename('value')
    out.to_csv(a.prefix + '_K_curve.csv', index_label='date')
    json.dump({'etf_file': os.path.basename(a.etf), 'etf_sha256': sb.sha256(a.etf), 'shares': shares, 'account_start': inp['dates'][1], 'end': inp['dates'][-1]},
              open(a.prefix + '_K_meta.json', 'w'), indent=1)


def run_slot_ledger(a, inp, info, part, amounts):
    """r1 H 장부를 계좌 금액만 바꿔 돌리고 OUT_PREFIX_<part>_H_*로 쓴다. 곡선(DataFrame)을 돌려준다."""
    with cc.account(amounts['cash'], amounts['slot']):
        state = ledger.run_hold(inp, a.div, a.delist)
    ledger.write_hold(f'{a.prefix}_{part}', state, {**base_meta(a, inp, info), 'part': part, 'cash0': amounts['cash'], 'slot_cash': amounts['slot']})
    return pd.DataFrame(state['curve']).assign(date=lambda d: pd.to_datetime(d.date)).set_index('date')


def write_s(prefix, core):
    """S = 보유 몫 + 분할 몫. 분할 몫은 r1 X 실행기가 같은 OUT_PREFIX로 쓴 _curve·_state."""
    x = pd.read_csv(prefix + '_curve.csv', parse_dates=['date']).set_index('date')
    st = pd.read_csv(prefix + '_state.csv', parse_dates=['date']).set_index('date')
    cc.combine(core, x, st).to_csv(prefix + '_S_curve.csv')


def write_c(prefix, core, small):
    if not core.index.equals(small.index): raise ValueError('core and small dates differ')
    c = pd.DataFrame({'core_value': core.value, 'small_value': small.value, 'value': core.value + small.value, 'stock_value': core.stock_value + small.stock_value,
                      'cash': core.cash + small.cash, 'receivable': core.receivable + small.receivable}, index=core.index)
    c.index.name = 'date'
    c.to_csv(prefix + '_C_curve.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('prefix'); ap.add_argument('--basket', required=True)
    ap.add_argument('--div', choices=sb.DIVS, required=True); ap.add_argument('--delist', choices=sb.DELISTS, required=True)
    ap.add_argument('--kind', choices=sorted(cc.N_JOBS), required=True); ap.add_argument('--etf', default=sb.ETF_CSV)
    ap.add_argument('--hk-only', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    inp, dates, info = ledger.load_inputs(a.basket, a.start, a.end)
    if a.kind == 'main':
        write_h_and_k(a, inp, dates, info)
        if a.hk_only: return
        write_s(a.prefix, run_slot_ledger(a, inp, info, 'core', cc.CORE))
    else:
        if a.hk_only: raise SystemExit('--hk-only is for main jobs')
        write_c(a.prefix, run_slot_ledger(a, inp, info, 'core', cc.CORE), run_slot_ledger(a, inp, info, 'small', cc.SMALL))


if __name__ == '__main__':
    main()
