"""같은 종목 대조(same-basket-r1) 보조 시험(배정 방식): 종목 하나·50만 원 독립 계좌의 실행과 20개 합산.

usage: python -B tools/research/same_basket_aux.py START END OUT_PREFIX --basket FILE --ticker T --ratio r44|r10
3월 실행 체크아웃을 작업 디렉터리로, MAGICSPLIT_UNIVERSE_MODE=strict_pit로 실행한다. same_basket_run.py를 아래 옵션으로 부른다(설계 6절):
후보는 그 종목 하나(--only), 시작 현금 50만 원, 주문율은 월초 그 계좌 순자산의 44% 또는 10%, 배당 5월 지급, 상폐 동결.
잔액이 모자란 주문은 엔진이 체결하지 않는다. 합산(aggregate)은 조립기가 쓴다: 20개 계좌 평가액을 날짜별로 더한다.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import same_basket_common as sb


def run_argv(a):
    return [a.start, a.end, a.prefix, '--basket', a.basket, '--warmup', '--delist', 'freeze', '--only', a.ticker,
            '--cash', str(int(sb.SLOT_CASH)), '--ratio', str(sb.AUX_RATIOS[a.ratio]), '--tax', 'period', '--div', 'approx', '--hold', 'none']


def aggregate(outs):
    """outs: 한 바구니·한 주문율의 계좌 산출물 목록(순위순). 날짜별 평가액 합(미수 배당 포함)과 현금 합."""
    total = sum(o['x']['curve'].value for o in outs)
    cash = sum(o['x']['state'].cash for o in outs)
    return total, cash


def group_errors(name, jobs, outs, basket):
    """10절 7: 20개 계좌가 명단을 한 번씩 덮고, 날짜가 같고, 시작 전날 잔액 합이 원금 1,000만 원이다.
    계좌별 현금 대사는 same_basket_checks.x_cash_errors가 따로 본다."""
    errs = []
    if [j.ticker for j in jobs] != list(basket): errs.append(f'{name}: aux accounts do not cover the basket in rank order')
    if len(outs) != sb.N_BASKET: return errs + [f'{name}: {len(outs)} aux accounts (need {sb.N_BASKET})']
    first = outs[0]['x']['curve'].index
    if any(not o['x']['curve'].index.equals(first) for o in outs): return errs + [f'{name}: aux account dates differ']
    values = np.column_stack([o['x']['curve'].value.to_numpy(float) for o in outs])       # 날짜 × 계좌, 계좌마다 따로 읽은 값
    cashes = np.column_stack([o['x']['state'].cash.reindex(first).to_numpy(float) for o in outs])
    if not np.isfinite(values).all() or not np.isfinite(cashes).all(): return errs + [f'{name}: aux accounts have missing values']
    if (values[0] != sb.SLOT_CASH).any() or (cashes[0] != sb.SLOT_CASH).any(): errs.append(f'{name}: an aux account does not start with the slot cash')
    total, cash = aggregate(outs)
    if not (total.index.equals(first) and np.allclose(total.to_numpy(float), values.sum(axis=1), rtol=0, atol=1e-6)
            and np.allclose(cash.to_numpy(float), cashes.sum(axis=1), rtol=0, atol=1e-6)):
        errs.append(f'{name}: aggregated aux total differs from the independent per-date sum of the 20 accounts')
    if values[0].sum() != sb.CASH0: errs.append(f'{name}: aux balances do not sum to the principal before start')
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start'); ap.add_argument('end'); ap.add_argument('prefix'); ap.add_argument('--basket', required=True)
    ap.add_argument('--ticker', required=True); ap.add_argument('--ratio', choices=sorted(sb.AUX_RATIOS), required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    import same_basket_run
    sys.argv = [same_basket_run.__file__] + run_argv(a)
    same_basket_run.main()


if __name__ == '__main__':
    main()
