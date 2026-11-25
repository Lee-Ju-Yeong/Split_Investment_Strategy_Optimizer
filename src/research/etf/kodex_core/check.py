"""작은 독립 검산 도구: 첫 수량, 일별 합산, 판정 부등식만 본다.

R2 엔진·이 패키지의 다른 모듈을 import하지 않는다. 정확한 분수 산술로 다시 계산한다.
사용: python -m src.research.etf.kodex_core.check first --budget 20000000/9 --price 10000 --fee 0.00015
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json


def _q(value) -> Fraction:
    return Fraction(str(value))


def first_quantity(budget, price, fee_rate) -> dict:
    """수수료 포함 예산 안의 최대 정수 주: price*q*(1+fee) <= budget."""
    budget, price, fee_rate = _q(budget), _q(price), _q(fee_rate)
    quantity = int(budget // (price * (1 + fee_rate)))
    gross = price * quantity
    fee = gross * fee_rate
    return {"quantity": quantity, "fee": fee, "cost": gross + fee, "change": budget - gross - fee}


def account_value(row) -> Fraction:
    """현금 + 수량×종가 + 확정 미수금."""
    return _q(row["cash"]) + _q(row["quantity"]) * _q(row["price"]) + _q(row["receivable"])


def daily_sum(core_rows, split_rows) -> list[dict]:
    """두 장부를 같은 날짜끼리 더하고 ETF 투자 비중 = ETF 평가액 합 / 전체 순자산."""
    if len(core_rows) != len(split_rows):
        raise ValueError("ledgers must have the same evaluation days")
    output = []
    for core, split in zip(core_rows, split_rows):
        if core["at"] != split["at"]:
            raise ValueError("evaluation times differ")
        equity = account_value(core) + account_value(split)
        held = _q(core["quantity"]) * _q(core["price"]) + _q(split["quantity"]) * _q(split["price"])
        output.append({"at": core["at"], "equity": equity, "exposure": held / equity})
    return output


def compare_daily(core_rows, split_rows, *, tolerance=Fraction(1, 10**6)) -> list[str]:
    """엔진이 보고한 장부 평가액과 독립 합산이 다르면 날짜를 돌려준다."""
    problems = []
    for core, split, mine in zip(core_rows, split_rows, daily_sum(core_rows, split_rows)):
        reported = _q(core["equity"]) + _q(split["equity"])
        if abs(reported - mine["equity"]) > tolerance:
            problems.append(f"{mine['at']}: reported {reported} independent {mine['equity']}")
    return problems


def inequalities(metrics, thresholds) -> dict:
    """비중 >= 0.80, 연복리 차 >= 0.01, |MDD| 차 <= 0.05, 스트레스 연복리 차 > 0."""
    m, t = {k: _q(v) for k, v in metrics.items()}, {k: _q(v) for k, v in thresholds.items()}
    return {"exposure": m["exposure_mean"] >= t["exposure_min"],
            "cagr_edge": m["delta_cagr"] >= t["cagr_edge_min"],
            "mdd_excess": m["delta_abs_mdd"] <= t["mdd_excess_max"],
            "stress_cagr_edge": m["stress_delta_cagr"] > t["stress_cagr_edge_gt"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("first")
    for name in ("budget", "price", "fee"):
        first.add_argument("--" + name, required=True)
    judge = sub.add_parser("judge")
    judge.add_argument("--metrics", required=True, help="JSON: exposure_mean, delta_cagr, delta_abs_mdd, stress_delta_cagr")
    judge.add_argument("--thresholds", required=True, help="JSON")
    args = parser.parse_args(argv)
    if args.command == "first":
        result = first_quantity(args.budget, args.price, args.fee)
    else:
        result = inequalities(json.loads(args.metrics), json.loads(args.thresholds))
    print(json.dumps({k: str(v) for k, v in result.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
