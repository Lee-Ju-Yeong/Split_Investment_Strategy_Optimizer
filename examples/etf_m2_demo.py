"""실제 M2 엔진의 1차 수동 진입·익절·자동 재진입 금지를 가상 3거래일로 검산한다."""
from datetime import date, datetime, time, timezone
from decimal import Decimal
import json

from src.research.etf.engine import run_engine
from src.research.etf.engine_types import Bar, Instrument, LevelRule


def run_demo():
    results = {}
    for path in ("O-H-L-C", "O-L-H-C"):
        bars = []
        for number, price in ((3, "1000"), (4, "1100"), (5, "900")):
            day, value = date(2024, 6, number), Decimal(price)
            opened = datetime.combine(day, time(0), timezone.utc)
            closed = datetime.combine(day, time(6), timezone.utc)
            bars.append(Bar("DEMO", day, opened, closed, value, value + 10, value - 10, value,
                            Decimal("100000"), adv20=Decimal("100000000")))
        rules = tuple(LevelRule(level, Decimal("10000"), Decimal(".1"), Decimal(".1")) for level in (1, 2))
        result = run_engine([Instrument("DEMO", "KR", "KRW", "국내주식형")], bars, {"DEMO": rules},
                            initial_cash={"KRW": Decimal("100000")}, path=path, commission_rate=Decimal("0"),
                            evaluation_times=[datetime(2024, 6, 5, 9, tzinfo=timezone.utc)])
        expected = [("BUY", 1, 10, Decimal("1000")), ("SELL", 1, 10, Decimal("1100"))]
        actual = [(row["side"], row["level"], row["quantity"], row["price"]) for row in result.trades]
        if actual != expected or result.positions or result.final_cash["KRW"] != Decimal("101000"):
            raise AssertionError("M2 합성 예제의 기대 거래·현금이 다릅니다")
        results[path] = {"trades": len(actual), "final_cash_krw": str(result.final_cash["KRW"]),
                         "positions": len(result.positions), "rejections": len(result.rejections)}
    return results


def main():
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
    print("M2 확인 완료: 가상 자료·수수료 0인 결정 규칙 검산")


if __name__ == "__main__":
    main()
