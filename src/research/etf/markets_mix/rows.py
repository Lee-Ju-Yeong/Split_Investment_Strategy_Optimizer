"""봉인 단위에서 판정 행을 만든다(설계 v2, 규칙은 v1.1 4·6절). 한 행 = (대상, 설정 번호, 시작일, 비용).

단독 행은 1억 원 단위의 두 경로를, 혼합 행은 두 5천만 원 몫의 경로 짝 네 가지를 가진다. 한 몫이 비교 불가이면 그 몫을 쓰는
혼합 행의 해당 짝은 빈칸이고 사유를 남긴다. 병기 항목(KODEX 200 보유 대비, 자기 보유 구분, 구간 차, 낙폭, 비중, 체결 건수)은
판정에 쓰지 않는다.
"""
from __future__ import annotations

from decimal import Decimal

from ..kodex_grid.reading import dec, exact_sub
from ..kodex_grid.records import AssemblyError
from . import judge
from .definition import SHARE, SOLO, Unit, period_labels, reference_unit
from .identity import MixContext
from .mixing import Frame, account_rows, metrics, moment, pair_keys, rows_of
from .unit import REASON_KEYS


SOLO_ROW, MIX_ROW = "solo", "mix"
SHOWN = ("cagr", "mdd", "exposure_mean", "ending_equity_krw")


def row_key(subject, slot) -> str:
    """slot: (설정 번호, 시작일, 비용)."""
    number, start, cost = slot
    return f"{subject}/{number:02d}@{start.isoformat()}/{cost}"


def reasons_of(result, symbol=None) -> dict:
    """경로별 사유 목록. 적재하지 못한 단위는 모든 경로에 같은 사유를 가진다. symbol을 주면 사유 앞에 붙인다."""
    prefix = "" if symbol is None else f"{symbol}:"
    if result.get("held_reason"):
        return {None: [prefix + result["held_reason"]]}
    return {path: [prefix + reason for key in REASON_KEYS for reason in row["reasons"][key]]
            for path, row in result["paths"].items()}


def _blank(reasons) -> dict:
    return {"comparable": False, "reasons": list(reasons), "delta_cagr": None, "strategy": None, "hold": None,
            "mdd_deepening": None, "segments": None, "trades": None}


def _segments(strategy, hold, labels) -> dict:
    """구간별 '전략 연복리 − 보유 연복리'. 이름은 정의의 표기 이름으로 바꾼다."""
    pairs = list(zip(strategy["segments"], hold["segments"], strict=True))
    if [(s["segment"], h["segment"]) for s, h in pairs] != [(fixed, fixed) for fixed, _ in labels]:
        raise AssemblyError("segments of an account differ from the declared periods")
    return {shown: exact_sub(s["cagr"], h["cagr"]) for (s, h), (_, shown) in zip(pairs, labels)}


def _valued(strategy, hold, labels, *, trades) -> dict:
    return {"comparable": True, "reasons": [], "delta_cagr": exact_sub(strategy["cagr"], hold["cagr"]),
            "strategy": {key: dec(strategy[key]) for key in SHOWN}, "hold": {key: dec(hold[key]) for key in SHOWN},
            "mdd_deepening": exact_sub(dec(strategy["mdd"]).copy_abs(), dec(hold["mdd"]).copy_abs()),
            "segments": _segments(strategy, hold, labels), "trades": trades}


def _trades(rows) -> dict:
    """분할 장부의 매수·매도 체결 건수(몫이 둘이면 합과 몫별 값)."""
    total = {side: sum(row[side] for _, row in rows) for side in ("buys", "sells")}
    return total if len(rows) == 1 else {**total, "by_leg": {symbol: dict(row) for symbol, row in rows}}


def _frame(ctx: MixContext, records) -> Frame:
    """혼합 계좌의 지표 틀: 원금은 몫의 합, 첫 시가 시각은 모든 몫이 같아야 한다."""
    opened = {record["result"]["head"]["opened_at"] for record in records}
    if len(opened) != 1:
        raise AssemblyError("shares of a mix must open at the same time")
    initial = sum(Decimal(record["result"]["capital"]["hold"]) for record in records)
    return Frame(initial, moment(opened.pop()), ctx.base.segments, ctx.base.thresholds["exposure_min"])


def solo_slots(result, labels) -> dict:
    """단독 행의 경로별 자리."""
    if result.get("held_reason"):
        return None
    slots = {}
    for path, row in result["paths"].items():
        trades = {side: row["split_trades"][side] for side in ("buys", "sells")} if row["comparable"] else None
        slots[path] = (_valued(row["strategy"], row["hold"], labels, trades=trades) if row["comparable"]
                       else _blank(reasons_of(result)[path]))
    return slots


def _leg_reasons(legs, pair) -> list:
    """그 경로 짝에서 비교 불가인 몫의 사유(상품 표시와 함께)."""
    reasons = []
    for (symbol, record), path in zip(legs, pair):
        result = record["result"]
        found = reasons_of(result, symbol)
        if result.get("held_reason"):
            reasons += found[None]
        elif not result["paths"][path]["comparable"]:
            reasons += found[path]
    return reasons


def _mixed(ctx: MixContext, legs, pair, *, labels) -> dict:
    """모든 몫이 비교 가능한 한 경로 짝: 일별 계열을 더해 혼합 전략·혼합 보유의 지표를 구한다."""
    rows = [(symbol, record["result"]["paths"][path]) for (symbol, record), path in zip(legs, pair)]
    frame = _frame(ctx, [record for _, record in legs])
    series = [(symbol, row["series"]) for symbol, row in rows]
    strategy = metrics(account_rows(series, "strategy"), frame)
    hold = metrics(account_rows(series, "hold"), frame)
    return _valued(strategy, hold, labels, trades=_trades([(symbol, row["split_trades"]) for symbol, row in rows]))


def mix_slots(ctx: MixContext, legs, labels) -> dict:
    """혼합 행의 경로 짝별 자리. legs: ((상품, 그 몫의 단위 기록), ...)."""
    slots = {}
    for key, pair in pair_keys(ctx.base.paths, len(legs)):
        reasons = _leg_reasons(legs, pair)
        slots[key] = _blank(reasons) if reasons else _mixed(ctx, legs, pair, labels=labels)
    return slots


def _claims(row) -> list:
    """한 경로에서 봉인 계열로 다시 구해 맞출 (계열 이름, 봉인한 지표): 보유 계열은 병기 기준(hold_only)과 늘, 비교할 수
    있는 경로는 전략·보유 지표와도. 전략만 비교할 수 없는 경로에서도 보유는 맞춘다."""
    claims = [("hold", row["hold_only"])]
    return claims + ([("strategy", row["strategy"]), ("hold", row["hold"])] if row["comparable"] else [])


def _hold_expected(row, gap) -> bool:
    """보유 주문이 시작일에 체결됐고 만기까지 관측이 이어지면, 전략이 비어도 보유 계열과 그 지표가 있어야 한다."""
    return gap is None and row["start_orders"]["orders"]["hold"]["status"] == "filled"


def _path_reproduced(row, frame, unit) -> bool:
    """unit: (상품, 관측 종료 표시)."""
    (symbol, gap), series = unit, row["series"]
    expected = _hold_expected(row, gap)
    if series is None or row["hold_only"] is None:
        return series is None and row["hold_only"] is None and not row["comparable"] and not expected
    if not expected:
        return False
    return all(_plain(metrics(rows_of(series["at"], series[name], symbol), frame)) == _plain(claimed)
               for name, claimed in _claims(row))


def reproduced(ctx: MixContext, record) -> bool:
    """봉인한 일별 계열로 지표를 다시 구하면 봉인한 지표와 같은지(혼합 합산과 KODEX 200 병기 기준이 쓰는 값의 검산)."""
    result = record["result"]
    if result.get("held_reason"):
        return True
    frame, symbol = _frame(ctx, [record]), result["head"]["symbol"]
    gap = result["head"]["data_end"]
    return all(_path_reproduced(row, frame, (symbol, gap)) for row in result["paths"].values())


def _plain(value):
    """지표의 값 비교용 꼴: 수치는 Decimal 값으로, 구간은 이름과 값으로."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    try:
        return Decimal(value)
    except ArithmeticError:
        return value


class Table:
    """봉인 단위 기록에서 행을 만드는 표. records: {단위: 기록}."""

    def __init__(self, ctx: MixContext, records):
        self.ctx, self.records = ctx, records

    def record(self, kind, symbol, slot) -> dict:
        return self.records[Unit(kind, symbol, slot[1], slot[0], slot[2])]

    def reference(self, slot) -> dict:
        """같은 시작일·비용의 기준 상품(KODEX 200) 1억 원 보유 연복리(경로별). 보유 장부가 유효할 때만 있다.
        보유 장부는 분할 설정과 무관하므로 그 시작일·비용의 28번 단독 단위가 싣는 보유를 쓴다."""
        result = self.records[reference_unit(self.ctx.mix, slot[1], slot[2])]["result"]
        if result.get("held_reason"):
            return {}
        return {path: dec(row["hold_only"]["cagr"]) for path, row in result["paths"].items() if row["hold_only"]}

    def slots(self, subject, kind, slot) -> dict:
        labels = period_labels(self.ctx.mix, self.ctx.base, slot[1])
        if kind == SOLO_ROW:
            return solo_slots(self.record(SOLO, subject, slot)["result"], labels)
        legs = [(symbol, self.record(SHARE, symbol, slot)) for symbol in subject.legs]
        return mix_slots(self.ctx, legs, labels)

    def required(self, kind, subject) -> tuple:
        paths = self.ctx.base.paths
        return tuple(paths) if kind == SOLO_ROW else tuple(key for key, _ in pair_keys(paths, len(subject.legs)))

    def row(self, subject, kind, slot) -> dict:
        """slot: (설정 번호, 시작일, 비용). 단독의 subject는 상품 코드, 혼합의 subject는 Mix."""
        number, start, cost = slot
        name = subject if kind == SOLO_ROW else subject.id
        required = self.required(kind, subject)
        slots = self.slots(subject, kind, slot)
        if slots is None:  # 단독 단위를 적재하지 못함: 모든 경로가 같은 사유의 빈칸
            held = reasons_of(self.record(SOLO, subject, slot)["result"])[None]
            slots = {path: _blank(held) for path in required}
        _annotate(slots, self.reference(slot), self.ctx.mix.thresholds["regime_band"])
        verdict = judge.row_verdict({path: slots[path]["delta_cagr"] for path in required}, required)
        return {"key": row_key(name, slot), "subject": name, "kind": kind, "number": number,
                "start": start.isoformat(), "cost": cost, **verdict, "paths": slots}


def _annotate(slots, reference, band) -> None:
    """병기(판정 아님): 같은 시작일 KODEX 200 1억 원 보유 대비 연복리 차와 자기 보유 연복리 구분."""
    for key, slot in slots.items():
        first = key.split("|")[0]  # 혼합은 첫 몫의 경로에 해당하는 기준 보유와 견준다(보유는 경로와 무관해야 한다)
        other = reference.get(first)
        valued = slot["comparable"]
        slot["vs_reference_hold"] = exact_sub(slot["strategy"]["cagr"], other) if valued and other is not None else None
        slot["regime"] = judge.regime(slot["hold"]["cagr"], band) if valued else None
