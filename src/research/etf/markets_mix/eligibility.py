"""자격 집계(설계 v2 5절 4번, 규칙은 v1.1 5절). 장부를 열지 않고, 통과 여부와 사유의 종류만 낸다(수익률·가격 수준·한도 금액은 싣지 않는다).

상품 × 시작일 × 원금(단독 1억 원·몫 5천만 원)마다: 입력이 적재되는지(연초 변동성용 직전 251거래일 이력과 시작 전 20거래일
거래대금 창), 만기까지 일봉이 이어지는지, 시작일의 5% 한도가 그 원금 이상인지(같으면 허용. 기준 상품에는 적용하지 않는다).
실제 개설 주문과 개별 주문 한도는 장부가 실행 때 따로 본다.
"""
from __future__ import annotations

from pathlib import Path

from ..kodex_core import data as loader
from ..kodex_core import study
from ..kodex_core.inputs import DataHold
from ..kodex_grid.runner import _from
from ..kodex_grid.sealing import SEALED, OverwriteError, dump, with_seal, write_once, writer_lock
from .definition import SHARE, SOLO, Unit, unit_config
from .identity import MixContext, check_current, check_frozen, check_loaded, check_synthetic
from .permit import check_folder


SCHEMA = "markets-mix-eligibility-v2"
HOLD_CODES = (("volatility", "history_251_insufficient"), ("20 prior domestic sessions", "liquidity_window_insufficient"))
OTHER_HOLD, GAP, BELOW = "input_data_hold", "data_gap_after_start", "limit_below_account"


def hold_code(message: str) -> str:
    """자료 보류 문구를 종류로만 줄인다(문구에는 날짜·값이 들어갈 수 있어 공개 기록에 싣지 않는다)."""
    return next((code for marker, code in HOLD_CODES if marker in message), OTHER_HOLD)


def _load(ctx: MixContext, key):
    try:
        inputs = ctx.source.load(*key)
    except DataHold as error:
        return str(error)
    check_loaded(ctx, inputs)
    return inputs


def _limit_ok(ctx: MixContext, unit: Unit, loaded) -> bool:
    """시작일 시가 전 직전 20거래일 평균 거래대금의 5%가 그 계좌의 원금 이상인지(승인 코드의 한도 계산 그대로)."""
    config = unit_config(ctx.mix, unit, base=ctx.base, grid=ctx.spec)
    data = _from(loaded, unit.start)
    return study._capacity(data, config, ctx.mix.account(unit.kind))["ok"]


def unit_row(ctx: MixContext, unit: Unit, loaded) -> dict:
    """한 상품·시작일·원금의 자격: 통과 여부와 사유의 종류만."""
    row = {"symbol": unit.symbol, "start": unit.start.isoformat(), "kind": unit.kind,
           "account_krw": str(ctx.mix.account(unit.kind)), "floor_applies": ctx.mix.floor(unit) is not None}
    if isinstance(loaded, str):
        return {**row, "input_loaded": False, "bars_to_end": None, "limit_at_least_account": None,
                "eligible": False, "reasons": [hold_code(loaded)]}
    complete, limit = loaded.data_end is None, _limit_ok(ctx, unit, loaded)
    reasons = ([] if complete else [GAP]) + ([BELOW] if row["floor_applies"] and not limit else [])
    return {**row, "input_loaded": True, "bars_to_end": complete, "limit_at_least_account": limit,
            "eligible": not reasons, "reasons": reasons}


def _offsets_ok(ctx: MixContext, loaded) -> bool:
    """묶음마다 민감도 시작일이 주 시작일의 +5·+20 국내 거래일인지."""
    calendars = [item.calendar for item in loaded.values() if not isinstance(item, str)]
    if not calendars:
        return None
    try:
        for cohort in ctx.mix.cohorts:
            study.check_starts(calendars[0], cohort.starts(), (0, *ctx.mix.sensitivity["offsets"]))
    except ValueError:
        return False
    return True


def needed(ctx: MixContext) -> list:
    """자격을 볼 (종류, 상품, 시작일): 묶음의 단독 대상과 기준 보유(단독 1억 원), 혼합 몫(5천만 원)."""
    found = []
    for cohort in ctx.mix.cohorts:
        solo = list(dict.fromkeys((*cohort.solo, ctx.mix.reference)))
        for start in cohort.starts():
            found += [(SOLO, symbol, start) for symbol in solo] + [(SHARE, symbol, start) for symbol in cohort.legs()]
    return found


def _subject_rows(ctx: MixContext, rows) -> list:
    """판정 대상(단독·혼합) × 시작일의 자격. 한 몫이라도 자격이 없으면 그 혼합의 그 시작일 행은 빈칸이 된다."""
    eligible = {(row["kind"], row["symbol"], row["start"]): row["eligible"] for row in rows}
    found = []
    for cohort in ctx.mix.cohorts:
        for start in cohort.starts():
            day, primary = start.isoformat(), start == cohort.start
            found += [{"cohort": cohort.id, "subject": symbol, "kind": "solo", "start": day, "primary": primary,
                       "legs": None, "eligible": eligible[(SOLO, symbol, day)]} for symbol in cohort.solo]
            found += [{"cohort": cohort.id, "subject": mix.id, "kind": "mix", "start": day, "primary": primary,
                       "legs": {leg: eligible[(SHARE, leg, day)] for leg in mix.legs},
                       "eligible": all(eligible[(SHARE, leg, day)] for leg in mix.legs)} for mix in cohort.mixes]
    return found


def _blank_rows(ctx: MixContext, subjects) -> dict:
    """자격 단계에서 빈칸이 되는 판정 행 수: 주 판정(36개 설정)과 28번 민감도(두 비용)."""
    settings, costs = len(ctx.spec.settings()), 2
    primary = sum(settings for row in subjects if row["primary"] and not row["eligible"])
    sensitivity = [row for row in subjects if not row["eligible"]]
    return {"primary_rows": primary, "primary_rows_planned": ctx.mix.rows_expected["total"],
            "sensitivity_rows": costs * len(sensitivity) - sum(row["primary"] for row in sensitivity),
            "ineligible": [f"{row['subject']}@{row['start']}" for row in sensitivity]}


def collect(ctx: MixContext) -> tuple:
    """(공개 기록의 본문, 봉인할 보류 문구). 입력만 읽고 장부는 열지 않는다."""
    number, cost = ctx.mix.sensitivity["number"], ctx.mix.costs["primary"]
    keys = needed(ctx)
    loader._source.cache_clear()
    try:
        loaded = {(symbol, start): _load(ctx, (symbol, start)) for symbol, start in dict.fromkeys((s, d) for _, s, d in keys)}
        rule = start_rule(ctx)
    finally:
        loader._source.cache_clear()
    rows = [unit_row(ctx, Unit(kind, symbol, start, number, cost), loaded[(symbol, start)]) for kind, symbol, start in keys]
    sealed = {f"{symbol}@{start.isoformat()}": item for (symbol, start), item in loaded.items() if isinstance(item, str)}
    subjects = _subject_rows(ctx, rows)
    body = {"units": rows, "subjects": subjects, "blank_rows": _blank_rows(ctx, subjects), "start_rules": rule,
            "sensitivity_starts_are_offsets": _offsets_ok(ctx, loaded),
            "eligible_units": sum(row["eligible"] for row in rows), "checked_units": len(rows)}
    return body, sealed


def _calendar(ctx: MixContext, start):
    """국내 거래일 달력: 기준 상품(KODEX 200)을 그 시작일로 적재한 입력의 달력(등록 달력 파일 그대로)."""
    return ctx.source.load(ctx.mix.reference, start).calendar


def _year_firsts(calendar, since: int, until) -> tuple:
    """since년부터 until의 해까지 각 해 첫 국내 거래일과, 거래일이 하나도 없는 해(빠뜨리지 않고 남긴다)."""
    years = range(since, until.year + 1)
    days = {year: calendar.first_session_of_year(year) for year in years}
    return {year: day for year, day in days.items() if day is not None}, [year for year, day in days.items() if day is None]


LATER_ONLY = (GAP,)  # 시작일 선택에 쓰지 않는 사유: 개설 뒤에야 알 수 있는 일봉 결측


def _known_at_open(ctx: MixContext, key):
    """시작일 시가에 알 수 있는 입력(그날까지의 일봉). 그 입력기가 없는 합성 입력기는 전체 적재를 쓴다."""
    load = getattr(ctx.source, "load_until", None)
    if load is None:
        return _load(ctx, key)
    try:
        inputs = load(*key)
    except DataHold as error:
        return str(error)
    check_loaded(ctx, inputs)
    return inputs


def _rows_for(ctx: MixContext, rule, loaded) -> list:
    """loaded: (시작일, 적재 결과)."""
    (day, inputs), number, cost = loaded, ctx.mix.sensitivity["number"], ctx.mix.costs["primary"]
    return [unit_row(ctx, Unit(kind, rule["symbol"], day, number, cost), inputs) for kind in rule["accounts"]]


def _to_end(ctx: MixContext, rule, day) -> dict:
    """만기까지의 적재(시작일 판정에는 쓰지 않고 따로 남긴다): 일봉이 이어지는지와 사유의 종류."""
    rows = _rows_for(ctx, rule, (day, _load(ctx, (rule["symbol"], day))))
    reasons = sorted({reason for row in rows for reason in row["reasons"] if reason in LATER_ONLY or not row["input_loaded"]})
    complete = all(row["input_loaded"] and row["bars_to_end"] for row in rows)
    return {"bars_to_end": complete, "later_reasons": [] if complete else reasons}


def _rule_day(ctx: MixContext, rule, day) -> dict:
    """그 연초 거래일에 시작했다면, 시가에 알 수 있는 251일 이력과 두 원금 한도를 채우는지. 만기까지의 적재 오류는
    따로 남긴다."""
    rows = _rows_for(ctx, rule, (day, _known_at_open(ctx, (rule["symbol"], day))))
    known = sorted({reason for row in rows for reason in row["reasons"] if reason not in LATER_ONLY})
    loaded_ok = all(row["input_loaded"] for row in rows)
    eligible = loaded_ok and all(row["limit_at_least_account"] for row in rows) and not known
    return {"start": day.isoformat(), "eligible": eligible, "reasons": known, **_to_end(ctx, rule, day)}


def _cohort_rule(ctx: MixContext, cohort, rule) -> dict:
    firsts, missing = _year_firsts(_calendar(ctx, cohort.start), rule["search_from_year"], cohort.start)
    first = firsts.get(cohort.start.year) == cohort.start
    earlier = [_rule_day(ctx, rule, day) for year, day in sorted(firsts.items()) if year < cohort.start.year]
    chosen = {**_rule_day(ctx, rule, cohort.start), "first_session_of_year": first}
    holds = first and not missing and chosen["eligible"] and not any(row["eligible"] for row in earlier)
    return {"cohort": cohort.id, "symbol": rule["symbol"], "earlier": earlier, "chosen": chosen,
            "missing_years": missing, "holds": holds}


def start_rule(ctx: MixContext) -> list:
    """규칙으로 정한 주 시작일: 그해 첫 국내 거래일이고 조건을 채우며, 찾기 시작한 해부터 그 전까지 모든 해의 첫 거래일은
    조건을 채우지 못해야 한다. 거래일이 없는 해가 있으면 규칙 실패다."""
    return [_cohort_rule(ctx, cohort, cohort.start_rule) for cohort in ctx.mix.cohorts
            if isinstance(cohort.start_rule, dict)]


def run_eligibility(ctx: MixContext, out) -> dict:
    """자격 집계를 out/eligibility/에 한 번만 쓴다. 실제 묶음이면 고정 정의만 쓴다(본실행 허가는 필요 없다: 장부가 없다)."""
    check_folder(out)
    check_current(ctx)
    (check_frozen if ctx.registered else check_synthetic)(ctx)
    folder = Path(out) / "eligibility"
    with writer_lock(out):
        if (folder / "record.json").exists():
            raise OverwriteError("an eligibility record is already here; use a new output folder")
        body, sealed = collect(ctx)
        record = with_seal({"schema": SCHEMA, "exploration": True, "formal_run": False, "identity": ctx.identity,
                            "performance_reported": False, **body})
        write_once(folder / "sealed" / "hold-messages.json", dump(sealed), mode=SEALED)
        write_once(folder / "record.json", dump(record))
        write_once(folder / "report.md", markdown(record))
    return record


def _mark(value) -> str:
    return "—" if value is None else "통과" if value else "미달"


def markdown(record) -> str:
    lines = ["# 자격 집계 (통과 여부와 사유의 종류만. 수익률·가격·한도 금액 없음)", "",
             f"- 자격 있는 (상품, 시작일, 원금): {record['eligible_units']} / {record['checked_units']}",
             f"- 민감도 시작일이 주 시작일의 +5·+20 국내 거래일: {_mark(record['sensitivity_starts_are_offsets'])}",
             f"- 자격 단계에서 빈칸이 되는 주 판정 행: {record['blank_rows']['primary_rows']} / "
             f"{record['blank_rows']['primary_rows_planned']}, 민감도 행: {record['blank_rows']['sensitivity_rows']} "
             f"(자격 없는 대상·시작일: {', '.join(record['blank_rows']['ineligible']) or '없음'})", "",
             *_rule_lines(record["start_rules"]),
             "| 상품 | 시작일 | 계좌 | 원금 | 입력 적재(251일·20일 창) | 만기까지 일봉 | 5% 한도 ≥ 원금 | 한도 적용 | 자격 | 사유 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in record["units"]:
        lines.append(f"| {row['symbol']} | {row['start']} | {row['kind']} | {row['account_krw']} | "
                     f"{_mark(row['input_loaded'])} | {_mark(row['bars_to_end'])} | "
                     f"{_mark(row['limit_at_least_account'])} | {'예' if row['floor_applies'] else '아니오'} | "
                     f"{_mark(row['eligible'])} | {', '.join(row['reasons']) or '없음'} |")
    lines += ["", "| 묶음 | 판정 대상 | 시작일 | 주 판정 시작일 | 몫별 자격 | 자격 |", "|---|---|---|---|---|---|"]
    for row in record["subjects"]:
        legs = "—" if row["legs"] is None else ", ".join(f"{leg} {_mark(ok)}" for leg, ok in row["legs"].items())
        lines.append(f"| {row['cohort']} | {row['subject']} | {row['start']} | {'예' if row['primary'] else '아니오'} | "
                     f"{legs} | {_mark(row['eligible'])} |")
    return "\n".join(lines) + "\n"


def _rule_lines(rules) -> list:
    lines = []
    for rule in rules:
        earlier = ", ".join(f"{row['start']} {_mark(row['eligible'])}" for row in rule["earlier"])
        chosen = rule["chosen"]
        lines += [f"- {rule['cohort']} 묶음 시작일 규칙({rule['symbol']}: 그때 알 수 있는 251일 이력과 두 원금 한도를 채우는 첫 "
                  f"연초 거래일): {chosen['start']} {_mark(chosen['eligible'])}(그해 첫 거래일 {_mark(chosen['first_session_of_year'])}, "
                  f"만기까지 일봉 {_mark(chosen['bars_to_end'])} — 따로 남김), 그 전 연초 거래일 {earlier or '없음'}, "
                  f"거래일 없는 해 {rule['missing_years'] or '없음'} → 규칙 성립 {_mark(rule['holds'])}", ""]
    return lines
