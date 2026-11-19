"""정식 M5 결과표에 사전 지정한 선택 규칙·중첩 순차검증·PBO를 적용한다. 보류 판정·통계적 성공 판정 없음.

바스켓 값은 설정×바스켓마다 시작일들의 ΔCAGR·ΔMDD 중앙값이고 두 장중 경로 중 나쁜 쪽을 쓴다.
설정 점수는 두 값이 함께 양수인 바스켓의 비율이며, 자기 점수와 격자 한 칸 이웃 평균의 평균으로 고른다.
계획한 비교가 거부·실패한 바스켓은 개선하지 못한 것으로 센다.
"""
from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median

from .engine_inputs import _sha
from .m5_parameters import load_config
from .m5_search import FORMAL_PATH, SCHEMA as PLAN_SCHEMA, load_formal


SCHEMA = "etf-m5-selection-v1"
LOWEST = Decimal("-Infinity")


def load_table(plan, run_dir, *, allow_failed_units=False):
    """(바스켓, 시작일, 설정, 경로) → (ΔCAGR, ΔMDD). 계획한 비교가 통과하지 못했으면 None이다.

    빠진 단위·행이 빠진 단위는 거부한다. 실패한 단위는 명시적으로 허용할 때만 개선하지 못한 것으로 센다.
    """
    table, failed = {}, []
    expected = {(index, name) for index in range(len(plan["candidates"])) for name in plan["paths"]}
    for basket in plan["baskets"]:
        for start in basket["evaluations"]:
            path = Path(run_dir) / "units" / basket["id"] / f"{start}.json"
            if not path.is_file():
                raise ValueError(f"run is incomplete: {basket['id']} {start}")
            unit = json.loads(path.read_text(encoding="utf-8"))
            rows = {(row["candidate"], row["path"]): row for row in unit.get("trials", [])}
            if unit.get("status") == "failed":
                if not allow_failed_units:
                    raise ValueError(f"run has a failed unit: {basket['id']} {start}")
                failed.append([basket["id"], start])
            elif set(rows) != expected:
                raise ValueError(f"run is incomplete: {basket['id']} {start} lacks planned trials")
            for index in range(len(plan["candidates"])):
                for name in plan["paths"]:
                    comparison = rows.get((index, name), {}).get("comparison", {})
                    matched = comparison.get("matched") if comparison.get("status") == "passed" else None
                    table[(basket["id"], start, index, name)] = None if matched is None else (
                        Decimal(matched["delta_cagr"]), Decimal(matched["delta_mdd"]))
    return table, failed


def verify_run(plan_path, run_dir):
    """분석하는 계획·선택 규칙이 실행 당시와 같은지 확인한다. 다른 계획의 실행 폴더를 섞지 않는다."""
    pinned = json.loads((Path(run_dir) / "provenance.json").read_text(encoding="utf-8"))
    if pinned.get("plan_sha256") != _sha(Path(plan_path)):
        raise ValueError("run directory was produced by a different plan")
    if pinned.get("formal_sha256") != _sha(FORMAL_PATH):
        raise ValueError("formal configuration changed after the run")
    return pinned


def basket_value(table, basket, starts, index, paths):
    """시작일 중앙값을 경로마다 구하고 지표별로 나쁜 경로를 쓴다. 통과하지 못한 비교가 있으면 값이 없다."""
    values = []
    for name in paths:
        rows = [table[(basket, start, index, name)] for start in starts]
        if any(row is None for row in rows):
            return None
        values.append((median(row[0] for row in rows), median(row[1] for row in rows)))
    return min(value[0] for value in values), min(value[1] for value in values)


def setting_scores(table, plan, window):
    """구간에 계획한 평가가 있는 바스켓이 분모다. 값이 없는 바스켓은 개선하지 못한 것으로 센다."""
    members = [(basket["id"], [start for start in basket["evaluations"] if start in window])
               for basket in plan["baskets"]]
    members = [(name, starts) for name, starts in members if starts]
    if not members:
        raise ValueError("window has no planned evaluations")
    scores = []
    for index in range(len(plan["candidates"])):
        values = [basket_value(table, name, starts, index, plan["paths"]) for name, starts in members]
        known = [value for value in values if value is not None]
        improved = sum(value[0] > 0 and value[1] > 0 for value in known)
        scores.append({"share": Fraction(improved, len(members)), "improved": improved, "baskets": len(members),
                       "valued": len(known), "values": known})
    return scores


def neighbours(candidates, grid):
    """격자에서 한 파라미터만 한 칸 다른 설정들."""
    position = {json.dumps(candidate, sort_keys=True): index for index, candidate in enumerate(candidates)}
    result = []
    for candidate in candidates:
        near = []
        for key, values in grid.items():
            at = values.index(candidate[key])
            for step in (at - 1, at + 1):
                if 0 <= step < len(values):
                    other = position.get(json.dumps(dict(candidate, **{key: values[step]}), sort_keys=True))
                    if other is not None:
                        near.append(other)
        result.append(sorted(near))
    return result


def _lower_quartile(values):
    ordered = sorted(value[1] for value in values)
    return ordered[math.ceil(len(ordered) / 4) - 1] if ordered else LOWEST


def select(scores, near):
    """자기 점수와 이웃 평균의 평균이 가장 높은 설정. 동점이면 ΔMDD 하위 25% 값이 큰 쪽, 그다음 앞 번호.

    이웃 수가 적은 격자 가장자리 설정이 좋은 이웃 하나에 끌려 뽑히지 않게 자기 점수에 절반을 둔다.
    """
    smoothed = [(scores[index]["share"] + sum((scores[other]["share"] for other in near[index]), Fraction(0))
                 / len(near[index])) / 2 for index in range(len(scores))]
    best = max(range(len(scores)), key=lambda index: (smoothed[index], _lower_quartile(scores[index]["values"]), -index))
    return best, smoothed


def _rank(scores, index):
    """1이 가장 좋다. 동점은 평균 순위."""
    share = scores[index]["share"]
    above = sum(row["share"] > share for row in scores)
    equal = sum(row["share"] == share for row in scores)
    return above + (equal + 1) / 2


def _summary(score):
    values = score["values"]
    return {"joint_improvement_share": score["share"], "improved_baskets": score["improved"],
            "baskets": score["baskets"], "valued_baskets": score["valued"],
            "median_delta_cagr": median(value[0] for value in values) if values else None,
            "median_delta_mdd": median(value[1] for value in values) if values else None}


def _embargoed(train, test, embargo):
    """1·7월 실행은 6개월씩 겹친다. 평가 직전 시작일을 학습에서 뺀다."""
    before = sorted(start for start in train if start < min(test))
    return before[:len(before) - embargo] if embargo else before


def _year(start):
    return date.fromisoformat(start).year


def walk_forward(table, plan, near, rule):
    """바깥: 직전 3개 연도에서 고르고 다음 연도에서 본다. 안쪽 2년→1년 선택은 안정성 진단으로 함께 남긴다."""
    starts = sorted({start for basket in plan["baskets"] for start in basket["evaluations"]})
    folds, skipped = [], []
    for year in sorted({_year(start) for start in starts}):
        test = [start for start in starts if _year(start) == year]
        span = [start for start in starts if year - rule["outer_train_years"] <= _year(start) < year]
        if {_year(start) for start in span} != set(range(year - rule["outer_train_years"], year)):
            skipped.append({"test_year": year, "reason": "outer training years are not all available"})
            continue
        train = _embargoed(span, test, rule["embargo_starts"])
        scores = setting_scores(table, plan, set(train))
        chosen, smoothed = select(scores, near)
        outcome = setting_scores(table, plan, set(test))
        fold = {"test_year": year, "train_starts": train, "test_starts": test, "selected": chosen,
                "candidate": plan["candidates"][chosen], "train_smoothed_share": smoothed[chosen],
                "train": _summary(scores[chosen]), "test": _summary(outcome[chosen]),
                "test_rank": _rank(outcome, chosen), "settings": len(scores)}
        inner_test = [start for start in train if _year(start) == year - 1]
        inner_span = [start for start in train if _year(start) < year - 1]
        inner_train = _embargoed(inner_span, inner_test, rule["embargo_starts"]) if inner_test else []
        if inner_train:
            inner, _ = select(setting_scores(table, plan, set(inner_train)), near)
            fold["inner"] = {"train_starts": inner_train, "test_starts": inner_test, "selected": inner,
                             "same_as_outer": inner == chosen,
                             "test": _summary(setting_scores(table, plan, set(inner_test))[inner])}
        folds.append(fold)
    return folds, skipped


def _blocks(starts, count):
    size, extra = divmod(len(starts), count)
    blocks, at = [], 0
    for index in range(count):
        width = size + (index < extra)
        blocks.append(starts[at:at + width])
        at += width
    return blocks


def overfit_probability(table, plan, near, block_count):
    """CSCV. 시작일을 시간순 묶음으로 나눠 절반에서 고른 설정의 나머지 절반 상대 순위를 본다."""
    starts = sorted({start for basket in plan["baskets"] for start in basket["evaluations"]})
    if len(starts) < block_count or block_count % 2:
        return {"status": "not_computed", "reason": "too few evaluation starts or an odd block count",
                "starts": len(starts), "blocks": block_count}
    blocks = _blocks(starts, block_count)
    logits = []
    for chosen_blocks in combinations(range(block_count), block_count // 2):
        inside = {start for index in chosen_blocks for start in blocks[index]}
        outside = set(starts) - inside
        best, _ = select(setting_scores(table, plan, inside), near)
        scores = setting_scores(table, plan, outside)
        relative = 1 - (_rank(scores, best) - Fraction(1, 2)) / len(scores)
        logits.append(math.log(relative / (1 - relative)))
    return {"status": "computed", "blocks": [[block[0], block[-1], len(block)] for block in blocks],
            "splits": len(logits), "pbo": Fraction(sum(value <= 0 for value in logits), len(logits)),
            "median_logit": median(logits),
            "limit": "묶음 경계의 겹치는 1년 실행은 제거하지 않았다"}


def analyse(plan, run_dir, *, grid=None, allow_failed_units=False):
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid M5 plan")
    rule = load_formal()["selection_rule"]
    table, failed = load_table(plan, run_dir, allow_failed_units=allow_failed_units)
    near = neighbours(plan["candidates"], load_config()["grid"] if grid is None else grid)
    folds, skipped = walk_forward(table, plan, near, rule)
    planned = len(table)
    return {"schema": SCHEMA, "scope": plan["scope"], "synthetic": plan["synthetic"], "rule": rule,
            "comparisons_planned": planned, "comparisons_passed": sum(value is not None for value in table.values()),
            "failed_units": failed,
            "folds": folds, "skipped_folds": skipped,
            "overfit": overfit_probability(table, plan, near, rule["pbo_blocks"]),
            "statistical_success_evaluated": False, "holdout_evaluated": False, "actual_after_tax_approved": False}


def _encode(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="새 분석 JSON 경로")
    parser.add_argument("--allow-failed-units", action="store_true",
                        help="실패한 단위를 개선하지 못한 것으로 세고 보고서에 목록을 남긴다")
    args = parser.parse_args()
    pinned = verify_run(args.plan, args.run_dir)
    report = analyse(json.loads(args.plan.read_text(encoding="utf-8")), args.run_dir,
                     allow_failed_units=args.allow_failed_units)
    report.update(plan_sha256=_sha(args.plan), run_provenance_sha256=_sha(args.run_dir / "provenance.json"),
                  source_git_sha=pinned["source_git_sha"])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, default=_encode)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "folds": len(report["folds"]), "overfit": report["overfit"]},
                     ensure_ascii=False, indent=2, default=_encode))


if __name__ == "__main__":
    main()
