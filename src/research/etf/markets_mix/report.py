"""조립 결과의 보고용 표(summary.md). 판정 문장은 묶음별 주 시작일·기본 비용의 행에만 쓰고, 나머지는 병기다."""
from __future__ import annotations

from decimal import Decimal


STATUS = {"inferior": "열위", "counterexample": "반례", "tie": "동률", "path_dependent": "경로 의존", "blank": "빈칸"}
REGIME = {"down": "하락", "flat": "횡보", "up": "상승", None: "—"}
MARK = {"met": "충족", "not_met": "미충족", "undetermined": "미확정", "dependent": "의존", "same": "같음", "yes": "예",
        "no": "아니오"}
ORDER = ("inferior", "counterexample", "tie", "path_dependent", "blank")


def points(value) -> str:
    """연복리 차를 %p로(표기만 반올림한다. 판정은 반올림 전 값으로 했다)."""
    return "—" if value is None else f"{Decimal(value) * 100:+.3f}"


def _span(values) -> str:
    """경로들의 값 가운데 가장 작은 것과 큰 것."""
    found = [Decimal(value) for value in values if value is not None]
    if not found:
        return "—"
    low, high = min(found), max(found)
    return points(low) if low == high else f"{points(low)} ~ {points(high)}"


def _signs(row) -> str:
    return " ".join(sign or "·" for sign in row["signs"].values())


def _head(result) -> list:
    done, frame = result["conclusion"], result["frame"]
    counts = ", ".join(f"{STATUS[name]} {done['counts'][name]}" for name in ORDER)
    lines = ["# 다른 시장과 혼합 계좌 (markets-mix-r1) — 조립 결과", "",
             "탐색 실행(exploration=True, formal_run=False). 이미 본 과거 기간의 탐색이며 새 기간 검증이 아니다.", "",
             "## 결론", "", f"- **{done['sentence']}**", f"- 범위: {done['scope']}",
             f"- 행 {done['rows']}개: {counts}",
             *[f"- {row['cohort']} 묶음({row['start']} 시작) {row['rows']}행: "
               + ", ".join(f"{STATUS[name]} {row['counts'][name]}" for name in ORDER) for row in done["by_cohort"]],
             f"- 경로: {', '.join(frame['paths'])} (혼합은 두 몫의 경로 짝 네 가지)", ""]
    return lines + [f"- {note}" for note in done["limits"]] + [""]


def _counter_lines(done) -> list:
    if not done["counterexamples"]:
        return []
    counted = ", ".join(f"{row['start']} {row['subject']} {row['count']}개" for row in done["counterexample_settings"])
    lines = ["## 반례 행", "", f"- 시작일·대상별 반례 설정 수: {counted}", "",
             "| 대상 | 시작일 | 설정 | 같은 시작일 KODEX 200 보유보다 더 벌었나(경로별) | 자기 보유 구분(경로별) |",
             "|---|---|---|---|---|"]
    for row in done["counterexamples"]:
        beat = ", ".join("—" if value is None else "예" if value else "아니오" for value in row["beat_reference_hold"].values())
        own = ", ".join(REGIME[value] for value in row["own_hold_regime"].values())
        lines.append(f"| {row['subject']} | {row['start']} | {row['number']} | {beat} | {own} |")
    return lines + [""]


def _open_lines(done) -> list:
    if not done["left_open"]:
        return []
    lines = ["## 열위도 반례도 아닌 행(사유와 함께)", "", "| 대상 | 시작일 | 설정 | 상태 | 경로별 부호 | 사유 |",
             "|---|---|---|---|---|---|"]
    for row in done["left_open"]:
        reasons = sorted({reason for found in row["reasons"].values() for reason in found})
        signs = " ".join(sign or "·" for sign in row["signs"].values())
        lines.append(f"| {row['subject']} | {row['start']} | {row['number']} | {STATUS[row['status']]} | {signs} | "
                     f"{'; '.join(reasons) or '없음'} |")
    return lines + [""]


def _region_counts(table) -> str:
    found = [row["status"] for row in table]
    return " / ".join(f"{MARK[name]} {found.count(name)}" for name in ("met", "not_met", "undetermined"))


def _subject_lines(result) -> list:
    lines = ["## 대상별 요약 (병기)", "",
             "| 대상 | 종류 | 시작일 | 열위 | 반례 | 동률 | 경로 의존 | 빈칸 | 반례 설정 | 자기 보유 | 8점 영역(차 > 0) | 8점 영역(차 ≥ +1%p) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in result["subjects"]:
        counts = " | ".join(str(row["counts"][name]) for name in ORDER)
        own = ", ".join(REGIME[name] for name in row["own_hold_regimes"]) or "—"
        lines.append(f"| {row['subject']} | {row['kind']} | {row['start']} | {counts} | {row['counterexample_numbers'] or '없음'} | {own} | "
                     f"{_region_counts(row['regions']['edge_gt'])} | "
                     f"{_region_counts(row['regions']['review_base_edge_min'])} |")
    return lines + [""]


def _trades(row) -> str:
    found = [slot["trades"] for slot in row["paths"].values() if slot["trades"]]
    if not found:
        return "—"
    buys, sells = [item["buys"] for item in found], [item["sells"] for item in found]
    return f"{min(buys)}~{max(buys)} / {min(sells)}~{max(sells)}"


def _row_line(row, settings, periods) -> str:
    slots, setting = list(row["paths"].values()), settings[row["number"]]
    label = f"{setting['core']}%·kg {setting['kg']}·kp {setting['kp']}·{setting['levels']}차"
    segments = " | ".join(_span((slot["segments"] or {}).get(name) for slot in slots) for name in periods)
    exposure = _span(slot["strategy"]["exposure_mean"] if slot["strategy"] else None for slot in slots)
    return (f"| {row['number']} | {label} | {STATUS[row['status']]} | {_signs(row)} | "
            f"{_span(slot['delta_cagr'] for slot in slots)} | {_span(slot['vs_reference_hold'] for slot in slots)} | "
            f"{segments} | {_span(slot['mdd_deepening'] for slot in slots)} | {exposure} | {_trades(row)} |")


def _detail_lines(result) -> list:
    settings = {row["number"]: row for row in result["frame"]["settings"]}
    periods_of = {row["start"]: row["periods"] for row in result["frame"]["cohorts"]}
    lines = ["## 대상별 36행 (값은 경로들의 최소 ~ 최대, 단위 %p. 평균 비중은 %)", ""]
    for subject in result["subjects"]:
        periods = periods_of[subject["start"]]
        lines += [f"### {subject['subject']} ({subject['start']} 시작)", "",
                  "| 번호 | 설정 | 상태 | 경로별 부호 | 차(전체) | KODEX 200 보유 대비 | "
                  + " | ".join(f"차({name})" for name in periods) + " | 낙폭 심화 | 평균 비중 | 매수 / 매도 건수 |",
                  "|" + "---|" * (9 + len(periods))]
        lines += [_row_line(row, settings, periods) for row in result["rows"]
                  if (row["subject"], row["start"]) == (subject["subject"], subject["start"])]
        lines.append("")
    return lines


def _sensitivity_lines(result) -> list:
    lines = ["## 28번 민감도 (시작일·높은 비용. 주 판정을 바꾸지 않는다)", "",
             "| 대상 | 시작일 | 비용 | 상태 | 경로별 부호 | 차(전체) | 사유 |", "|---|---|---|---|---|---|---|"]
    for view in result["sensitivity"]:
        for row in view["rows"]:
            reasons = sorted({reason for slot in row["paths"].values() for reason in slot["reasons"]})
            lines.append(f"| {row['subject']} | {row['start']} | {row['cost']} | {STATUS[row['status']]} | {_signs(row)} | "
                         f"{_span(slot['delta_cagr'] for slot in row['paths'].values())} | {'; '.join(reasons) or '없음'} |")
    lines += ["", "| 대상 | 묶음 | 시작일 의존 | 비용 의존 | 재검토 문턱(시작일별) |", "|---|---|---|---|---|"]
    for view in result["sensitivity"]:
        review = ", ".join(f"{start} {MARK[mark]}" for start, mark in view["review_threshold"].items())
        lines.append(f"| {view['subject']} | {view['cohort']} | {MARK[view['start_dependent']]} | "
                     f"{MARK[view['cost_dependent']]} | {review} |")
    return lines + [""]


def markdown(result) -> str:
    """result: 직렬화한 조립 결과."""
    lines = (_head(result) + _counter_lines(result["conclusion"]) + _open_lines(result["conclusion"])
             + _subject_lines(result) + _sensitivity_lines(result) + _detail_lines(result))
    lines += ["## 실행 식별값", ""]
    lines += [f"- {key}: `{value}`" for key, value in result["identity"].items() if key != "inputs"]
    return "\n".join(lines) + "\n"
