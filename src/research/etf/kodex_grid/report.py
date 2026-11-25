"""조립 결과(JSON으로 직렬화한 꼴)를 사람이 읽는 표로 옮긴다. 표시용 반올림만 하고 판정은 하지 않는다."""
from __future__ import annotations

from decimal import Decimal


COSTS = ("base", "stress")
COST_NAMES = {"base": "기본", "stress": "스트레스"}
STATUS = {"pass": "통과", "fail": "미달", "undetermined": "미확정", "positive": "양수", "observed": "관찰",
          "selected": "선택 확정", "none": "영역 없음"}
NOTICE = ("탐색 전용 결과(exploration=true, formal_run=false). 이미 본 과거 기간의 탐색이며 통과해도 확정이 아니다. "
          "가격은 공급자 수정가격 근사이고 분배금 반영은 검증되지 않았다. 표의 수치는 표시용 반올림이며 "
          "판정은 반올림 전 값으로 했다.")


def pp(value, digits=3) -> str:
    """비율의 차를 %p로(표시용)."""
    return "—" if value is None else format(Decimal(value) * 100, f"+.{digits}f")


def pct(value, digits=2) -> str:
    return "—" if value is None else format(Decimal(value) * 100, f".{digits}f")


def mark(value) -> str:
    return "—" if value is None else "O" if value else "X"


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


NOT_RUN_TEXT = {"none": "영역 없음(8개 영역 전부 확정 미달)이므로 2단계를 실행하지 않았다.",
                "undetermined": "선택 미확정(자료 부족으로 순위가 앞선 영역의 판정이 남아 있음)이므로 2단계를 실행하지 않았다."}


def _view(row, path) -> dict:
    """한 경로의 '함께 표기' 항목. 보류된 행은 빈 사전."""
    return (row["display"]["paths"] or {}).get(path) or {}


def _side(row, path, cost) -> dict:
    """한 경로·한 비용의 수치. 수치가 없는 자리는 빈 사전."""
    return _view(row, path).get(cost) or {}


def _ident(row) -> list:
    return [row["number"], f"{row['core']}%", row["kg"], row["kp"], row["levels"]]


IDENT = ["번호", "코어", "kg", "kp", "차수"]


def _verdict_table(rows, paths) -> str:
    headers = IDENT + ["판정"] + [f"{COST_NAMES[c]} 차 {p} (%p)" for c in COSTS for p in paths] + ["공통 승리 구간", "사유"]
    body = []
    for row in rows:
        verdict, common = row["verdict"], row["verdict"]["common_segments"]
        deltas = [pp(_side(row, p, c).get("total_delta")) for c in COSTS for p in paths]
        wins = "—" if common is None else f"{len(common)}개" + (f" ({', '.join(common)})" if common else "")
        body.append(_ident(row) + [STATUS[verdict["status"]]] + deltas + [wins, "; ".join(verdict["reasons"])])
    return table(headers, body)


def _segment_table(rows, frame) -> str:
    """등록한 경로와 구간으로 표의 골격을 먼저 만들고, 값이 없는 칸은 빈값으로 둔다."""
    paths, labels = frame["paths"], frame["labels"]
    body = []
    for row in rows:
        for path in paths:
            deltas = _side(row, path, "base").get("segment_deltas") or {}
            worst = _view(row, path).get("worst_segments")
            body.append([row["number"], path] + [pp(deltas.get(label)) for label in labels]
                        + ["—" if worst is None else ", ".join(worst)])
    return table(["번호", "경로"] + [f"{label} (%p)" for label in labels] + ["최악 구간"], body)


def _risk_table(rows, paths) -> str:
    headers = ["번호", "경로", "낙폭 심화 (%p)", "평균 비중 (%)", "종전 낙폭 한도", "종전 비중 한도", "전체 차 양수"]
    body = []
    for row in rows:
        for path in paths:
            view = _view(row, path)
            limits = view.get("legacy_limits") or {}
            body.append([row["number"], path, pp(_side(row, path, "base").get("mdd_deepening")),
                         pct(_side(row, path, "base").get("exposure_mean")), mark(limits.get("mdd_excess_ok")),
                         mark(limits.get("exposure_ok")), mark(view.get("total_delta_positive"))])
    return table(headers, body)


def _count(value) -> str:
    return "—" if value is None else str(value)


def _blank_cell(reasons) -> str:
    return "유효" if reasons is None else "; ".join(reasons) or "사유 없음"


def _blank_section(rows, paths) -> list:
    """빈칸으로 둔 비용과 그 사유. 보류·거부·결측 비용의 수치는 싣지 않고 다른 비용의 유효한 수치는 그대로 둔다."""
    body = [[row["number"], path] + [_blank_cell(_view(row, path)["blank"][cost]) for cost in COSTS]
            for row in rows for path in paths if any(_view(row, path)["blank"][cost] is not None for cost in COSTS)]
    lines = ["### 빈칸으로 둔 비용과 사유", "",
             "판정과 같은 비용별 유효성 검사를 거친 수치만 표와 요약에 싣는다. 봉인한 단위 원본은 그대로다.", ""]
    return lines + ([table(["번호", "경로", "기본 비용", "스트레스 비용"], body)] if body else ["빈칸 없음."])


def _years(view, cost) -> str:
    closed = (view.get("buys_closed_years") or {}).get(cost)
    return "—" if closed is None else ", ".join(day[:4] for day in closed) or "없음"


def _activity_cells(view, cost) -> list:
    trades, reentries = (view.get("split_trades") or {}).get(cost), (view.get("reentries") or {}).get(cost)
    if trades is None or reentries is None:
        return ["—"] * 5
    unfilled = reentries["unfilled"]
    reasons = "—" if unfilled is None else ", ".join(f"{name.split(':')[0]}×{n}" for name, n in unfilled.items())
    return [_count(trades["buys"]), _count(trades["sells"]),
            f"{_count(reentries['attempts'])}/{_count(reentries['filled'])}", reasons or "없음", _years(view, cost)]


def _activity_table(rows, paths) -> str:
    headers = ["번호", "경로", "비용", "매수 체결", "매도 체결", "재진입 시도/체결", "미체결 사유", "매수 차단 연도"]
    body = [[row["number"], path, COST_NAMES[cost]] + _activity_cells(_view(row, path), cost)
            for row in rows for path in paths for cost in COSTS]
    return table(headers, body)


def _order(order) -> str:
    if order["status"] != "filled":
        return f"{order['status']} ({order.get('reason', '')})"
    return f"체결 {order['quantity']}주 / {Decimal(order['notional']):,.0f}원"


def _orders_table(rows, paths) -> str:
    headers = ["번호", "비용", "코어", "분할", "단독 보유", "코어+분할 원금"]
    body = []
    for row in rows:
        orders = _view(row, paths[0]).get("start_orders")
        for cost in COSTS:
            if not orders:
                body.append([row["number"], COST_NAMES[cost], "—", "—", "—", "—"])
                continue
            start = orders[cost]
            total = "—" if start["core_plus_split"] is None else f"{Decimal(start['core_plus_split']):,.0f}"
            body.append([row["number"], COST_NAMES[cost]] + [_order(start["orders"][name])
                                                           for name in ("core", "split", "hold")] + [total])
    return table(headers, body)


def _share_table(rows, paths) -> str:
    headers = ["번호", "경로", "분할 몫 누적수익 (%)", "같은 원금 보유 (%)", "차 (%p)", "맞춤 비중 (%)",
               "맞춤 계좌 연복리 (%)", "전략 − 맞춤 (%p)"]
    body = []
    for row in rows:
        for path in paths:
            view = _view(row, path)
            share, matched = (view.get("split_share") or {}).get("base") or {}, view.get("matched") or {}
            body.append([row["number"], path, pct(share.get("split"), 3), pct(share.get("hold_same_capital"), 3),
                         pp(share.get("delta")), pct(matched.get("fraction")), pct(matched.get("cagr"), 3),
                         pp(matched.get("strategy_minus_matched"))])
    return table(headers, body)


def _regions_table(regions) -> str:
    headers = ["순위", "코어", "kg", "kp", "설정 번호 8개", "상태", "통과/미달/미확정"]
    body = [[row["rank"], f"{row['core']}%", "·".join(row["kg"]), "·".join(row["kp"]),
             ", ".join(str(n) for n in row["numbers"]), STATUS[row["status"]],
             f"{row['counts']['pass']}/{row['counts']['fail']}/{row['counts']['undetermined']}"] for row in regions]
    return table(headers, body)


def _selection_lines(selection) -> list:
    state, region = selection["state"], selection["region"]
    lines = [f"- 영역 선택: **{STATUS[state] if state != 'undetermined' else '선택 미확정'}**"]
    if region is not None:
        lines.append(f"- 선택 영역: 순위 {region['rank']} (코어 {region['core']}%, kg {'·'.join(region['kg'])}, "
                     f"kp {'·'.join(region['kp'])}), 잠긴 8점 {region['numbers']}, "
                     f"재확인 통과 시 후보 설정 {selection['candidate_if_reconfirmed']}번")
    if selection["blocked_by_rank"] is not None:
        lines.append(f"- 순위 {selection['blocked_by_rank']} 영역이 미확정이라 그 아래 순위를 선택하지 않는다")
    sentence = selection["conclusion"]["sentence"]
    lines.append(f"- 결론: {sentence if sentence else '2단계(KODEX 재확인·확장) 뒤에 정한다'}")
    lines.append(f"- 2단계: {'실행' if selection['stage2'] == 'run' else NOT_RUN_TEXT[state]}")
    return lines


def _ranking_table(summary, paths) -> str:
    headers = ["순서", "번호"] + [f"기본 차 {p} (%p)" for p in paths] + ["둘 중 작은 차 (%p)", "비고"]
    body = [[index, row["number"]] + [pp(row["deltas"].get(p)) for p in paths]
            + [pp(row["min_delta"]), "; ".join(row.get("reasons", []))]
            for index, row in enumerate(summary["ranking"], start=1)]
    return table(headers, body)


def _axis_table(summary, paths) -> str:
    headers = ["축", "값"] + [f"평균 차 {p} (%p, 유효/예정)" for p in paths]
    body = []
    for axis, values in summary["axis_means"].items():
        for value, cells in values.items():
            body.append([axis, value] + [f"{pp(cells[p]['mean_delta'])} ({cells[p]['valid']}/{cells[p]['planned']})"
                                         for p in paths])
    return table(headers, body)


def _sells_table(summary) -> str:
    body = [[row["number"], row["path"], _count(row["sells"]) + (" (0건)" if row["zero_sells"] else ""),
             pp(row["delta"])] for row in summary["sell_pairs"]]
    return table(["번호", "경로", "매도 체결 건수(기본)", "기본 차 (%p)"], body)


def _identity_lines(identity) -> list:
    names = ("approved_source_sha256", "exploration_source_sha256", "grid_file_sha256", "grid_applied_sha256",
             "base_config_file_sha256")
    lines = [f"- {name}: `{identity[name]}`" for name in names]
    actual = identity["inputs"].get("actual", {})
    return lines + [f"- 입력 {key}: `{value}`" for key, value in actual.items()]


def _summary_section(result, paths) -> list:
    summary, counts = result["summary"], result["summary"]["positive_counts"]
    return ["## 설명용 요약 (판정 아님 — 이 표로 영역이나 후보를 바꾸지 않는다)", "",
            f"두 경로 모두 기본 비용 전체 차 > 0: **{counts['both_paths_positive']} / {counts['total']}개**, "
            f"양수 아님(한 경로라도 0 이하): {counts['not_both_positive']}개, "
            f"양수 여부 미확정: {counts['undetermined']}개", "",
            "### 기본 비용 차 순(둘 중 작은 차가 큰 설정부터, 동률은 번호순)", "", _ranking_table(summary, paths), "",
            "### 축 값별 평균 차(경로별 유효한 차만 산술평균)", "", _axis_table(summary, paths), "",
            "### 매도 체결 건수와 기본 비용 차(같은 경로끼리)", "",
            "매도가 0건이어도 현금을 많이 보유했다면 ETF 100% 보유와 노출이 다르다.", "", _sells_table(summary)]


def stage1_markdown(result) -> str:
    rows, paths = result["settings"], result["frame"]["paths"]
    isolated = ", ".join(str(n) for n in result["isolated_passes"]) or "없음"
    parts = ["# KODEX 200 분할매매 변수 지형 조사 — 주 시작일 36개 (탐색)", "", NOTICE, "",
             "## 영역 선택", "", *_selection_lines(result["selection"]),
             f"- 통과 영역에 들지 못한 통과점(공개만 하고 후보로 삼지 않음): {isolated}", "",
             "## 영역 8개", "", _regions_table(result["regions"]), "",
             "## 설정 36개 판정", "", _verdict_table(rows, paths), "",
             "## 함께 표기 (판정 아님)", "", "### 구간별 기본 비용 차", "", _segment_table(rows, result["frame"]), "",
             "### 낙폭·비중·종전 한도", "", _risk_table(rows, paths), "",
             "### 분할 장부 체결과 재진입", "", _activity_table(rows, paths), "",
             "### 시작 주문", "", _orders_table(rows, paths), "",
             "### 분할 몫 자체 비교와 비중 맞춤 계좌(기본 비용)", "", _share_table(rows, paths), "",
             *_blank_section(rows, paths), "",
             *_summary_section(result, paths), "", "## 실행 식별값", "", *_identity_lines(result["identity"]), ""]
    return "\n".join(parts)


def _extension_table(rows, paths) -> str:
    """확장 행: 확장 점의 판정과 두 기본 경로의 전체 차만 싣는다(KODEX 4절 판정은 붙이지 않는다)."""
    headers = ["번호", "점 상태"] + [f"{p} 상태" for p in paths] + [f"기본 차 {p} (%p)" for p in paths] + ["사유"]
    body = []
    for row in rows:
        point, deltas = row["extension_point"], row["display"]["base_deltas"]
        states = [STATUS[point["paths"][p]["status"]] if point["paths"] else "—" for p in paths]
        body.append([row["number"], STATUS[point["status"]]] + states
                    + [pp(deltas.get(p)) for p in paths] + ["; ".join(point["reasons"])])
    return table(headers, body)


def _product_line(symbol, product) -> str:
    return (f"- {symbol}: **{STATUS[product['status']]}** (확정 양수 P={product['positive']}, 미확정 U="
            f"{product['undetermined']}, 확정 미달 {product['failed']}, 분모 {product['denominator']}, "
            f"기준 {product['points_min']})")


def _conclusion_lines(conclusion) -> list:
    lines = [f"**결론: {conclusion['sentence']}** (단계: {conclusion['stage']})"]
    if conclusion.get("candidate") is not None:
        lines.append(f"- 후속 확인용 후보: 설정 {conclusion['candidate']}번")
    if conclusion.get("next_step"):
        lines.append(f"- 다음 단계: {conclusion['next_step']}")
    return lines + [f"- {text}" for text in conclusion.get("limits", [])]


def _stage2_run(result) -> list:
    reconfirm, extension = result["kodex_reconfirm"], result["extension"]
    paths = result["frame"]["paths"]
    counts = reconfirm["verdict"]["counts"]
    pairs = [{**row, "number": f"{row['number']} @{row['start']}"} for row in reconfirm["rows"]]
    parts = ["## KODEX 재확인 (16쌍)", "",
             f"**{STATUS[reconfirm['verdict']['status']]}** — 통과 {counts['pass']}, 미달 {counts['fail']}, "
             f"미확정 {counts['undetermined']} / {reconfirm['verdict']['pairs']}쌍", "",
             _verdict_table(pairs, paths), "", *_blank_section(pairs, paths), "",
             "## 확장 상품 (KODEX 재확인과 별도 칸)", "",
             f"여러 상품에서 관찰: **{STATUS[extension['verdict']['status']]}**", ""]
    for symbol in extension["symbols"]:  # 등록한 상품 순서대로
        parts += [_product_line(symbol, extension["verdict"]["products"][symbol]), "",
                  _extension_table(extension["rows"][symbol], paths), ""]
    return parts


def stage2_markdown(result) -> str:
    head = ["# KODEX 200 분할매매 변수 지형 조사 — 2단계 (탐색)", "", NOTICE, "",
            "## 주 시작일 선택", "", *_selection_lines(result["selection"]), ""]
    body = _stage2_run(result) if result["stage2"] == "run" else [NOT_RUN_TEXT[result["reason"]], ""]
    tail = ["## 결론", "", *_conclusion_lines(result["conclusion"]), "",
            "## 실행 식별값", "", *_identity_lines(result["identity"]), ""]
    return "\n".join(head + body + tail)
