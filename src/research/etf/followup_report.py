"""Rank R2 settings using paired returns with the planned families in the denominator."""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from decimal import Decimal as D, localcontext
from itertools import product
from statistics import median

from .followup_plan import DEPTHS, MIXES, PATHS, SIZES, Setting


ENTRY_MODES = ("once", "monthly_fixed", "monthly_refresh")


def phase_settings(phase, previous=None):
    if phase == "pilot":
        return dict.fromkeys(MIXES, [Setting(8, 10)])
    if phase == "sizing":
        return dict.fromkeys(MIXES, [Setting(size, levels) for size, levels in product(SIZES, DEPTHS)])
    if phase not in {"spacing", "reentry", "forward"}:
        raise ValueError("unknown R2 phase")
    result = {}
    for mix in MIXES:
        ranked = [Setting(**row["setting"]) for row in previous[mix]["ranked"]]
        if phase == "spacing":
            configs = [replace(s, kg=kg, kp=kp) for s in ranked[:2] for kg, kp in product(range(1, 4), repeat=2)]
        else:
            selected = ranked[0]
            configs = [replace(selected, entry=mode) for mode in ENTRY_MODES]
            if phase == "forward":
                configs += [replace(selected, budget="0.60"), replace(selected, stress=True)]
        result[mix] = sorted(set(configs))
    return result


def jobs_for(plan, phase, settings):
    keys = plan["forward"] if phase == "forward" else plan["development"]
    for family in plan["families"]:
        dates = [key for key in keys if int(key[:4]) == family["year"]]
        if phase == "pilot":
            if not dates or family["year"] not in (2020, 2022):
                continue
            dates = dates[:1]
        for key in dates:
            year, month = key.split("-")
            for size in sorted({setting.size for setting in settings[family["mix"]]}):
                yield {"id": f"{family['id']}-{key}-{size}", "family": family["id"], "mix": family["mix"],
                       "quarter": key, "weights": family["baskets"][str(size)],
                       "window": plan["windows"][key],
                       "history": [plan["windows"][f"{int(year) - lag}-{month}"] for lag in (2, 1)],
                       "settings": [asdict(s) for s in settings[family["mix"]] if s.size == size]}


def _median(values):
    return median(values) if values else None


def _delta(row):
    matched = row.get("matched", {})
    if matched.get("status") != "passed":
        return None
    return D(matched["delta_cagr"]), D(matched["delta_mdd"])


def _score(rows):
    families = defaultdict(lambda: defaultdict(list))
    outcomes = Counter()
    for unit, trial in rows:
        val = _delta(trial)
        families[unit["family"]][trial["path"]].append(val)
        outcomes[trial.get("matched", trial)["status"]] += 1
    family_values = {}
    for family, paths in families.items():
        family_values[family] = {}
        for path in PATHS:
            chosen = paths[path]
            if chosen and all(value is not None for value in chosen):
                family_values[family][path] = [_median(list(values)) for values in zip(*chosen)]
    joint = sum(all(min(row.get(path, [D(0)])) > 0 for path in PATHS) for row in family_values.values())
    paths = {}
    for path in PATHS:
        eligible = [v[path] for v in family_values.values() if path in v]
        paths[path] = {"paired": len(eligible), "improved": sum(min(pair) > 0 for pair in eligible),
                       "median_cagr": _median([pair[0] for pair in eligible]),
                       "median_mdd": _median([pair[1] for pair in eligible])}
    return {"families": len(family_values), "improved_both": joint, "paths": paths, "trial_status": dict(outcomes)}


def _rank_key(item):
    setting, score = item
    worst = min((v["median_mdd"] if v["median_mdd"] is not None else D("-Infinity")
                 for v in score["paths"].values()))
    return (-score["improved_both"], -worst, setting.levels, setting.size, setting.kg, setting.kp,
            ENTRY_MODES.index(setting.entry), setting.budget, setting.stress)


def _tertiles(values):
    values = sorted(values)
    if not values:
        return []
    return [values[(i * len(values) - 1) // 3] for i in (1, 2)]


def _band(value, cuts):
    if value is None or not cuts:
        return "unavailable"
    return ("low", "mid", "high")[sum(D(value) > cut for cut in cuts)]


def _runs_summary(entries):
    return {path: _run_path([trial for _, trial in entries if trial["path"] == path]) for path in PATHS}


def _run_path(trials):
    rows = []
    for trial in trials:
        if trial["status"] == "passed":
            rows.append(trial["strategy"])
    names = ("cagr", "mdd", "average_investment_fraction", "max_filled_level",
             "mean_usd_cash", "mean_usd_holdings", "fx_fees_krw")
    return {"strategy_passed": len(rows), "median": {name: _median([D(row[name]) for row in rows]) for name in names},
            "deep_fills": sum(r["deep_fills"] for r in rows),
            "insufficient_cash": sum(r["insufficient_cash"] for r in rows),
            "reentry": {name: sum(r["reentry"][name] if name != "unavailable" else
                                 sum(r["reentry"][name].values()) for r in rows)
                        for name in ("attempted", "filled", "rejected", "unavailable")}}


def _direct(rows):
    by_path = {}
    for path in PATHS:
        diffs = []
        for _, row in rows:
            hold = row.get("buy_hold", {})
            if row["path"] != path or "cagr" not in hold:
                continue
            dims = ("cagr", "mdd")
            diffs.append(tuple(D(row["strategy"][name]) - D(hold[name]) for name in dims))
        by_path[path] = {"paired": len(diffs), "joint_improvement": sum(min(d) > 0 for d in diffs),
                         "median_cagr": _median([d[0] for d in diffs]), "median_mdd": _median([d[1] for d in diffs])}
    return by_path


def _describe(rows, cuts):
    quarters, volatility, years = defaultdict(list), defaultdict(list), defaultdict(list)
    for unit, trial in rows:
        quarters[unit["quarter"][-2:]].append((unit, trial))
        years[unit["quarter"][:4]].append((unit, trial))
        volatility[_band(unit["starting_volatility"], cuts)].append((unit, trial))
    return dict(runs=_runs_summary(rows), buy_hold=_direct(rows),
                quarters={k: _runs_summary(v) for k, v in quarters.items()},
                years={k: _score(v) for k, v in years.items()},
                starting_volatility={k: _runs_summary(v) for k, v in volatility.items()})


def _ab(rows, mode_a, mode_b):
    pairs = defaultdict(dict)
    for unit, trial in rows:
        setting = trial["setting"]
        if setting["stress"] or setting["budget"] != "0.80":
            continue
        pairs[unit["id"], trial["path"]][setting["entry"]] = trial
    answer = {}
    for path in PATHS:
        values = []
        planned = sum(p == path for _, p in pairs)
        for (_, p), modes in pairs.items():
            if p != path or any(mode not in modes or modes[mode]["status"] != "passed" for mode in (mode_a, mode_b)):
                continue
            a, b = (modes[mode]["strategy"] for mode in (mode_a, mode_b))
            values.append(tuple(D(b[name]) - D(a[name]) for name in ("cagr", "mdd", "average_investment_fraction")))
        answer[path] = {"planned": planned, "paired": len(values), "joint_improvement": sum(min(v[:2]) > 0 for v in values),
                        "median_delta": [_median([v[i] for v in values]) for i in range(3)]}
    return answer


def _candidate(scores, previous):
    selected = Setting(**previous["ranked"][0]["setting"])
    base, stressed = scores[selected], scores[replace(selected, stress=True)]
    failed = []
    for path in PATHS:
        values, stress = base["paths"][path], stressed["paths"][path]
        checks = {
            "joint_60_pct": values["improved"] * 5 >= 3 * base["families"],
            "cagr_0.5pp": values["median_cagr"] is not None and values["median_cagr"] >= D(".005"),
            "mdd_2pp": values["median_mdd"] is not None and values["median_mdd"] >= D(".02"),
            "cost_stress_positive": all(stress[key] is not None and stress[key] > 0
                                        for key in ("median_cagr", "median_mdd")),
        }
        failed.extend(f"{path}: {name}" for name, ok in checks.items() if not ok)
    return {"setting": asdict(selected), "proposed": not failed, "failed_conditions": failed}


def summarize(phase, units, previous=None):
    with localcontext(prec=80):
        grouped = {mix: defaultdict(list) for mix in MIXES}
        for unit in units:
            for trial in unit["trials"]:
                grouped[unit["mix"]][Setting(**trial["setting"])].append((unit, trial))
        result = {}
        for mix, sets in grouped.items():
            scores = {setting: _score(rows) for setting, rows in sets.items()}
            ranked = sorted(scores.items(), key=_rank_key)
            if not ranked:
                raise ValueError(f"no settings in {phase}/{mix}")
            lead = ranked[0][0]
            distinct = {u["id"]: D(u["starting_volatility"]) for u, _ in sets[lead]
                        if u["starting_volatility"] is not None}
            cuts = ([D(v) for v in previous[mix]["volatility_cuts"]] if phase == "forward"
                    else _tertiles(list(distinct.values())))
            result[mix] = {"volatility_cuts": cuts, "ranked": [
                {"setting": asdict(setting), **score, **_describe(sets[setting], cuts)} for setting, score in ranked]}
            if phase in {"reentry", "forward"}:
                rows = [entry for values in sets.values() for entry in values]
                result[mix]["reentry_vs_once"] = _ab(rows, "once", "monthly_fixed")
                result[mix]["refresh_vs_fixed"] = _ab(rows, "monthly_fixed", "monthly_refresh")
            if phase == "forward":
                result[mix]["candidate"] = _candidate(scores, previous[mix])
        return result
