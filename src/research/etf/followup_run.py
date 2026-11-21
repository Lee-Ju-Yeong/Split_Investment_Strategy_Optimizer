"""R2 execution units and shared, manifest-verified price input."""
from collections import Counter
from dataclasses import asdict, replace
from datetime import date, timedelta
from decimal import Decimal as D, localcontext
from functools import lru_cache
import resource
import time

from . import protocol
from .baseline_engine import run_baseline
from .baseline_policy import BaselinePolicy
from .baseline_report import paired_difference, performance
from .engine import Reentry
from .engine_inputs import _calendar, _prices, _snapshot
from .fast_kernel import FastKernel, run_fast_engine
from .followup_plan import PATHS, Setting
from .m5_inputs import _load_m5_inputs, read_manifest
from .m5_parameters import CostGuardRejection, candidate_rules, entry_rates, estimate_volatility
from .m5_pilot import _window_start
from .m5_runner import _compare, _exposure, matched_policy


class StudyData:
    def __init__(self, root, manifest_path, plan, *, forward=False):
        keys = plan["forward"] if forward else plan["development"]
        years = {int(key[:4]) for key in keys}
        symbols = sorted({symbol for family in plan["families"] if family["year"] in years
                          for symbol in family["baskets"]["8"]})
        self.manifest = read_manifest(root, manifest_path, symbols)
        record, selected, _, bundle = _snapshot(root, root / self.manifest["split_csv"], symbols)
        self.root, self.manifest_path = root, manifest_path
        end = date.fromisoformat(max(plan["windows"][key][1] for key in keys))
        self.histories = _prices(root, bundle, record, selected, end)
        self.calendars = {market: _calendar(root / "fx" / f"calendar_{name}.csv")
                          for market, name in (("KR", "krx"), ("US", "nyse"))}
        self.markets = {symbol: row["market"] for symbol, row in selected.items()}
        self.protocol = protocol.load()

    @lru_cache(maxsize=32768)
    def volatility(self, symbol, start):
        return estimate_volatility(self.histories[symbol], self.calendars[self.markets[symbol]], start)

    @lru_cache(maxsize=9)
    def inputs(self, symbols, span):
        start, end = map(date.fromisoformat, span)
        fx = self.protocol["fx"]
        return _load_m5_inputs(self.root, self.manifest_path, list(symbols), start, end, manifest=self.manifest,
                               max_fx_age_days=fx["max_publication_age_days"],
                               fx_closure_age_days=fx["max_closure_publication_age_days"],
                               verified_price_rows=self.histories)


def _month(day):
    return day.year * 12 + day.month


class MonthlyEntry:
    def __init__(self, study, inputs, setting, commission_rate, initial_estimates=None):
        self.study, self.inputs, self.setting = study, inputs, setting
        self.commission_rate = commission_rate
        self.initial_estimates = initial_estimates
        self.first_sessions = {}
        for market, calendar in study.calendars.items():
            for day in calendar:
                self.first_sessions.setdefault((market, _month(day)), day)

    def __call__(self, instrument, bar, last_exit):
        symbol, day = bar.symbol, bar.session_date
        if self.first_sessions.get((instrument.market, _month(last_exit) + 1)) != day:
            return None
        try:
            estimate = (self.initial_estimates[symbol] if self.setting.entry == "monthly_fixed"
                        else self.study.volatility(symbol, day))
        except ValueError as error:
            return Reentry(unavailable=str(error))
        try:
            rates = entry_rates(asdict(self.setting), self.inputs, instrument, bar, estimate,
                                commission_rate=self.commission_rate)
        except CostGuardRejection as error:
            return Reentry(unavailable=str(error))
        return Reentry(rates=rates if self.setting.entry == "monthly_refresh" else None)


def _options(inputs, context, setting, path):
    fx = context["protocol"]["fx"]["conversion"]
    fx_key = "cost_stress_one_way" if setting.stress else "cost_one_way"
    commission = context["protocol"]["execution"]["costs_one_way"]["stress_one_way_all"] if setting.stress else None
    options = dict(inputs.engine_kwargs(), initial_cash={"KRW": context["capital"]}, path=path,
                   usd_target_fraction=context["usd"], fx_cost=D(str(fx[fx_key])),
                   cash_floor_fraction=D(0), symbol_cap_fraction=D("0.5"),
                   commission_rate=None if commission is None else D(str(commission)))
    options["_evaluation_cutoff"] = cutoff = max(inputs.evaluation_times)
    options["fx_rates"] = [rate for rate in inputs.fx_rates if rate.available_at <= cutoff]
    if context["usd"] == 0:
        options["conversion_times"] = ()
    return options


def _strategy(study, kernel, loaded, context, setting, path):
    inputs, estimates = loaded
    options = _options(inputs, context, setting, path)
    commission = options["commission_rate"]
    rules = candidate_rules(asdict(setting), inputs, estimates, commission_rate=commission)
    reentry = None if setting.entry == "once" else MonthlyEntry(study, inputs, setting, commission, estimates)
    with localcontext(prec=80):
        fraction = D(setting.budget) / setting.levels
        return run_fast_engine(kernel, inputs.instruments, inputs.bars, rules, **options,
                               first_level_fraction=fraction, level_budget_weights=dict(context["weights"]),
                               _reentry_provider=reentry)


def _loaded(study, symbols, span):
    inputs = study.inputs(symbols, tuple(span))
    firsts = {}
    for bar in inputs.bars:
        firsts.setdefault(bar.symbol, bar.session_date)
    return inputs, {symbol: study.volatility(symbol, firsts[symbol]) for symbol in symbols}


def _window(inputs, usd):
    start = _window_start(inputs) if usd else min(bar.open_at for bar in inputs.bars)
    return start, max(inputs.evaluation_times)


def _context(study, job):
    with localcontext(prec=80):
        weights = tuple((symbol, D(weight)) for symbol, weight in job["weights"].items())
        return {"weights": weights, "usd": sum((w for s, w in weights if study.markets[s] == "US"), D(0)),
                "capital": D(study.protocol["portfolio"]["initial_capital_krw"]), "protocol": study.protocol}


def _stats(result, inputs, context, window):
    summary = asdict(performance(result, inputs.instruments, initial_equity_krw=context["capital"],
                                 started_at=window[0], ended_at=window[1]))
    fills = Counter(row["level"] for row in result.trades if row["side"] == "BUY")
    reentries = [row for row in result.cash_events if row["kind"] == "monthly_reentry"]
    orders = {(row["symbol"], row["at"]) for row in reentries if row["unavailable"] is None}
    rejects = {(row["symbol"], row["generated_at"]) for row in result.rejections}
    filled = {(row["symbol"], row["generated_at"]) for row in result.trades if row["reason"] == "monthly_reentry"}
    if filled | (orders & rejects) != orders or filled & rejects:
        raise RuntimeError("monthly reentry attempts do not reconcile to the execution ledger")
    usd = {item.symbol for item in inputs.instruments if item.currency == "USD"}
    with localcontext(prec=80):
        vals = result.valuations
        usd_holdings = [(sum(row["holdings_krw"][s] for s in usd) / row["total_krw"]) for row in vals]
        usd_cash = [(row["cash"]["USD"] * row["fx_rate"] / row["total_krw"]) for row in vals]
        return dict(summary, buy_fills=dict(fills), max_filled_level=max(fills, default=0),
                    deep_fills=sum(n for level, n in fills.items() if 6 <= level <= 10),
                    insufficient_cash=sum(row["reason"] == "insufficient_cash" for row in result.rejections),
                    reentry={"attempted": len(reentries), "filled": len(filled), "rejected": len(orders & rejects),
                             "dates": [[row["symbol"], row["at"]] for row in reentries],
                             "unavailable": dict(Counter(row["unavailable"] for row in reentries if row["unavailable"]))},
                    mean_usd_holdings=sum(usd_holdings) / len(vals), mean_usd_cash=sum(usd_cash) / len(vals),
                    fx_fees_krw=sum((row["fee_krw"] for row in result.cash_events if row["kind"] == "fx_conversion"), D(0)),
                    warnings=list(result.warnings))


def _buy_hold(inputs, context, window, setting, path):
    policy = BaselinePolicy(D(1), context["weights"], window[0] - timedelta(seconds=1),
                            window[0], "0" * 64, role="reference", mode="buy_hold")
    result = run_baseline(policy, inputs.instruments, inputs.bars, **_options(inputs, context, setting, path),
                          _cap_buy_to_cash=True)
    return _stats(result, inputs, context, window)


def _matched_baseline(study, kernel, history, context, setting, path, window, inputs, result):
    records = []
    for loaded in history:
        data = loaded[0]
        learned = _strategy(study, kernel, loaded, context, setting, path)
        records.append(_exposure(learned, data.instruments, _window(data, context["usd"])))
    policy = replace(matched_policy(records, inputs.instruments, window[0]), weights=context["weights"])
    return _compare(policy, result, performance(result, inputs.instruments, initial_equity_krw=context["capital"],
                                               started_at=window[0], ended_at=window[1]), inputs,
                    dict(_options(inputs, context, setting, path), _cap_buy_to_cash=True), context["capital"], window)


def _error(error):
    return {"status": "rejected" if isinstance(error, CostGuardRejection) else "failed",
            "error_type": type(error).__name__, "error": str(error)}


def _trial(study, kernel, data, context, setting, path, holds):
    evaluation, history = data
    row = {"setting": asdict(setting), "path": path}
    try:
        if isinstance(evaluation, Exception):
            raise evaluation
        inputs = evaluation[0]
        window = _window(inputs, context["usd"])
        result = _strategy(study, kernel, evaluation, context, setting, path)
        metrics = _stats(result, inputs, context, window)
        row.update(status="passed", strategy=metrics)
    except (ValueError, ArithmeticError) as error:
        row.update(_error(error))
        return row
    try:
        ref = (path, setting.stress)
        if ref not in holds:
            holds[ref] = _buy_hold(inputs, context, window, setting, path)
        row["buy_hold"] = holds[ref]
    except (ValueError, ArithmeticError) as error:
        row["buy_hold"] = _error(error)
    try:
        if isinstance(history, Exception):
            raise history
        row["matched"] = dict(status="passed", **_matched_baseline(
            study, kernel, history, context, setting, path, window, inputs, result))
    except (ValueError, ArithmeticError) as error:
        row["matched"] = _error(error)
    return row


def _load_pair(study, job):
    symbols = tuple(job["weights"])
    try:
        evaluation = _loaded(study, symbols, job["window"])
    except ValueError as error:
        return error, error
    try:
        return evaluation, [_loaded(study, symbols, span) for span in job["history"]]
    except ValueError as error:
        return evaluation, error


_STUDY = _KERNEL = None


def prepare_worker(study, build):
    global _STUDY, _KERNEL
    _STUDY, _KERNEL = study, FastKernel(build)


def execute_unit(job):
    study, kernel = _STUDY, _KERNEL
    started = time.perf_counter()
    data = _load_pair(study, job)
    context = _context(study, job)
    evaluations = data[0]
    holds = {}
    rows = [_trial(study, kernel, data, context, Setting(**config), path, holds)
            for config in job["settings"] for path in PATHS]
    with localcontext(prec=80):
        risk = None if isinstance(evaluations, Exception) else sum(
            weight * evaluations[1][symbol].value for symbol, weight in context["weights"])
    return dict(job, starting_volatility=risk, trials=rows, elapsed_seconds=time.perf_counter() - started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
