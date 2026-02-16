"""
Analysis/reporting helpers for GPU parameter simulation results.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from .context import PARAM_ORDER, _ensure_core_deps


# -----------------------------------------------------------------------------
# Analysis and Result Saving
# -----------------------------------------------------------------------------
def analyze_and_save_results(param_combinations_gpu, daily_values_gpu, trading_dates_pd, save_to_file=True):
    np, pd = _ensure_core_deps()
    try:
        from ...performance_analyzer import PerformanceAnalyzer
    except ImportError:  # pragma: no cover
        # Legacy mode: imported as top-level `optimization.*` with `src/` on sys.path.
        from performance_analyzer import PerformanceAnalyzer

    print("\n--- 🔬 Analyzing detailed performance metrics ---")
    start_time = time.time()

    param_combinations_cpu = param_combinations_gpu.get()
    daily_values_cpu = daily_values_gpu.get()

    results_list = []
    for idx in range(daily_values_cpu.shape[0]):
        history_df_mock = pd.DataFrame(
            pd.Series(daily_values_cpu[idx], index=trading_dates_pd),
            columns=["total_value"],
        )
        analyzer = PerformanceAnalyzer(history_df_mock)
        results_list.append(analyzer.get_metrics(formatted=False))

    param_names = list(PARAM_ORDER)
    params_df = pd.DataFrame(param_combinations_cpu, columns=param_names)
    metrics_df = pd.DataFrame(results_list)
    full_results_df = pd.concat([params_df, metrics_df], axis=1)

    full_results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sorted_df = full_results_df.sort_values(by="calmar_ratio", ascending=False).dropna(subset=["calmar_ratio"])

    if not sorted_df.empty:
        best_params_dict = sorted_df.iloc[0].to_dict()
    else:
        best_params_dict = {}

    print("\n🏆 Top 10 Performing Parameter Combinations (by Calmar Ratio):")
    display_columns = [
        "calmar_ratio",
        "cagr",
        "mdd",
        "sharpe_ratio",
        "stop_loss_rate",
        "max_splits_limit",
        "max_inactivity_period",
        "sell_profit_rate",
        "additional_buy_drop_rate",
    ]
    display_df = sorted_df.head(10).get(display_columns, pd.DataFrame())
    if not display_df.empty:
        for col in ["cagr", "mdd", "annualized_volatility", "stop_loss_rate"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map("{:.2%}".format)
        for col in ["calmar_ratio", "sharpe_ratio", "sortino_ratio"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map("{:.2f}".format)
        print(display_df.to_string(index=False))

    if save_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"gpu_simulation_results_{timestamp}.csv")
        sorted_df.to_csv(filepath, index=False, float_format="%.4f")
        print(f"\n✅ Full analysis saved to: {filepath}")

    print(f"⏱️  Analysis took: {time.time() - start_time:.2f} seconds.")
    return best_params_dict, sorted_df


__all__ = [
    "analyze_and_save_results",
]
