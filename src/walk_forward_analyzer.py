# src/walk_forward_analyzer.py

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from .config_loader import load_config

if TYPE_CHECKING:
    import pandas as pd

def _normalize_additional_buy_priority(value: object) -> str:
    """
    Canonicalize `additional_buy_priority` to the string values expected by workers.

    Notes:
    - GPU optimization results may carry this as numeric 0/1.
    - CPU strategy treats any non-"lowest_order" as drop-priority, but GPU workers map string->0/1.
    """
    if value is None:
        return "lowest_order"

    if isinstance(value, (int, float)):
        try:
            return "highest_drop" if int(value) == 1 else "lowest_order"
        except (TypeError, ValueError):
            return "lowest_order"

    if isinstance(value, str):
        v = value.strip()
        if v in {"lowest_order"}:
            return "lowest_order"
        if v in {"highest_drop", "biggest_drop"}:
            return "highest_drop"
        return v

    return "lowest_order"


def _select_legacy_best_params(simulation_results_df: "pd.DataFrame") -> dict:
    """
    Legacy selector: choose the single best row by `calmar_ratio`.
    Returns an empty dict when selection is impossible.
    """
    if simulation_results_df is None or simulation_results_df.empty:
        return {}

    if "calmar_ratio" in simulation_results_df.columns:
        sorted_df = (
            simulation_results_df.sort_values("calmar_ratio", ascending=False)
            .dropna(subset=["calmar_ratio"])
        )
        row = sorted_df.iloc[0] if not sorted_df.empty else simulation_results_df.iloc[0]
    else:
        row = simulation_results_df.iloc[0]

    params_dict = row.to_dict()
    params_dict["additional_buy_priority"] = _normalize_additional_buy_priority(
        params_dict.get("additional_buy_priority")
    )
    return params_dict


def compute_robust_score(
    cluster_summary: "pd.DataFrame",
    *,
    metric: str = "calmar_ratio",
    k: float = 1.0,
    size_col: str = "size",
) -> "pd.Series":
    """
    Issue #68:
    - robust_score = (mean - k*std) * log1p(cluster_size)
    """
    import numpy as np
    import pandas as pd

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    missing = [c for c in (mean_col, std_col, size_col) if c not in cluster_summary.columns]
    if missing:
        raise KeyError(f"Missing required columns for robust score: {missing}")

    mean = pd.to_numeric(cluster_summary[mean_col], errors="coerce")
    std = pd.to_numeric(cluster_summary[std_col], errors="coerce").fillna(0.0)
    size = pd.to_numeric(cluster_summary[size_col], errors="coerce").fillna(0.0)
    return (mean - (k * std)) * np.log1p(size)


def apply_robust_gates(
    fold_metrics_df: "pd.DataFrame",
    *,
    metric: str = "calmar_ratio",
    min_oos_is_ratio: float = 0.60,
    min_fold_pass_rate: float = 0.70,
    max_oos_mdd_p95: float = 0.25,
) -> tuple["pd.DataFrame", dict]:
    """
    Issue #68 gates:
    - median(OOS/IS) >= 0.60
    - fold_pass_rate >= 70%
    - OOS_MDD_p95 <= 25%

    Expected columns:
    - `is_{metric}`, `oos_{metric}`, `oos_mdd`
    """
    import numpy as np
    import pandas as pd

    def _quantile_higher(series: "pd.Series", q: float) -> float:
        """
        Conservative empirical quantile.

        For discrete fold samples we prefer a "higher" quantile (no linear interpolation)
        so that a small tail of worse MDD values is not smoothed away.
        """
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        if values.size == 0:
            return float("nan")
        values = np.sort(values)
        idx = int(np.ceil(q * (len(values) - 1)))
        idx = max(0, min(idx, len(values) - 1))
        return float(values[idx])

    if fold_metrics_df is None or fold_metrics_df.empty:
        return pd.DataFrame(), {
            "gate_passed": False,
            "reason": "empty_fold_metrics",
        }

    is_col = f"is_{metric}"
    oos_col = f"oos_{metric}"
    required_cols = {is_col, oos_col, "oos_mdd"}
    missing = sorted(required_cols - set(fold_metrics_df.columns))
    if missing:
        raise KeyError(f"Missing required columns for robust gates: {missing}")

    df = fold_metrics_df.copy()
    is_vals = pd.to_numeric(df[is_col], errors="coerce")
    oos_vals = pd.to_numeric(df[oos_col], errors="coerce")

    ratio = np.where(is_vals > 0, oos_vals / is_vals, 0.0)
    df["oos_is_ratio"] = pd.Series(ratio, index=df.index)
    df["fold_pass"] = df["oos_is_ratio"] >= min_oos_is_ratio

    median_ratio = float(df["oos_is_ratio"].median())
    fold_pass_rate = float(df["fold_pass"].mean())

    oos_mdd_abs = pd.to_numeric(df["oos_mdd"], errors="coerce").abs()
    oos_mdd_p95 = _quantile_higher(oos_mdd_abs, 0.95)

    gate_passed = bool(
        (median_ratio >= min_oos_is_ratio)
        and (fold_pass_rate >= min_fold_pass_rate)
        and (oos_mdd_p95 <= max_oos_mdd_p95)
    )

    summary = {
        "gate_passed": gate_passed,
        "metric": metric,
        "median_oos_is_ratio": median_ratio,
        "fold_pass_rate": fold_pass_rate,
        "oos_mdd_p95": oos_mdd_p95,
        "threshold_min_oos_is_ratio": min_oos_is_ratio,
        "threshold_min_fold_pass_rate": min_fold_pass_rate,
        "threshold_max_oos_mdd_p95": max_oos_mdd_p95,
        "num_folds": int(len(df)),
    }
    return df, summary

# --- Clustering Helper Function ---
def find_robust_parameters(
    simulation_results_df: "pd.DataFrame",
    param_cols: list,
    metric_cols: list,
    k_range: tuple = (2, 11),
    min_cluster_size_ratio: float = 0.05,
    score_metric: str = "calmar_ratio",
    score_k: float = 1.0,
) -> tuple[dict, "pd.DataFrame | None"]:
    """
    K-Means 클러스터링을 사용하여 시뮬레이션 결과에서 가장 강건한 파라미터 조합을 찾습니다.
    (WFO 파이프라인에 통합하기 위해 시각화 코드는 제거된 버전)
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    print("\n--- 4a. Robust Parameter Search via Clustering ---")
    features = param_cols + metric_cols
    df = simulation_results_df[features].dropna()
    
    if df.empty or len(df) < k_range[0]:
        print("[Warning] Not enough data for clustering. Returning best result by Calmar.")
        best_by_calmar = simulation_results_df.sort_values('calmar_ratio', ascending=False).iloc[0]
        return best_by_calmar.to_dict(), None

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    
    best_k, best_score = -1, -1
    k_candidates = range(k_range[0], min(k_range[1], len(df)))
    for k in k_candidates:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(scaled_features)
        if len(np.unique(cluster_labels)) < 2: continue
        score = silhouette_score(scaled_features, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    if best_k == -1: best_k = k_range[0] # Fallback
    print(f"  - Optimal k detected: {best_k} (Silhouette Score: {best_score:.4f})")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_summary = df.groupby('cluster')[metric_cols].mean()
    cluster_summary['size'] = df['cluster'].value_counts()

    if score_metric not in metric_cols:
        raise ValueError(f"score_metric '{score_metric}' must be included in metric_cols")

    cluster_summary[f"{score_metric}_mean"] = cluster_summary[score_metric]
    cluster_summary[f"{score_metric}_std"] = (
        df.groupby("cluster")[score_metric].std().reindex(cluster_summary.index).fillna(0.0)
    )
    cluster_summary["robustness_score"] = compute_robust_score(
        cluster_summary,
        metric=score_metric,
        k=score_k,
        size_col="size",
    )
    
    min_cluster_size = int(len(df) * min_cluster_size_ratio)
    qualified_clusters = cluster_summary[cluster_summary['size'] >= min_cluster_size]
    
    robust_cluster_id = qualified_clusters['robustness_score'].idxmax() if not qualified_clusters.empty else cluster_summary['robustness_score'].idxmax()
    print(f"  - Most robust cluster identified: Cluster {robust_cluster_id}")
    
    robust_cluster_df = df[df['cluster'] == robust_cluster_id]
    centroid = kmeans.cluster_centers_[robust_cluster_id]
    
    nn = NearestNeighbors(n_neighbors=1).fit(scaled_features[df.index.isin(robust_cluster_df.index)])
    _, indices = nn.kneighbors([centroid])
    
    closest_point_index = robust_cluster_df.index[indices[0][0]]
    best_params_series = simulation_results_df.loc[closest_point_index]
    
    # WFO 결과 저장을 위해 클러스터링 결과가 포함된 DF 반환
    clustered_df_full = df.reset_index().merge(simulation_results_df.drop(columns=features, errors='ignore'), left_on='index', right_index=True)

    best_params_dict = best_params_series.to_dict()
    best_params_dict["additional_buy_priority"] = _normalize_additional_buy_priority(
        best_params_dict.get("additional_buy_priority")
    )

    return best_params_dict, clustered_df_full
# --- 분석 및 시각화 헬퍼 함수 ---
def plot_wfo_results(final_curve: "pd.Series", params_df: "pd.DataFrame", results_dir: str):
    """최종 WFO 결과(수익곡선, 파라미터 분포)를 시각화하고 저장합니다."""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .performance_analyzer import PerformanceAnalyzer

    print("\n" + "="*80)
    print("🎨 Generating WFO result plots...")
    print("="*80)
    
    # 1. 최종 WFO Equity Curve 및 MDD 플롯
    history_df = pd.DataFrame(final_curve, columns=['total_value'])
    analyzer = PerformanceAnalyzer(history_df)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity Curve
    ax1.set_title('Walk-Forward Optimization Equity Curve', fontsize=16)
    ax1.plot(analyzer.daily_values.index, analyzer.daily_values, color='b', label='Equity Curve')
    ax1.set_ylabel('Portfolio Value'); ax1.legend(loc='upper left'); ax1.grid(True)

    # Drawdown
    drawdown = (analyzer.daily_values - analyzer.daily_values.cummax()) / analyzer.daily_values.cummax()
    ax2.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown'); ax2.set_xlabel('Date'); ax2.legend(loc='upper left'); ax2.grid(True)
    
    plt.tight_layout()
    equity_curve_path = os.path.join(results_dir, "wfo_equity_curve.png")
    plt.savefig(equity_curve_path, dpi=300)
    plt.close()
    print(f"✅ WFO Equity Curve plot saved to: {equity_curve_path}")

    # 2. 파라미터 안정성(분포) 플롯
    numeric_params = params_df.select_dtypes(include='number').columns.drop('fold', errors='ignore')
    if not numeric_params.empty:
        cols = 3
        rows = (len(numeric_params) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

        for i, param in enumerate(numeric_params):
            sns.histplot(data=params_df, x=param, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {param}')
        
        for j in range(i + 1, len(axes)): fig.delaxes(axes[j])

        plt.tight_layout()
        param_dist_path = os.path.join(results_dir, "wfo_parameter_distribution.png")
        plt.savefig(param_dist_path, dpi=300); plt.close()
        print(f"✅ Parameter Distribution plot saved to: {param_dist_path}")

# --- Orchestrator 메인 로직 ---

def run_walk_forward_analysis():
    """
    Walk-Forward Optimization 프로세스 전체를 총괄하는 오케스트레이터 함수.
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # 실제 워커 함수 및 분석 모듈은 GPU 환경에서만 필요하므로 lazy import
    from .debug_gpu_single_run import run_single_backtest
    from .parameter_simulation_gpu import find_optimal_parameters
    from .performance_analyzer import PerformanceAnalyzer

    # 1. 설정 로드
    config = load_config()
    wfo_settings = config['walk_forward_settings']
    backtest_settings = config['backtest_settings']
    initial_cash = backtest_settings['initial_cash'] 
    robust_selection_enabled = bool(wfo_settings.get("robust_selection_enabled", True))
    selection_mode = "robust" if robust_selection_enabled else "legacy"

    robust_score_metric = str(wfo_settings.get("robust_score_metric", "calmar_ratio"))
    robust_score_k = float(wfo_settings.get("robust_score_k", 1.0))
    robust_gate_metric = str(wfo_settings.get("robust_gate_metric", "calmar_ratio"))
    robust_gates_cfg = wfo_settings.get("robust_gates", {})
    if not isinstance(robust_gates_cfg, dict):
        robust_gates_cfg = {}
    gate_min_oos_is_ratio = float(robust_gates_cfg.get("min_oos_is_ratio", 0.60))
    gate_min_fold_pass_rate = float(robust_gates_cfg.get("min_fold_pass_rate", 0.70))
    gate_max_oos_mdd_p95 = float(robust_gates_cfg.get("max_oos_mdd_p95", 0.25))

    print(
        "[WFO] "
        f"selection_mode={selection_mode}, "
        f"robust_score_metric={robust_score_metric}, "
        f"robust_gate_metric={robust_gate_metric}"
    )

    # 2. [핵심] 모든 기간 파라미터 자동 계산
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🚀 Starting Robustness-Focused Walk-Forward Optimization")

    # 사용자 설정값 추출
    total_start_date = pd.to_datetime(backtest_settings['start_date'])
    total_end_date = pd.to_datetime(backtest_settings['end_date'])
    total_folds = wfo_settings['total_folds']
    period_length_days = wfo_settings['period_length_days']
        
    # --- 확정 WFO 기간 생성 (no-overlap 우선, 불가 시 최소 겹침 + 균등분포) ---
    S = pd.to_datetime(backtest_settings['start_date']).normalize()
    E = pd.to_datetime(backtest_settings['end_date']).normalize()
    N = int(wfo_settings['total_folds'])
    L_days = int(wfo_settings['period_length_days'])
    L = pd.Timedelta(days=L_days)

    if N <= 0 or L_days <= 0:
        raise ValueError("total_folds and period_length_days must be positive.")

    # 1) 무겹침 가능성 평가
    #   d = OOS_Start - IS_Start, 겹침 = L - d
    #   무겹침 필요조건: d >= L
    #   경계조건: last_IS_start = E - d - (L-1) >= S  ->  d <= (E - S).days - (L-1)
    Dmax_days = (E - S).days - (L_days - 1)   # d가 가질 수 있는 최대값(경계 위배 없이)
    d_days = None

    if Dmax_days >= L_days:
        # 여유로움 → 무겹침 채택
        d_days = L_days
    else:
        # 여유 부족 → 겹침 최소(= d 최대)와 균등성의 균형
        # 기본값: 절반쯤 이동(균형) -> 이전에 합의한 d≈L/2 (+1 보정)
        d_days = min(Dmax_days, (L_days + 1) // 2 + 1)
        if d_days < 1:
            d_days = 1  # 인과성 보장

    # 2) 마지막 폴드가 E에 맞도록 IS 최종 시작점 역산
    d = pd.Timedelta(days=d_days)
    last_is_start = E - d - (L - pd.Timedelta(days=1))

    # 3) IS 시작들의 균등 분포
    #    span_days가 작아도 N개 균등 배치(정수 보정: 몫/나머지 방식)
    span_days = (last_is_start - S).days
    print("span_days:",span_days)
    if span_days <= 0:
        raise ValueError(f"Configuration Error: Cannot fit {N} folds. The total period is too short for the given period length ({L_days} days). Please reduce 'total_folds' or 'period_length_days'.")
    if N == 1:
        is_starts = [S]
    else:
        base_step = span_days // (N - 1)
        remainder = span_days % (N - 1)
        is_starts = [S]
        for i in range(1, N):
            add = base_step + (1 if i <= remainder else 0)
            is_starts.append(is_starts[-1] + pd.Timedelta(days=add))

    # 4) 폴드 구간 구성
    fold_periods = []
    for i, is_start in enumerate(is_starts):
        is_end   = is_start + L - pd.Timedelta(days=1)
        oos_start = is_start + d
        oos_end   = oos_start + L - pd.Timedelta(days=1)

        # 안전 체크
        if oos_start < is_start + pd.Timedelta(days=1):
            raise ValueError("Causality violated: OOS must start at least 1 day after IS start.")
        if oos_end > E:
            raise ValueError("Boundary violated: OOS end beyond end_date.")
        if is_start < S or is_end > E:
            raise ValueError("IS period out of bounds.")

        fold_periods.append({
            'Fold': i + 1,
            'IS_Start': is_start.date(), 'IS_End': is_end.date(),
            'OOS_Start': oos_start.date(), 'OOS_End': oos_end.date()
        })

    print("\n--- Calculated Walk-Forward Folds ---")
    print(pd.DataFrame(fold_periods).to_string(index=False))

    # 참고 출력(선택): 실제 겹침일
    overlap_days = L_days - d_days  # (0이면 무겹침)
    print(f"\n[WFO] d = {d_days} days → overlap = {overlap_days} days (per fold)")


    
    #  새로운 롤링 윈도우 루프
    all_oos_curves, all_optimal_params = [], []
    fold_gate_rows = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", f"wfo_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    pbar = tqdm(fold_periods, desc="WFO Progress")
    for period in pbar:
        fold_num, is_start, is_end, oos_start, oos_end = period.values()
        pbar.set_description(f"WFO Fold {fold_num}/{total_folds}")

        print(f"\n--- Fold {fold_num} IS Period: {is_start} ~ {is_end} ---")
        
        # [MODIFIED] 1. IS 기간의 "전체" 시뮬레이션 결과 확보
        _, is_simulation_results_df = find_optimal_parameters(
             start_date=is_start.strftime('%Y-%m-%d'),
             end_date=is_end.strftime('%Y-%m-%d'),
             initial_cash=initial_cash
        )
        print(f"  - IS simulation complete. Analyzing {len(is_simulation_results_df)} combinations.")

        if robust_selection_enabled:
            # [NEW] 2. 클러스터링으로 강건 파라미터 탐색
            robust_params_dict, clustered_df = find_robust_parameters(
                simulation_results_df=is_simulation_results_df,
                param_cols=['additional_buy_drop_rate', 'sell_profit_rate', 'stop_loss_rate', 'max_inactivity_period'],
                metric_cols=['cagr', 'mdd', 'calmar_ratio'],
                k_range=(2, 8),
                min_cluster_size_ratio=0.05,
                score_metric=robust_score_metric,
                score_k=robust_score_k,
            )
        else:
            robust_params_dict = _select_legacy_best_params(is_simulation_results_df)
            clustered_df = None
        
        # 디버깅을 위해 각 폴드의 클러스터링 결과 저장
        if clustered_df is not None:
            fold_cluster_path = os.path.join(results_dir, f"fold_{fold_num}_clustered_results.csv")
            clustered_df.to_csv(fold_cluster_path, index=False)
            print(f"  - Fold {fold_num} clustered analysis saved.")

        robust_params_dict['fold'] = fold_num
        all_optimal_params.append(robust_params_dict)
        print(f"  - Robust params for Fold {fold_num} selected.")
        
        if total_folds == 1:
            print("\n[INFO] Single fold run. OOS performance is same as IS robust parameter performance.")
            # 단일 폴드에서는 OOS 커브가 의미 없으므로 IS 결과를 사용 (혹은 생략)
            break

        print(f"--- Fold {fold_num} OOS Period: {oos_start} ~ {oos_end} ---")
        
        # 3. 찾은 파라미터로 OOS 기간 백테스트
        # OOS 기간의 초기 자금은 이전 OOS 기간의 최종 자금으로 연결
        oos_equity_curve = run_single_backtest(
             start_date=oos_start.strftime('%Y-%m-%d'),
             end_date=oos_end.strftime('%Y-%m-%d'),
             params_dict=robust_params_dict,
             initial_cash=initial_cash if not all_oos_curves else all_oos_curves[-1].iloc[-1]
         )
        all_oos_curves.append(oos_equity_curve)    

        if not oos_equity_curve.empty:
            oos_analyzer = PerformanceAnalyzer(pd.DataFrame(oos_equity_curve, columns=["total_value"]))
            oos_metrics = oos_analyzer.get_metrics(formatted=False)
        else:
            oos_metrics = {}

        fold_gate_rows.append(
            {
                "fold": fold_num,
                "selection_mode": selection_mode,
                f"is_{robust_gate_metric}": float(robust_params_dict.get(robust_gate_metric, np.nan)),
                f"oos_{robust_gate_metric}": float(oos_metrics.get(robust_gate_metric, np.nan)),
                "oos_mdd": float(oos_metrics.get("mdd", np.nan)),
            }
        )
            
    pbar.close()

    # 5. [수정] 최종 결과 종합 및 분석 (고도화)
    print("\n" + "="*80)
    print("📈 Walk-Forward Analysis Finished. Aggregating results...")
    print("="*80)

    if not all_oos_curves:
        print("[ERROR] No Out-of-Sample results were generated.")
        # 단일 폴드 실행 시 여기로 올 수 있으므로, 파라미터 분석만 수행
    else:
        final_wfo_curve = pd.concat(all_oos_curves).sort_index().groupby(level=0).mean()
        wfo_analyzer = PerformanceAnalyzer(pd.DataFrame(final_wfo_curve, columns=['total_value']))
        
        print("\n--- Final WFO Performance Metrics ---")
        for key, value in wfo_analyzer.get_metrics(formatted=True).items():
            print(f"  {key:<25}: {value}")
        
        curve_filepath = os.path.join(results_dir, "wfo_equity_curve_data.csv")
        final_wfo_curve.to_csv(curve_filepath)
        print(f"\n✅ Final WFO equity curve data saved to: {curve_filepath}")
        plot_wfo_results(final_wfo_curve, pd.DataFrame(all_optimal_params), results_dir)

   
   
    # 5-2. 파라미터 안정성 분석 및 결과 저장
    params_df = pd.DataFrame(all_optimal_params)
    print("\n📊 Optimal Parameter Stability Analysis (Descriptive Stats):")
    # 문자열 타입 파라미터는 제외하고 기술 통계 출력
    print(params_df.drop(columns=['additional_buy_priority'], errors='ignore').describe())
    
    params_filepath = os.path.join(results_dir, "wfo_robust_parameters.csv")
    params_df.to_csv(params_filepath, index=False)
   
    print(f"\n✅ Robust parameters for each fold saved to: {params_filepath}")

    if fold_gate_rows:
        fold_gate_df = pd.DataFrame(fold_gate_rows)
        fold_gate_report_df, gates_summary = apply_robust_gates(
            fold_gate_df,
            metric=robust_gate_metric,
            min_oos_is_ratio=gate_min_oos_is_ratio,
            min_fold_pass_rate=gate_min_fold_pass_rate,
            max_oos_mdd_p95=gate_max_oos_mdd_p95,
        )

        gate_report_path = os.path.join(results_dir, "wfo_gate_report.csv")
        fold_gate_report_df.to_csv(gate_report_path, index=False, float_format="%.6f")
        print(f"\n✅ Gate report saved to: {gate_report_path}")
        print(f"[Gate Summary] {gates_summary}")

if __name__ == '__main__':
    run_walk_forward_analysis()
