# src/walk_forward_analyzer.py

from __future__ import annotations

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/walk_forward_analyzer.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from .config_loader import load_config

if TYPE_CHECKING:
    import pandas as pd

# --- Clustering Helper Function ---
def find_robust_parameters(
    simulation_results_df: "pd.DataFrame",
    param_cols: list,
    metric_cols: list,
    k_range: tuple = (2, 11),
    min_cluster_size_ratio: float = 0.05
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
    
    # Z-score 계산 시 분모가 0이 되는 경우 방지
    calmar_std = cluster_summary['calmar_ratio'].std()
    denominator = calmar_std if calmar_std > 0 else 1
    cluster_summary['calmar_zscore'] = (cluster_summary['calmar_ratio'] - cluster_summary['calmar_ratio'].mean()) / denominator
    cluster_summary['robustness_score'] = cluster_summary['calmar_zscore'] * np.log1p(cluster_summary['size'])
    
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
    
    return best_params_series.to_dict(), clustered_df_full
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
        
        # [NEW] 2. 클러스터링으로 강건 파라미터 탐색
        robust_params_dict, clustered_df = find_robust_parameters(
            simulation_results_df=is_simulation_results_df,
            param_cols=['additional_buy_drop_rate', 'sell_profit_rate', 'stop_loss_rate', 'max_inactivity_period'],
            metric_cols=['cagr', 'mdd', 'calmar_ratio'],
            k_range=(2, 8),
            min_cluster_size_ratio=0.05
        )
        
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

if __name__ == '__main__':
    run_walk_forward_analysis()
