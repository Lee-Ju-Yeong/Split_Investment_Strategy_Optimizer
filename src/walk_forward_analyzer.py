# src/walk_forward_analyzer.py

import pandas as pd
from datetime import timedelta, datetime
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

from .config_loader import load_config
# [수정] 실제 워커 함수 및 분석 모듈 임포트
from .parameter_simulation_gpu import find_optimal_parameters
from .debug_gpu_single_run import run_single_backtest
from .performance_analyzer import PerformanceAnalyzer

# --- [신규] 분석 및 시각화 헬퍼 함수 ---
def plot_wfo_results(final_curve: pd.Series, params_df: pd.DataFrame, results_dir: str):
    """최종 WFO 결과(수익곡선, 파라미터 분포)를 시각화하고 저장합니다."""
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
    ax1.set_ylabel('Portfolio Value')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Drawdown
    drawdown = (analyzer.daily_values - analyzer.daily_values.cummax()) / analyzer.daily_values.cummax()
    ax2.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    equity_curve_path = os.path.join(results_dir, "wfo_equity_curve.png")
    plt.savefig(equity_curve_path, dpi=300)
    plt.close()
    print(f"✅ WFO Equity Curve plot saved to: {equity_curve_path}")

    # 2. 파라미터 안정성(분포) 플롯
    numeric_params = params_df.select_dtypes(include='number').columns.drop('fold', errors='ignore')
    num_params = len(numeric_params)
    if num_params == 0:
        print("[INFO] No numeric parameters to plot for stability analysis.")
        return

    cols = 3
    rows = (num_params + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, param in enumerate(numeric_params):
        sns.histplot(data=params_df, x=param, ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {param}')
    
    # 남는 subplot 숨기기
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    param_dist_path = os.path.join(results_dir, "wfo_parameter_distribution.png")
    plt.savefig(param_dist_path, dpi=300)
    plt.close()
    print(f"✅ Parameter Distribution plot saved to: {param_dist_path}")

# --- Orchestrator 메인 로직 ---

def run_walk_forward_analysis():
    """
    Walk-Forward Optimization 프로세스 전체를 총괄하는 오케스트레이터 함수.
    """
    # 1. 설정 로드
    config = load_config()
    wfo_settings = config['walk_forward_settings']
    backtest_settings = config['backtest_settings']
    
    initial_cash = backtest_settings['initial_cash'] 

    # 기간 설정
    total_start_date = pd.to_datetime(backtest_settings['start_date'])
    total_end_date = pd.to_datetime(backtest_settings['end_date'])
    
    in_sample_delta = timedelta(days=wfo_settings['in_sample_period_days'])
    out_of_sample_delta = timedelta(days=wfo_settings['out_of_sample_period_days'])
    step_delta = timedelta(days=wfo_settings['step_size_days'])

    # 2. 결과 저장을 위한 리스트 초기화
    all_oos_curves = []
    all_optimal_params = []
    
    # 3. 롤링 윈도우 루프 실행
    current_start = total_start_date
    fold_num = 1

    print("\n" + "="*80)
    print("🚀 Starting Walk-Forward Optimization Analysis")
    print("="*80)
    
    # tqdm을 사용하여 전체 진행 상황 표시
    # 전체 기간을 step_size로 나누어 대략적인 총 Fold 수를 계산
    total_folds_approx = (total_end_date - total_start_date) // step_delta
    pbar = tqdm(total=total_folds_approx, desc="WFO Progress")

    while True:
        # Fold별 기간 계산
        in_sample_end = current_start + in_sample_delta
        out_of_sample_start = in_sample_end + timedelta(days=1)
        out_of_sample_end = out_of_sample_start + out_of_sample_delta

        # 검증 기간(OOS)이 전체 백테스트 기간을 넘어서면 루프 종료
        if out_of_sample_end > total_end_date:
            out_of_sample_end = total_end_date
            print(f"\n[INFO] Final Out-of-Sample period adjusted to total end date: {out_of_sample_end.date()}")
            # 조정된 OOS 기간이 IS 기간보다 이전이면 분석 종료
            if out_of_sample_start >= out_of_sample_end:
                 print(f"\n[INFO] Adjusted OOS start date is after end date. Stopping analysis.")
                 break

        print(f"\n--- Fold {fold_num} {'-'*65}")
        print(f"  In-Sample Period (IS)  : {current_start.date()} ~ {in_sample_end.date()}")
        print(f"  Out-of-Sample Period (OOS): {out_of_sample_start.date()} ~ {out_of_sample_end.date()}")
        print("-"*(72))


        
        # 3-1. IS 기간으로 최적 파라미터 탐색
        optimal_params = find_optimal_parameters(
            start_date=current_start.strftime('%Y-%m-%d'),
            end_date=in_sample_end.strftime('%Y-%m-%d'),
            initial_cash=initial_cash
        )
        optimal_params['fold'] = fold_num
        all_optimal_params.append(optimal_params)
        print(f"  [Orchestrator] Found optimal params for Fold {fold_num}: {optimal_params}")

        # 3-2. 찾은 파라미터로 OOS 기간 백테스트 실행
        oos_equity_curve = run_single_backtest(
            start_date=out_of_sample_start.strftime('%Y-%m-%d'),
            end_date=out_of_sample_end.strftime('%Y-%m-%d'),
            params_dict=optimal_params,
            initial_cash=initial_cash
        )
        all_oos_curves.append(oos_equity_curve)
        print(f"  [Orchestrator] Completed OOS backtest for Fold {fold_num}. OOS curve length: {len(oos_equity_curve)}")

        # 마지막 Fold를 처리했으므로 루프 종료
        if out_of_sample_end == total_end_date:
            pbar.update(1)
            break
            
        # 다음 Fold를 위해 윈도우 이동
        current_start += step_delta
        fold_num += 1
        pbar.update(1)
        
    pbar.close()

    # 4. [수정] 최종 결과 종합 및 분석 (고도화)
    print("\n" + "="*80)
    print("📈 Walk-Forward Optimization Finished. Aggregating results...")
    print("="*80)

    if not all_oos_curves or all(s.empty for s in all_oos_curves):
        print("[ERROR] No Out-of-Sample results were generated.")
        return

    # 4-1. OOS 수익 곡선 연결 및 성과 분석
    final_wfo_curve = pd.concat(all_oos_curves).sort_index()
    # 중복 인덱스 발생 시 평균값 사용 (거의 발생하지 않음)
    final_wfo_curve = final_wfo_curve.groupby(final_wfo_curve.index).mean()
    
    print("✅ Successfully stitched all OOS equity curves.")
    
    wfo_history_df = pd.DataFrame(final_wfo_curve, columns=['total_value'])
    wfo_analyzer = PerformanceAnalyzer(wfo_history_df)
    wfo_metrics = wfo_analyzer.get_metrics(formatted=True)

    print("\n--- Final WFO Performance Metrics ---")
    for key, value in wfo_metrics.items():
        print(f"  {key:<25}: {value}")
    
    # 4-2. 파라미터 안정성 분석 및 결과 저장
    params_df = pd.DataFrame(all_optimal_params)
    print("\n📊 Optimal Parameter Stability Analysis (Descriptive Stats):")
    # 문자열 타입 파라미터는 제외하고 기술 통계 출력
    print(params_df.drop(columns=['additional_buy_priority'], errors='ignore').describe())
    
    # 4-3. 결과 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", f"wfo_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 파라미터 기록 저장
    params_filepath = os.path.join(results_dir, "wfo_optimal_parameters.csv")
    params_df.to_csv(params_filepath, index=False)
    print(f"\n✅ Optimal parameters for each fold saved to: {params_filepath}")
    
    # 최종 수익 곡선 데이터 저장
    curve_filepath = os.path.join(results_dir, "wfo_equity_curve_data.csv")
    final_wfo_curve.to_csv(curve_filepath)
    print(f"✅ Final WFO equity curve data saved to: {curve_filepath}")

    # 4-4. 시각화
    plot_wfo_results(final_wfo_curve, params_df, results_dir)

if __name__ == '__main__':
    run_walk_forward_analysis()