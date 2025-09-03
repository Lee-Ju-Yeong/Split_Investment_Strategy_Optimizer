# src/walk_forward_analyzer.py

import pandas as pd
from datetime import timedelta, datetime
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns

from .config_loader import load_config
# 실제 워커 함수 및 분석 모듈 임포트
from .parameter_simulation_gpu import find_optimal_parameters
from .debug_gpu_single_run import run_single_backtest
from .performance_analyzer import PerformanceAnalyzer

# --- 분석 및 시각화 헬퍼 함수 ---
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


    # 2. [핵심] 모든 기간 파라미터 자동 계산
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🚀 Starting Walk-Forward Optimization (Fully Automated Period Calculation)")

    # 사용자 설정값 추출
    total_start_date = pd.to_datetime(backtest_settings['start_date'])
    total_end_date = pd.to_datetime(backtest_settings['end_date'])
    total_folds = wfo_settings['total_folds']
    period_length_days = wfo_settings['period_length_days']
    
    if total_folds < 1:
        raise ValueError("total_folds must be 1 or greater.")
    if total_folds == 1:
            print("  [INFO] total_folds is set to 1. Running a single optimization for the whole period.")

    # 자동 계산
    total_duration = total_end_date - total_start_date
    period_delta = timedelta(days=period_length_days)

    # 단일 기간 길이가 전체 기간보다 길면 실행 불가
    if period_delta >= total_duration:
        raise ValueError("period_length_days cannot be longer than the total duration.")
    
    # [최종 로직] 제약조건을 만족하는 '제어된 겹침' WFO 기간 계산
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("🚀 Starting Walk-Forward Optimization (Constraint-based Controlled Overlap)")

    # 사용자 설정값 추출
    total_start_date = pd.to_datetime(backtest_settings['start_date'])
    total_end_date = pd.to_datetime(backtest_settings['end_date'])
    total_folds = wfo_settings['total_folds']
    period_length_days = wfo_settings['period_length_days']

    if total_folds < 1:
        raise ValueError("total_folds must be 1 or greater.")
    if total_folds == 1:
            print("  [INFO] total_folds is set to 1. Running a single optimization for the whole period.")

    period_delta = timedelta(days=period_length_days)
    total_duration = total_end_date - total_start_date

    if period_delta >= total_duration:
        raise ValueError(f"period_length_days ({period_length_days}d) cannot be longer than or equal to the total duration ({total_duration.days}d).")

    # [핵심] 모든 Fold 기간을 미리 계산하여 리스트에 저장
    fold_periods = []
    if total_folds == 1:
        fold_periods.append({
            'Fold': 1,
            'IS_Start': total_start_date.date(), 'IS_End': total_end_date.date(),
            'OOS_Start': None, 'OOS_End': None
        })
    else:
        # step_delta는 한 Fold에서 다음 Fold로의 전진 거리(새로운 데이터 기간)를 의미
        # (전체 기간 - 1개 Fold 길이)를 총 Fold 수로 나누어 계산
        step_delta = (total_duration - period_delta) / total_folds
        
        if step_delta.days < 1:
            print(f"[WARNING] step_delta is {step_delta.total_seconds() / 86400:.2f} days, which is less than 1.")
            print("This implies very heavy overlap. Consider reducing period_length_days or total_folds.")

        current_is_start = total_start_date
        for i in range(total_folds):
            is_start = current_is_start
            is_end = is_start + period_delta
            
            oos_start = is_start + step_delta
            oos_end = oos_start + period_delta
            
            fold_periods.append({
                'Fold': i + 1,
                'IS_Start': is_start.date(), 'IS_End': is_end.date(),
                'OOS_Start': oos_start.date(), 'OOS_End': oos_end.date()
            })
            
            # 다음 Fold의 IS 시작점은 현재 IS 시작점에서 step_delta만큼 이동
            current_is_start += step_delta
    # [추가] 계산된 Fold 기간을 표로 출력하여 사전 확인
    print("\n--- Calculated Walk-Forward Folds ---")
    folds_df = pd.DataFrame(fold_periods)
    print(folds_df.to_string(index=False))
    print("="*80)

    # 3. 결과 저장을 위한 리스트 초기화
    all_oos_curves = []
    all_optimal_params = []
    
    # 4. 새로운 롤링 윈도우 루프
    pbar = tqdm(fold_periods, desc="WFO Progress")
    for period in pbar:
        fold_num = period['Fold']
        is_start = period['IS_Start']
        is_end = period['IS_End']
        oos_start = period['OOS_Start']
        oos_end = period['OOS_End']
        
        # OOS 기간이 없는 경우 (total_folds=1) 건너뛰기
        if oos_start is None:
            print("\n[INFO] Skipping OOS backtest as only one fold is defined.")
            continue

        pbar.set_description(f"WFO Progress | IS: {is_start}->{is_end}")

        print(f"\n--- Fold {fold_num} {'-'*65}")
        print(f"  In-Sample Period (IS)  : {is_start} ~ {is_end}")
        print(f"  Out-of-Sample Period (OOS): {oos_start} ~ {oos_end}")
        print("-"*(72))

        # 4-1. IS 기간으로 최적 파라미터 탐색
        optimal_params_dict, _ = find_optimal_parameters(
            start_date=is_start.strftime('%Y-%m-%d'),
            end_date=is_end.strftime('%Y-%m-%d'),
            initial_cash=initial_cash
        )
        optimal_params_dict['fold'] = fold_num
        all_optimal_params.append(optimal_params_dict)

        # [개선] 출력을 보기 좋게 포맷팅합니다.
        formatted_params_str = ", ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
            for k, v in optimal_params_dict.items()
        ])
        print(f"  [Orchestrator] Found optimal params for Fold {fold_num}: {{{formatted_params_str}}}")

        # 4-2. 찾은 파라미터로 OOS 기간 백테스트 실행
        oos_equity_curve = run_single_backtest(
            start_date=oos_start.strftime('%Y-%m-%d'),
            end_date=oos_end.strftime('%Y-%m-%d'),
            params_dict=optimal_params_dict,
            initial_cash=initial_cash
        )
        all_oos_curves.append(oos_equity_curve)
        print(f"  [Orchestrator] Completed OOS backtest for Fold {fold_num}. OOS curve length: {len(oos_equity_curve)}")
            
    pbar.close()

    # 5. [수정] 최종 결과 종합 및 분석 (고도화)
    print("\n" + "="*80)
    print("📈 Walk-Forward Optimization Finished. Aggregating results...")
    print("="*80)

    if not all_oos_curves or all(s.empty for s in all_oos_curves):
        print("[ERROR] No Out-of-Sample results were generated.")
        return

    # 5-1. OOS 수익 곡선 연결 및 성과 분석
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
    
    # 5-2. 파라미터 안정성 분석 및 결과 저장
    params_df = pd.DataFrame(all_optimal_params)
    print("\n📊 Optimal Parameter Stability Analysis (Descriptive Stats):")
    # 문자열 타입 파라미터는 제외하고 기술 통계 출력
    print(params_df.drop(columns=['additional_buy_priority'], errors='ignore').describe())
    
    # 5-3. 결과 파일 저장
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

    # 5-4. 시각화
    plot_wfo_results(final_wfo_curve, params_df, results_dir)

if __name__ == '__main__':
    run_walk_forward_analysis()