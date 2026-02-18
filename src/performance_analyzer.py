"""
performance_analyzer.py

This module contains the functions for analyzing the performance of the Magic Split Strategy.
"""

import sys
from pathlib import Path

# BOOTSTRAP: allow direct execution (`python src/performance_analyzer.py`) while keeping package imports.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    file_path = Path(__file__).resolve()
    sys.path.insert(0, str(file_path.parent.parent))
    __package__ = file_path.parent.name  # "src"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PerformanceAnalyzer:
    """
    백테스팅 결과를 분석하고 주요 성과 지표를 계산하며,
    시각적 리포트를 생성하는 클래스.
    """
    def __init__(self, history_df: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        분석기 클래스를 초기화합니다.

        Args:
            history_df (pd.DataFrame): 인덱스가 날짜(datetime)이고 'total_value' 컬럼을 포함하는 DataFrame.
            risk_free_rate (float, optional): 연간 무위험 수익률. 기본값은 2% (0.02).
        """
        # --- 올바르게 수정된 오류 체크 로직 ---
        if not isinstance(history_df, pd.DataFrame) or history_df.empty or 'total_value' not in history_df.columns:
            raise ValueError("history_df는 'total_value' 컬럼을 포함하는 비어있지 않은 pandas DataFrame이어야 합니다.")
        
        self.history_df = history_df
        self.daily_values = self.history_df['total_value']
        
        # daily_values가 비어있는 경우를 한번 더 체크
        if self.daily_values.empty:
            raise ValueError("'total_value' 컬럼에 유효한 데이터가 없습니다.")

        self.risk_free_rate = risk_free_rate
        self.daily_returns = self.daily_values.pct_change().dropna()
        
        # 주요 지표들을 계산하여 인스턴스 변수에 저장
        self.metrics = self._calculate_all_metrics()
    def _calculate_all_metrics(self) -> dict:
        """
        모든 핵심 성과 지표를 계산하여 딕셔너리로 반환하는 내부 메서드.
        """
        if len(self.daily_values) < 2:
            return {}

        # 기본 변수
        total_days = len(self.daily_values)
        years = total_days / 252.0  # 1년 영업일 기준

        # 수익률 지표
        initial_value = self.daily_values.iloc[0]
        final_value = self.daily_values.iloc[-1]
        final_cumulative_returns = (final_value / initial_value) - 1
        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0

        # 위험도 지표
        annualized_volatility = self.daily_returns.std() * np.sqrt(252)
        cumulative_max = self.daily_values.cummax()
        drawdown = (self.daily_values - cumulative_max) / cumulative_max
        mdd = drawdown.min()

        # 위험 조정 수익률
        daily_risk_free = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_returns = self.daily_returns - daily_risk_free
        
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
        
        downside_returns = self.daily_returns[self.daily_returns < daily_risk_free]
        downside_std = downside_returns.std()
        sortino_ratio = (self.daily_returns.mean() - daily_risk_free) / downside_std * np.sqrt(252) if downside_std != 0 else 0
        
        calmar_ratio = cagr / abs(mdd) if mdd != 0 else 0

        return {
            'period_start': self.daily_values.index.min().strftime('%Y-%m-%d'),
            'period_end': self.daily_values.index.max().strftime('%Y-%m-%d'),
            'initial_value': initial_value,
            'final_value': final_value,
            'final_cumulative_returns': final_cumulative_returns,
            'cagr': cagr,
            'annualized_volatility': annualized_volatility,
            'mdd': mdd,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }

    def get_metrics(self, formatted: bool = True) -> dict:
        """
        계산된 성과 지표를 반환합니다.

        Args:
            formatted (bool): True이면 사람이 보기 좋은 포맷(%)으로, False이면 원본 숫자 값으로 반환.

        Returns:
            dict: 성과 지표 딕셔너리.
        """
        if not formatted:
            return self.metrics

        return {
            "분석 기간": f"{self.metrics['period_start']} ~ {self.metrics['period_end']}",
            "초기 자산": f"{self.metrics['initial_value']:,.0f} 원",
            "최종 자산": f"{self.metrics['final_value']:,.0f} 원",
            "최종 누적 수익률": f"{self.metrics['final_cumulative_returns']:.2%}",
            "연평균 복리 수익률 (CAGR)": f"{self.metrics['cagr']:.2%}",
            "연간 변동성": f"{self.metrics['annualized_volatility']:.2%}",
            "최대 낙폭 (MDD)": f"{self.metrics['mdd']:.2%}",
            "샤프 지수": f"{self.metrics['sharpe_ratio']:.2f}",
            "소르티노 지수": f"{self.metrics['sortino_ratio']:.2f}",
            "칼마 지수": f"{self.metrics['calmar_ratio']:.2f}"
        }

    def plot_equity_curve(self, title: str = 'Portfolio Equity Curve', save_path: str = None):
        """
        수익 곡선 및 MDD를 시각화합니다.
        
        Args:
            title (str): 그래프 제목.
            save_path (str, optional): 그래프를 저장할 파일 경로. None이면 저장하지 않음.
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 한글 폰트 설정 (환경에 맞는 폰트 경로 지정 필요)
        # plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

        # 1. Equity Curve
        ax1.set_title(title, fontsize=16)
        ax1.plot(self.daily_values.index, self.daily_values, color='b', label='Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend(loc='upper left')
        
        # 2. Drawdown
        drawdown = (self.daily_values - self.daily_values.cummax()) / self.daily_values.cummax()
        ax2.fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend(loc='upper left')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"그래프가 '{save_path}'에 저장되었습니다.")
        plt.show()

# --- 사용 예시 ---
if __name__ == '__main__':
    # 1. 샘플 데이터 생성
    np.random.seed(42)
    days = 252 * 5
    dates = pd.to_datetime(pd.date_range('2020-01-01', periods=days, freq='B'))
    initial_value = 100_000_000
    returns = pd.Series(np.random.randn(days) / 100 + 0.0008, index=dates)
    daily_values_sample = initial_value * (1 + returns).cumprod()
    daily_values_sample.name = "PortfolioValue"

    # 2. PerformanceAnalyzer 인스턴스 생성
    try:
        analyzer = PerformanceAnalyzer(daily_values_sample)

        # 3. 포맷팅된 지표 출력
        print("--- Formatted Performance Metrics ---")
        formatted_metrics = analyzer.get_metrics(formatted=True)
        for key, value in formatted_metrics.items():
            print(f"{key:<25}: {value}")

        print("\n--- Raw Performance Metrics for Sorting ---")
        raw_metrics = analyzer.get_metrics(formatted=False)
        print(raw_metrics)

        # 4. 시각화
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analyzer.plot_equity_curve(
            title="Sample Strategy Performance", 
            save_path=f"equity_curve_{timestamp}.png"
        )
    except ValueError as e:
        print(f"오류 발생: {e}")
