import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from indicator_calculator import calculate_indicators

class TestIndicatorCalculator(unittest.TestCase):

    def setUp(self):
        """테스트에 사용할 OHLCV 샘플 데이터프레임 생성"""
        dates = pd.date_range(start='2022-01-01', periods=30, freq='D')
        data = {
            'open_price': np.random.uniform(98, 102, size=30),
            'high_price': np.random.uniform(103, 105, size=30),
            'low_price': np.random.uniform(95, 97, size=30),
            'close_price': np.arange(100, 130), # 예측 가능한 값을 위해 간단한 시퀀스 사용
            'volume': np.random.randint(1000, 5000, size=30)
        }
        self.ohlcv_df = pd.DataFrame(data, index=dates)
        # 롤링 계산을 위해 데이터 추가
        more_dates = pd.date_range(start='2021-01-01', periods=252*2, freq='B')
        more_data = {
            'open_price': 100, 'high_price': 105, 'low_price': 95, 'close_price': np.linspace(50, 99, len(more_dates)), 'volume': 2000
        }
        more_df = pd.DataFrame(more_data, index=more_dates)
        
        # 실제 롤링 계산에 필요한 충분한 데이터를 만들기 위해 과거 데이터를 합침
        past_dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='B')
        past_data = {
            'open_price': np.random.uniform(80, 90, size=len(past_dates)),
            'high_price': np.random.uniform(90, 100, size=len(past_dates)),
            'low_price': np.random.uniform(70, 80, size=len(past_dates)),
            'close_price': np.linspace(50, 100, len(past_dates)),
            'volume': np.random.randint(1000, 5000, size=len(past_dates))
        }
        self.ohlcv_df_full = pd.concat([pd.DataFrame(past_data, index=past_dates), self.ohlcv_df])


    def test_calculate_indicators_empty_df(self):
        """빈 데이터프레임이 입력될 때 빈 데이터프레임을 반환하는지 테스트"""
        indicators = calculate_indicators(pd.DataFrame())
        self.assertTrue(indicators.empty)

    def test_calculate_ma(self):
        """이동평균(MA)이 정확하게 계산되는지 테스트"""
        # min_periods를 고려하여 충분한 데이터로 테스트
        close_prices = pd.Series(np.arange(1, 21))
        # ATR 계산에 필요한 high, low 컬럼 추가
        df = pd.DataFrame({
            'high_price': close_prices + 1,
            'low_price': close_prices - 1,
            'close_price': close_prices
        })
        
        indicators = calculate_indicators(df)
        
        # 5일 이동평균
        self.assertAlmostEqual(indicators['ma_5'].iloc[4], 3.0)
        self.assertAlmostEqual(indicators['ma_5'].iloc[19], 18.0)
        # min_periods=5 이므로 처음 4개는 NaN
        self.assertTrue(pd.isna(indicators['ma_5'].iloc[3]))
        
        # 20일 이동평균
        self.assertAlmostEqual(indicators['ma_20'].iloc[19], 10.5)
        self.assertTrue(pd.isna(indicators['ma_20'].iloc[18]))

    def test_calculate_atr_ratio(self):
        """ATR 및 ATR 비율이 정확하게 계산되는지 테스트"""
        # ATR 계산을 위한 간단한 데이터
        data = {
            'high_price': [10, 12, 11, 13, 14],
            'low_price': [8, 9, 10, 11, 12],
            'close_price': [9, 11, 10.5, 12.5, 13.5]
        }
        df = pd.DataFrame(data, index=pd.date_range('2023-01-01', periods=5))
        
        # TR 계산: [H-L, |H-prev_C|, |L-prev_C|]의 max
        # 1일차: 10-8 = 2
        # 2일차: max(12-9, |12-9|, |9-9|) = max(3, 3, 0) = 3
        # 3일차: max(11-10, |11-11|, |10-11|) = max(1, 0, 1) = 1
        # 4일차: max(13-11, |13-10.5|, |11-10.5|) = max(2, 2.5, 0.5) = 2.5
        # 5일차: max(14-12, |14-12.5|, |12-12.5|) = max(2, 1.5, 0.5) = 2
        # 이 데이터로는 min_periods=14를 만족 못해 ATR이 NaN이 됨

        # 충분한 데이터로 테스트
        df_long = pd.DataFrame({
            'high_price': np.full(20, 12),
            'low_price': np.full(20, 8),
            'close_price': np.full(20, 10)
        })
        
        indicators = calculate_indicators(df_long)
        
        # TR은 매일 4 (12-8) 또는 2 (|12-10|, |8-10|) -> 4
        # 14일 ATR은 4가 되어야 함
        expected_atr = 4.0
        expected_atr_ratio = expected_atr / 10.0 # ATR / close_price

        self.assertAlmostEqual(indicators['atr_14_ratio'].iloc[-1], expected_atr_ratio)


    def test_price_position_ratios(self):
        """주가 위치 비율 지표 (5년, 10년) 계산을 테스트"""
        # 이 테스트는 긴 기간의 데이터가 필요함
        # setUp에서 생성한 self.ohlcv_df_full 사용
        indicators = calculate_indicators(self.ohlcv_df_full)
        
        # 결과가 0과 1 사이에 있는지 확인
        self.assertTrue((indicators['price_vs_5y_low_pct'].dropna() >= 0).all())
        self.assertTrue((indicators['price_vs_5y_low_pct'].dropna() <= 1).all())
        self.assertTrue((indicators['price_vs_10y_low_pct'].dropna() >= 0).all())
        self.assertTrue((indicators['price_vs_10y_low_pct'].dropna() <= 1).all())

        # 마지막 값 직접 계산하여 비교
        last_close = self.ohlcv_df_full['close_price'].iloc[-1]
        
        # 5년치 데이터
        df_5y = self.ohlcv_df_full.iloc[-252*5:]
        low_5y = df_5y['low_price'].min()
        high_5y = df_5y['high_price'].max()
        expected_ratio_5y = (last_close - low_5y) / (high_5y - low_5y)

        # 10년치 데이터 (여기선 전체 데이터가 10년이 안될 수 있음)
        df_10y = self.ohlcv_df_full.iloc[-252*10:]
        low_10y = df_10y['low_price'].min()
        high_10y = df_10y['high_price'].max()
        expected_ratio_10y = (last_close - low_10y) / (high_10y - low_10y)
        
        # clip(0,1) 적용
        expected_ratio_5y = np.clip(expected_ratio_5y, 0, 1)
        expected_ratio_10y = np.clip(expected_ratio_10y, 0, 1)
        
        self.assertAlmostEqual(indicators['price_vs_5y_low_pct'].iloc[-1], expected_ratio_5y)
        self.assertAlmostEqual(indicators['price_vs_10y_low_pct'].iloc[-1], expected_ratio_10y)


if __name__ == '__main__':
    unittest.main()
