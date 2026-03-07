import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date
import sys
import os

# repo root를 sys.path에 추가하여 canonical package import를 사용
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_handler import DataHandler

class TestDataHandler(unittest.TestCase):

    def setUp(self):
        """테스트 케이스 실행 전 설정"""
        self.db_config = {
            'host': 'fake_host',
            'user': 'fake_user',
            'password': 'fake_password',
            'database': 'fake_db',
        }
        # DataHandler의 __init__에서 connection_pool 생성을 mock 처리
        self.pool_patcher = patch('mysql.connector.pooling.MySQLConnectionPool')
        self.mock_pool = self.pool_patcher.start()

        # get_connection이 반환할 mock connection과 cursor 설정
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_pool.return_value.get_connection.return_value = self.mock_conn
        self.mock_conn.cursor.return_value = self.mock_cursor

        self.data_handler = DataHandler(self.db_config)
        # 각 테스트의 독립성을 위해 lru_cache를 초기화
        self.data_handler.clear_load_stock_data_cache()


    def tearDown(self):
        """테스트 케이스 실행 후 정리"""
        self.pool_patcher.stop()

    @patch('pandas.read_sql')
    def test_get_trading_dates(self, mock_read_sql):
        """주어진 기간 내의 거래일 목록을 올바르게 가져오는지 테스트"""
        expected_dates_raw = ['2022-01-03', '2022-01-04', '2022-01-05']
        mock_read_sql.return_value = pd.DataFrame({'date': pd.to_datetime(expected_dates_raw)})
        
        # 테스트 대상 메서드 호출
        start_date_str = '2022-01-01'
        end_date_str = '2022-01-31'
        trading_dates = self.data_handler.get_trading_dates(start_date_str, end_date_str)
        
        # 결과 검증
        expected_dates = [pd.Timestamp(d) for d in expected_dates_raw]
        self.assertEqual(trading_dates, expected_dates)
        mock_read_sql.assert_called_once()
        _, kwargs = mock_read_sql.call_args
        self.assertEqual(kwargs['params'], (start_date_str, end_date_str))

    def test_get_trading_dates_adjusted_gate_raises_before_start(self):
        with self.assertRaises(ValueError):
            self.data_handler.get_trading_dates('2013-01-01', '2013-12-31')

    @patch('pandas.read_sql')
    def test_get_trading_dates_raw_mode_allows_pre_gate(self, mock_read_sql):
        raw_handler = DataHandler(self.db_config, price_basis="raw")
        raw_handler.clear_load_stock_data_cache()
        mock_read_sql.return_value = pd.DataFrame({'date': pd.to_datetime(['2010-01-04'])})

        trading_dates = raw_handler.get_trading_dates('2010-01-01', '2010-01-31')

        self.assertEqual(trading_dates, [pd.Timestamp('2010-01-04')])

    @patch('pandas.read_sql')
    def test_load_stock_data(self, mock_read_sql):
        """특정 종목의 주식 데이터를 올바르게 로드하는지 테스트"""
        # pd.read_sql이 반환할 모의 DataFrame 설정
        d = {
            'date': pd.to_datetime(['2022-01-03', '2022-01-04', '2022-01-05']),
            'open_price': [101, 102, 103],
            'high_price': [103, 104, 105],
            'low_price': [100, 101, 102],
            'close_price': [102, 103, 104],
            # ... 다른 컬럼들
        }
        mock_df = pd.DataFrame(data=d)
        mock_read_sql.return_value = mock_df.copy()
        
        # 테스트 대상 메서드 호출
        stock_data = self.data_handler.load_stock_data('005930', '2022-01-04', '2022-01-05')
        
        # 결과 검증
        self.assertIsInstance(stock_data, pd.DataFrame)
        self.assertEqual(list(stock_data.index), [pd.Timestamp('2022-01-04'), pd.Timestamp('2022-01-05')])
        self.assertEqual(stock_data.shape[0], 2)

    @patch('pandas.read_sql')
    def test_load_stock_data_empty(self, mock_read_sql):
        """데이터가 없을 경우 빈 DataFrame을 반환하는지 테스트"""
        mock_read_sql.return_value = pd.DataFrame()
        
        stock_data = self.data_handler.load_stock_data('000000', '2022-01-01', '2022-01-31')
        
        self.assertTrue(stock_data.empty)

    @patch('pandas.read_sql')
    def test_load_stock_data_adjusted_query_contains_adj_fields(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            {
                'date': pd.to_datetime(['2022-01-03']),
                'open_price': [100.0],
                'high_price': [101.0],
                'low_price': [99.0],
                'close_price': [100.0],
            }
        )

        self.data_handler.load_stock_data('005930', '2022-01-03', '2022-01-03')

        query = mock_read_sql.call_args.args[0]
        self.assertIn("dsp.adj_ratio", query)
        self.assertIn("dsp.adj_close AS close_price", query)

    @patch('pandas.read_sql')
    def test_load_stock_data_adjusted_query_uses_stored_adj_ohlc_with_stale_guard(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            {
                'date': pd.to_datetime(['2022-01-03']),
                'open_price': [100.0],
                'high_price': [101.0],
                'low_price': [99.0],
                'close_price': [100.0],
            }
        )
        self.data_handler.has_stored_adj_ohlc = True
        self.data_handler.clear_load_stock_data_cache()

        self.data_handler.load_stock_data('005930', '2022-01-03', '2022-01-03')

        query = mock_read_sql.call_args.args[0]
        self.assertIn("CASE", query)
        self.assertIn("ABS(dsp.adj_open - (dsp.open_price * dsp.adj_ratio)) > 1e-5", query)
        self.assertIn("ABS(dsp.adj_high - (dsp.high_price * dsp.adj_ratio)) > 1e-5", query)
        self.assertIn("ABS(dsp.adj_low - (dsp.low_price * dsp.adj_ratio)) > 1e-5", query)
        self.assertIn("dsp.adj_open", query)
        self.assertIn("dsp.adj_high", query)
        self.assertIn("dsp.adj_low", query)

    @patch('pandas.read_sql')
    def test_load_stock_data_adjusted_mode_ignores_pre_start_null_ohlc(self, mock_read_sql):
        mock_read_sql.return_value = pd.DataFrame(
            {
                'date': pd.to_datetime(['2006-12-19', '2014-01-03']),
                'open_price': [float('nan'), 12000.0],
                'high_price': [float('nan'), 12100.0],
                'low_price': [float('nan'), 11900.0],
                'close_price': [float('nan'), 12050.0],
            }
        )

        stock_data = self.data_handler.load_stock_data('000060', '2014-01-03', '2014-01-03')

        self.assertEqual(list(stock_data.index), [pd.Timestamp('2014-01-03')])
        self.assertFalse(stock_data.isna().any().any())

    @patch('pandas.read_sql')
    def test_load_stock_data_cache_key_is_normalized(self, mock_read_sql):
        d = {
            'date': pd.to_datetime(['2022-01-03', '2022-01-04', '2022-01-05']),
            'open_price': [101, 102, 103],
            'high_price': [103, 104, 105],
            'low_price': [100, 101, 102],
            'close_price': [102, 103, 104],
        }
        mock_read_sql.return_value = pd.DataFrame(data=d)

        stock_data_str = self.data_handler.load_stock_data('005930', '2022-01-04', '2022-01-05')
        stock_data_ts = self.data_handler.load_stock_data(
            '005930',
            pd.Timestamp('2022-01-04'),
            pd.Timestamp('2022-01-05'),
        )

        self.assertEqual(mock_read_sql.call_count, 1)
        self.assertTrue(stock_data_str.equals(stock_data_ts))

    @patch('pandas.read_sql')
    def test_load_stock_data_cache_key_includes_universe_mode(self, mock_read_sql):
        def _make_df():
            return pd.DataFrame(
                {
                    'date': pd.to_datetime(['2022-01-03']),
                    'open_price': [101],
                    'high_price': [103],
                    'low_price': [100],
                    'close_price': [102],
                }
            )
        mock_read_sql.side_effect = [_make_df(), _make_df()]
        self.data_handler.load_stock_data('005930', '2022-01-03', '2022-01-03')
        self.assertEqual(mock_read_sql.call_count, 1)

        self.data_handler.universe_mode = "strict_pit"
        self.data_handler.load_stock_data('005930', '2022-01-03', '2022-01-03')
        self.assertEqual(
            mock_read_sql.call_count,
            2,
            msg="cache key must include universe_mode to avoid cross-mode cache reuse",
        )

    def test_get_latest_price(self):
        """특정 날짜의 최근 가격을 올바르게 가져오는지 테스트"""
        # load_stock_data가 반환할 모의 DataFrame 생성
        dates = pd.to_datetime(['2022-01-03', '2022-01-04', '2022-01-06'])
        mock_df = pd.DataFrame({'close_price': [100, 102, 105]}, index=dates)

        with patch.object(self.data_handler, 'load_stock_data', return_value=mock_df):
            # 거래가 있는 날짜의 가격 조회
            price = self.data_handler.get_latest_price(date(2022, 1, 4), '005930', '2022-01-01', '2022-01-31')
            self.assertEqual(price, 102)

            # 거래가 없는 날짜의 가격 조회 (이전 거래일 가격을 가져와야 함)
            price_asof = self.data_handler.get_latest_price(date(2022, 1, 5), '005930', '2022-01-01', '2022-01-31')
            self.assertEqual(price_asof, 102)

    def test_get_latest_price_no_data(self):
        """데이터가 없을 때 None을 반환하는지 테스트"""
        with patch.object(self.data_handler, 'load_stock_data', return_value=pd.DataFrame()):
            price = self.data_handler.get_latest_price(date(2022, 1, 4), '000000', '2022-01-01', '2022-01-31')
            self.assertIsNone(price)

    @patch('pandas.read_sql')
    def test_get_filtered_stock_codes(self, mock_read_sql):
        """특정 날짜의 필터링된 종목 코드를 올바르게 가져오는지 테스트"""
        mock_read_sql.return_value = pd.DataFrame({'stock_code': ['005930', '000660']})
        
        target_date = date(2022, 1, 10)
        codes = self.data_handler.get_filtered_stock_codes(target_date)
        
        self.assertEqual(codes, ['005930', '000660'])
        
        date_str = target_date.strftime('%Y-%m-%d')
        # read_sql이 올바른 쿼리와 파라미터로 호출되었는지 검증
        mock_read_sql.assert_called_once()
        args, kwargs = mock_read_sql.call_args
        self.assertIn("SELECT stock_code FROM WeeklyFilteredStocks", args[0])
        self.assertEqual(kwargs['params'], [date_str])

    def test_default_universe_mode_is_optimistic_survivor(self):
        self.assertEqual(self.data_handler.universe_mode, "optimistic_survivor")

    @patch('pandas.read_sql')
    def test_get_pit_universe_codes_survivor_mode_filters_eventual_delisted(self, mock_read_sql):
        survivor_handler = DataHandler(self.db_config, universe_mode="optimistic_survivor")
        survivor_handler.clear_load_stock_data_cache()
        mock_read_sql.side_effect = [
            pd.DataFrame({"stock_code": ["000001", "000002", "000003"]}),
            pd.DataFrame({"stock_code": ["000002"]}),
        ]

        codes, source = survivor_handler.get_pit_universe_codes_as_of("2024-01-02")

        self.assertEqual(codes, ["000001", "000003"])
        self.assertEqual(source, "SNAPSHOT_ASOF_SURVIVOR_ONLY")

if __name__ == '__main__':
    unittest.main()
