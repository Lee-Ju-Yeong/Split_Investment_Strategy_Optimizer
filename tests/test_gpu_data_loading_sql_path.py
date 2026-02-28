import types
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.optimization.gpu import data_loading


class _FakeSeries:
    def __init__(self, dtype="object"):
        self.dtype = dtype


class _FakeGdf:
    def __init__(self, date_dtype="object"):
        self.columns = ["date"]
        self._cols = {"date": _FakeSeries(dtype=date_dtype)}

    def __getitem__(self, key):
        return self._cols[key]

    def __setitem__(self, key, value):
        self._cols[key] = value


class TestGpuDataLoadingSqlPath(unittest.TestCase):
    def test_read_sql_to_cudf_prefers_fast_path(self):
        fake_gdf = _FakeGdf(date_dtype="datetime64[ns]")
        fake_cudf = types.SimpleNamespace(
            read_sql_query=MagicMock(return_value=fake_gdf),
            from_pandas=MagicMock(),
            to_datetime=MagicMock(),
        )

        with patch(
            "src.optimization.gpu.data_loading._ensure_gpu_deps",
            return_value=(None, fake_cudf, None, None),
        ), patch("src.optimization.gpu.data_loading._ensure_core_deps") as mock_core_deps:
            result = data_loading._read_sql_to_cudf("SELECT 1", object(), parse_dates=["date"])

        self.assertIs(result, fake_gdf)
        fake_cudf.read_sql_query.assert_called_once()
        fake_cudf.from_pandas.assert_not_called()
        mock_core_deps.assert_not_called()

    def test_read_sql_to_cudf_converts_parse_dates_on_fast_path(self):
        fake_gdf = _FakeGdf(date_dtype="object")
        fake_cudf = types.SimpleNamespace(
            read_sql_query=MagicMock(return_value=fake_gdf),
            from_pandas=MagicMock(),
            to_datetime=MagicMock(return_value="converted-date-column"),
        )

        with patch(
            "src.optimization.gpu.data_loading._ensure_gpu_deps",
            return_value=(None, fake_cudf, None, None),
        ):
            _ = data_loading._read_sql_to_cudf("SELECT 1", object(), parse_dates=["date"])

        fake_cudf.to_datetime.assert_called_once()
        self.assertEqual(fake_gdf._cols["date"], "converted-date-column")

    def test_read_sql_to_cudf_falls_back_to_pandas(self):
        fake_df = object()
        fake_cudf = types.SimpleNamespace(from_pandas=MagicMock(return_value="gdf-from-pandas"))
        fake_pd = types.SimpleNamespace(read_sql=MagicMock(return_value=fake_df))

        with patch(
            "src.optimization.gpu.data_loading._ensure_gpu_deps",
            return_value=(None, fake_cudf, None, None),
        ), patch(
            "src.optimization.gpu.data_loading._ensure_core_deps",
            return_value=(None, fake_pd),
        ):
            result = data_loading._read_sql_to_cudf("SELECT 1", object(), parse_dates=["date"])

        self.assertEqual(result, "gdf-from-pandas")
        fake_pd.read_sql.assert_called_once()
        fake_cudf.from_pandas.assert_called_once_with(fake_df)

    def test_read_sql_to_cudf_fast_path_error_fails_fast(self):
        fake_cudf = types.SimpleNamespace(
            read_sql_query=MagicMock(side_effect=ValueError("fast-path failed")),
            from_pandas=MagicMock(),
            to_datetime=MagicMock(),
        )

        with patch(
            "src.optimization.gpu.data_loading._ensure_gpu_deps",
            return_value=(None, fake_cudf, None, None),
        ), patch("src.optimization.gpu.data_loading._ensure_core_deps") as mock_core_deps:
            with self.assertRaises(ValueError):
                data_loading._read_sql_to_cudf("SELECT 1", object(), parse_dates=["date"])

        fake_cudf.from_pandas.assert_not_called()
        mock_core_deps.assert_not_called()


class TestGpuDataTypeNormalization(unittest.TestCase):
    class _FakeSeries:
        def __init__(self, dtype):
            self.dtype = dtype
            self.last_cast = None

        def astype(self, dtype):
            self.last_cast = dtype
            self.dtype = dtype
            return self

    class _FakeGdf:
        def __init__(self):
            self._cols = {
                "open_price": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "high_price": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "low_price": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "close_price": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "atr_14_ratio": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "cheap_score": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "cheap_score_confidence": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "flow5_mcap": TestGpuDataTypeNormalization._FakeSeries("float64"),
                "market_cap": TestGpuDataTypeNormalization._FakeSeries("int64"),
            }
            self.columns = list(self._cols.keys())

        def __getitem__(self, key):
            return self._cols[key]

        def __setitem__(self, key, value):
            self._cols[key] = value

    def test_normalize_loaded_types_cast_policy(self):
        gdf = self._FakeGdf()
        normalized = data_loading._normalize_loaded_types(gdf)

        for col in data_loading._FLOAT32_COLUMNS:
            self.assertEqual(normalized[col].dtype, "float32")
        self.assertEqual(normalized["market_cap"].dtype, "float64")


class _FakeSetIndexGdf(_FakeGdf):
    @property
    def shape(self):
        return (0, 0)

    def set_index(self, _columns):
        return self


class TestGpuDataLoadingQuery(unittest.TestCase):
    def test_preload_query_omits_unused_volume_field(self):
        captured_queries = []

        def fake_read_sql_to_cudf(query, _engine, parse_dates=None):
            captured_queries.append(query)
            gdf = _FakeSetIndexGdf()
            gdf._cols["ticker"] = _FakeSeries(dtype="str")
            gdf._cols["date"] = _FakeSeries(dtype="datetime64[ns]")
            return gdf

        with patch(
            "src.optimization.gpu.data_loading._get_sql_engine",
            return_value="engine",
        ), patch(
            "src.optimization.gpu.data_loading._read_sql_to_cudf",
            side_effect=fake_read_sql_to_cudf,
        ), patch("src.optimization.gpu.data_loading._ensure_gpu_deps"):
            data_loading.preload_all_data_to_gpu(
                engine="mysql://dummy",
                start_date="20140101",
                end_date="20140131",
                use_adjusted_prices=False,
            )

        self.assertEqual(len(captured_queries), 1)
        query = captured_queries[0]
        self.assertNotIn("CAST(dsp.volume AS SIGNED)", query)
        self.assertIn("mcd.market_cap AS market_cap", query)

    def test_preload_query_uses_stored_adj_ohlc_with_stale_guard(self):
        captured_queries = []

        def fake_read_sql_to_cudf(query, _engine, parse_dates=None):
            captured_queries.append(query)
            gdf = _FakeSetIndexGdf()
            gdf._cols.update(
                {
                    "ticker": _FakeSeries(dtype="str"),
                    "date": _FakeSeries(dtype="datetime64[ns]"),
                    "open_price": _FakeSeries(dtype="float32"),
                    "high_price": _FakeSeries(dtype="float32"),
                    "low_price": _FakeSeries(dtype="float32"),
                    "close_price": _FakeSeries(dtype="float32"),
                }
            )
            gdf.columns = list(gdf._cols.keys())
            return gdf

        with patch(
            "src.optimization.gpu.data_loading._get_sql_engine",
            return_value="engine",
        ), patch(
            "src.optimization.gpu.data_loading._read_sql_to_cudf",
            side_effect=fake_read_sql_to_cudf,
        ), patch(
            "src.optimization.gpu.data_loading._ensure_gpu_deps"
        ), patch(
            "src.optimization.gpu.data_loading._has_stored_adj_ohlc_columns",
            return_value=True,
        ), patch(
            "src.optimization.gpu.data_loading._validate_adjusted_ohlc_not_null"
        ):
            data_loading.preload_all_data_to_gpu(
                engine="mysql://dummy",
                start_date="20140101",
                end_date="20140131",
                use_adjusted_prices=True,
            )

        self.assertEqual(len(captured_queries), 1)
        query = captured_queries[0]
        self.assertIn("CASE", query)
        self.assertIn("ABS(dsp.adj_open - (dsp.open_price * dsp.adj_ratio)) > 1e-5", query)
        self.assertIn("ABS(dsp.adj_high - (dsp.high_price * dsp.adj_ratio)) > 1e-5", query)
        self.assertIn("ABS(dsp.adj_low - (dsp.low_price * dsp.adj_ratio)) > 1e-5", query)
        self.assertIn("dsp.adj_open", query)
        self.assertIn("dsp.adj_high", query)
        self.assertIn("dsp.adj_low", query)


class TestGpuAdjustedValidation(unittest.TestCase):
    def test_validate_adjusted_ohlc_not_null_raises(self):
        gdf = pd.DataFrame(
            {
                "open_price": [1.0, None],
                "high_price": [1.0, 2.0],
                "low_price": [1.0, 2.0],
                "close_price": [1.0, 2.0],
            }
        )

        with self.assertRaises(ValueError):
            data_loading._validate_adjusted_ohlc_not_null(gdf, "2013-11-20")

    def test_validate_adjusted_ohlc_not_null_reports_first_ticker_date(self):
        idx = pd.MultiIndex.from_tuples(
            [("000001", pd.Timestamp("2014-01-02")), ("000002", pd.Timestamp("2014-01-03"))],
            names=["ticker", "date"],
        )
        gdf = pd.DataFrame(
            {
                "open_price": [None, 2.0],
                "high_price": [1.0, 2.0],
                "low_price": [1.0, 2.0],
                "close_price": [1.0, 2.0],
            },
            index=idx,
        )

        with self.assertRaisesRegex(ValueError, "ticker=000001, date=2014-01-02"):
            data_loading._validate_adjusted_ohlc_not_null(gdf, "2013-11-20")


if __name__ == "__main__":
    unittest.main()
