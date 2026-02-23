import types
import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
