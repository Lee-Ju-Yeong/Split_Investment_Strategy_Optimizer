import unittest
from unittest.mock import MagicMock, patch

from src.optimization.gpu import data_loading


class TestGpuDataLoadingEngineCache(unittest.TestCase):
    def setUp(self):
        data_loading._get_sql_engine.cache_clear()

    def tearDown(self):
        data_loading._get_sql_engine.cache_clear()

    def test_get_sql_engine_reuses_same_url(self):
        fake_engine = object()
        fake_create_engine = MagicMock(return_value=fake_engine)

        with patch(
            "src.optimization.gpu.data_loading._ensure_gpu_deps",
            return_value=(None, None, fake_create_engine, None),
        ):
            engine_first = data_loading._get_sql_engine("mysql+pymysql://user:pw@host/db")
            engine_second = data_loading._get_sql_engine("mysql+pymysql://user:pw@host/db")

        self.assertIs(engine_first, engine_second)
        fake_create_engine.assert_called_once_with("mysql+pymysql://user:pw@host/db")

    def test_get_sql_engine_creates_per_distinct_url(self):
        fake_engine1 = object()
        fake_engine2 = object()
        fake_create_engine = MagicMock(side_effect=[fake_engine1, fake_engine2])

        with patch(
            "src.optimization.gpu.data_loading._ensure_gpu_deps",
            return_value=(None, None, fake_create_engine, None),
        ):
            engine_first = data_loading._get_sql_engine("mysql+pymysql://user:pw@host/db1")
            engine_second = data_loading._get_sql_engine("mysql+pymysql://user:pw@host/db2")

        self.assertIs(engine_first, fake_engine1)
        self.assertIs(engine_second, fake_engine2)
        self.assertEqual(fake_create_engine.call_count, 2)


if __name__ == "__main__":
    unittest.main()
