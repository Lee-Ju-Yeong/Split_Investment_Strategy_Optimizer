import os
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.optimization.gpu import kernel


class TestGpuKernelBatchSize(unittest.TestCase):
    @patch("src.optimization.gpu.kernel._query_free_memory_with_nvidia_smi", return_value=1024)
    @patch("src.optimization.gpu.kernel._query_free_memory_with_cupy_runtime", return_value=2048)
    def test_resolve_free_gpu_memory_prefers_nvidia_smi(self, _mock_runtime, _mock_smi):
        free_bytes, source = kernel._resolve_free_gpu_memory_bytes()
        self.assertEqual(free_bytes, 1024)
        self.assertEqual(source, "nvidia-smi")

    @patch("src.optimization.gpu.kernel._query_free_memory_with_nvidia_smi", return_value=None)
    @patch("src.optimization.gpu.kernel._query_free_memory_with_cupy_runtime", return_value=2048)
    def test_resolve_free_gpu_memory_falls_back_to_runtime(self, _mock_runtime, _mock_smi):
        free_bytes, source = kernel._resolve_free_gpu_memory_bytes()
        self.assertEqual(free_bytes, 2048)
        self.assertEqual(source, "cupy.runtime.memGetInfo")

    @patch("src.optimization.gpu.kernel._query_free_memory_with_nvidia_smi", return_value=None)
    @patch("src.optimization.gpu.kernel._query_free_memory_with_cupy_runtime", return_value=None)
    def test_resolve_free_gpu_memory_returns_unavailable_when_both_fail(self, _mock_runtime, _mock_smi):
        free_bytes, source = kernel._resolve_free_gpu_memory_bytes()
        self.assertIsNone(free_bytes)
        self.assertEqual(source, "unavailable")


if __name__ == "__main__":
    unittest.main()
