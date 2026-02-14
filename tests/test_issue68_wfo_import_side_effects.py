import importlib
import sys
import unittest


class TestIssue68WfoImportSideEffects(unittest.TestCase):
    def test_walk_forward_analyzer_import_works_without_gpu_deps(self):
        # Import should succeed even if GPU deps (cupy/cudf) are not installed.
        sys.modules.pop("src.walk_forward_analyzer", None)
        importlib.import_module("src.walk_forward_analyzer")

