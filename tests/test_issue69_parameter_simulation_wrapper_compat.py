import importlib
import sys
import unittest
from pathlib import Path


class TestIssue69ParameterSimulationWrapperCompat(unittest.TestCase):
    def test_parameter_simulation_lib_wrapper_import_in_package_and_legacy_modes(self):
        """
        Issue #69 (PR-9):
        - `src.parameter_simulation_gpu_lib` should stay as a compat wrapper
          after moving implementation to `src.optimization.gpu.*`.
        """

        lib_pkg = importlib.import_module("src.parameter_simulation_gpu_lib")
        self.assertTrue(hasattr(lib_pkg, "find_optimal_parameters"))
        self.assertTrue(hasattr(lib_pkg, "main"))
        self.assertTrue(hasattr(lib_pkg, "run_gpu_optimization"))
        self.assertTrue(hasattr(lib_pkg, "analyze_and_save_results"))

        impl_pkg = importlib.import_module("src.optimization.gpu.parameter_simulation")
        self.assertTrue(hasattr(impl_pkg, "find_optimal_parameters"))
        self.assertTrue(hasattr(impl_pkg, "main"))

        repo_root = Path(__file__).resolve().parent.parent
        src_dir = repo_root / "src"
        sys.path.insert(0, str(src_dir))
        try:
            lib_legacy = importlib.import_module("parameter_simulation_gpu_lib")
            self.assertTrue(hasattr(lib_legacy, "find_optimal_parameters"))
            self.assertTrue(hasattr(lib_legacy, "main"))
        finally:
            try:
                sys.path.remove(str(src_dir))
            except ValueError:
                pass

            for name in [
                "parameter_simulation_gpu_lib",
                "optimization",
                "optimization.gpu",
                "optimization.gpu.context",
                "optimization.gpu.data_loading",
                "optimization.gpu.kernel",
                "optimization.gpu.analysis",
                "optimization.gpu.parameter_simulation",
            ]:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
