import contextlib
import importlib
import io
import sys
import unittest


class TestIssue60ImportSideEffects(unittest.TestCase):
    def test_parameter_simulation_gpu_import_is_silent(self):
        # Ensure we capture import-time stdout even if the module was imported before.
        sys.modules.pop("src.parameter_simulation_gpu", None)
        sys.modules.pop("src.parameter_simulation_gpu_lib", None)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mod = importlib.import_module("src.parameter_simulation_gpu")

        self.assertEqual(buf.getvalue(), "")
        self.assertTrue(hasattr(mod, "find_optimal_parameters"))

