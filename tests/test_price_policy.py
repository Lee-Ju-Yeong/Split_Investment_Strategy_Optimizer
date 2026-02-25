import unittest

from src.price_policy import (
    is_adjusted_price_basis,
    resolve_price_policy,
    validate_backtest_window_for_price_policy,
)


class TestPricePolicy(unittest.TestCase):
    def test_resolve_price_policy_defaults_to_adjusted_mode(self):
        basis, gate = resolve_price_policy({})
        self.assertEqual(basis, "adjusted")
        self.assertEqual(gate, "2013-11-20")

    def test_resolve_price_policy_accepts_raw_basis(self):
        basis, gate = resolve_price_policy(
            {"price_basis": "raw", "adjusted_price_gate_start_date": "20131120"}
        )
        self.assertEqual(basis, "raw")
        self.assertEqual(gate, "2013-11-20")

    def test_resolve_price_policy_backward_compat_bool(self):
        basis, gate = resolve_price_policy({"use_adjusted_price_for_backtest": False})
        self.assertEqual(basis, "raw")
        self.assertEqual(gate, "2013-11-20")
        self.assertFalse(is_adjusted_price_basis(basis))

    def test_validate_adjusted_window_rejects_pre_gate_start(self):
        with self.assertRaises(ValueError):
            validate_backtest_window_for_price_policy(
                start_date="2013-01-01",
                end_date="2013-12-31",
                price_basis="adjusted",
                adjusted_price_gate_start_date="2013-11-20",
            )

    def test_validate_raw_window_allows_pre_gate_start(self):
        validate_backtest_window_for_price_policy(
            start_date="2001-01-01",
            end_date="2001-12-31",
            price_basis="raw",
            adjusted_price_gate_start_date="2013-11-20",
        )


if __name__ == "__main__":
    unittest.main()
