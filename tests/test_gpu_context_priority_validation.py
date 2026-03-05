import unittest

from src.optimization.gpu.context import _normalize_priority_option


class TestGpuContextPriorityValidation(unittest.TestCase):
    def test_normalize_priority_option_accepts_supported_values(self):
        self.assertEqual(_normalize_priority_option(0), 0)
        self.assertEqual(_normalize_priority_option(1), 1)
        self.assertEqual(_normalize_priority_option("0"), 0)
        self.assertEqual(_normalize_priority_option("1"), 1)
        self.assertEqual(_normalize_priority_option("lowest_order"), 0)
        self.assertEqual(_normalize_priority_option("highest_drop"), 1)
        self.assertEqual(_normalize_priority_option("biggest_drop"), 1)

    def test_normalize_priority_option_rejects_unsupported_values(self):
        with self.assertRaisesRegex(ValueError, "Unsupported additional_buy_priority"):
            _normalize_priority_option(2)
        with self.assertRaisesRegex(ValueError, "Unsupported additional_buy_priority"):
            _normalize_priority_option("momentum")
        with self.assertRaisesRegex(ValueError, "Unsupported additional_buy_priority"):
            _normalize_priority_option(0.5)
        with self.assertRaisesRegex(ValueError, "Unsupported additional_buy_priority"):
            _normalize_priority_option(1.2)
        with self.assertRaisesRegex(ValueError, "Unsupported additional_buy_priority"):
            _normalize_priority_option(-0.1)


if __name__ == "__main__":
    unittest.main()
