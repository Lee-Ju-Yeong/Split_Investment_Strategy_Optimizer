import unittest

from src.universe_policy import (
    DEFAULT_UNIVERSE_MODE,
    is_survivor_optimistic_mode,
    normalize_universe_mode,
    resolve_universe_mode,
)


class TestUniversePolicy(unittest.TestCase):
    def test_default_mode(self):
        self.assertEqual(resolve_universe_mode({}), DEFAULT_UNIVERSE_MODE)
        self.assertEqual(resolve_universe_mode({}, universe_mode=""), DEFAULT_UNIVERSE_MODE)
        self.assertEqual(resolve_universe_mode({"universe_mode": ""}), DEFAULT_UNIVERSE_MODE)

    def test_alias_normalization(self):
        self.assertEqual(normalize_universe_mode("pit_no_forced_tier_liquidation"), "strict_pit")
        self.assertEqual(normalize_universe_mode("survivor_only"), "optimistic_survivor")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            normalize_universe_mode("invalid")

    def test_is_survivor_mode(self):
        self.assertTrue(is_survivor_optimistic_mode("optimistic_survivor"))
        self.assertFalse(is_survivor_optimistic_mode("strict_pit"))


if __name__ == "__main__":
    unittest.main()
