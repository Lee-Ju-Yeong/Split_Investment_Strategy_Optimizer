"""US 단독 파일 연결을 별도 프로세스로 검사해 모듈별 CI 시간 여유를 확보한다."""
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
import tempfile
import unittest

from examples.etf_m5_demo import fixture
from src.research.etf.m5_pilot import run_pilot


class USOnlyPilotTests(unittest.TestCase):
    def test_us_only_training_includes_first_preopen_valuation(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        data = root / "us-only"
        manifest = fixture(data, us_only=True)
        output = root / "us-run"
        summary = run_pilot(data, manifest, ["SYNTHETIC_US", "SYNTHETIC_US_2"], output)
        self.assertEqual((summary["status"], summary["succeeded"]), ("passed", 4), summary)
        trial = json.loads((output / "candidate-01-O-H-L-C.json").read_text())
        valuations = trial["training_ledger"]["valuations"]
        first_trade = trial["training_ledger"]["trades"][0]
        self.assertLess(valuations[0]["at"], first_trade["at"])
        self.assertEqual(sum(map(D, valuations[0]["holdings_krw"].values())), D(0))
        with localcontext(prec=80):
            exposure = sum((sum(map(D, row["holdings_krw"].values())) / D(row["equity_krw"])
                            for row in valuations), D(0)) / len(valuations)
        self.assertEqual(D(trial["frozen_policy"]["investment_fraction"]), exposure)


if __name__ == "__main__":
    unittest.main()
