"""
설치된 pykrx 1.2.4의 실제 어댑터 4종을 KrxSession 주입 경로로 통과시키는 시험.
외부 호출 0회(가짜 HTTP). pykrx가 필요하므로 CPU CI 묶음에는 넣지 않는다.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common import krx_session as ks
from src.pipeline import krx_daily_bulk as bulk
from src.pipeline import krx_daily_bulk_normalize as norm


class JsonResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.text = "{}"

    def json(self):
        return self._payload


class FakeKrxHttp:
    """bld별로 KRX JSON 모양의 응답을 돌려준다."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if "MDCCOMS001D1" in url:
            return JsonResponse({"_error_code": "CD001"})
        if "COMS" in url:
            return JsonResponse({})
        return JsonResponse(self._payload(kwargs.get("data") or {}))

    @staticmethod
    def _payload(data):
        bld = data.get("bld", "")
        if bld.endswith("MDCSTAT01501"):
            rows = [{"ISU_SRT_CD": "005930", "MKT_ID": "STK", "TDD_OPNPRC": "70,000", "TDD_HGPRC": "71,000",
                     "TDD_LWPRC": "69,000", "TDD_CLSPRC": "70,500", "CMPPREVDD_PRC": "500", "ACC_TRDVOL": "1,000",
                     "MKTCAP": "7,050,000", "LIST_SHRS": "100"},
                    {"ISU_SRT_CD": "035720", "MKT_ID": "KSQ", "TDD_OPNPRC": "0", "TDD_HGPRC": "0", "TDD_LWPRC": "0",
                     "TDD_CLSPRC": "1,000", "CMPPREVDD_PRC": "0", "ACC_TRDVOL": "0", "MKTCAP": "1,000,000", "LIST_SHRS": "1,000"}]
            return {"OutBlock_1": rows}
        if bld.endswith("MDCSTAT03501"):
            return {"output": [{"ISU_SRT_CD": "005930", "BPS": "50,000", "PER": "10.5", "PBR": "1.4", "EPS": "5,000", "DVD_YLD": "2.0", "DPS": "1,400"}]}
        if bld.endswith("MDCSTAT02401"):
            code = "005930" if data.get("mktId") == "STK" else "035720"
            return {"output": [{"ISU_SRT_CD": code, "ISU_NM": "x", "ASK_TRDVOL": "1", "BID_TRDVOL": "1", "NETBID_TRDVOL": "0",
                                "ASK_TRDVAL": "1", "BID_TRDVAL": "1", "NETBID_TRDVAL": "100"}]}
        if bld.endswith("MDCSTAT30501"):
            code = "005930" if str(data.get("mktTpCd")) == "1" else "035720"
            return {"OutBlock_1": [{"ISU_CD": code, "ISU_ABBRV": "x", "BAL_QTY": "10", "LIST_SHRS": "100", "BAL_AMT": "1,234,000", "MKTCAP": "1", "BAL_RTO": "0.1"}]}
        return {"OutBlock_1": []}


class TestPykrxAdapters(unittest.TestCase):
    def test_one_trading_day_is_three_logins_plus_twelve_data_calls(self):
        from pykrx.website.comm import webio

        http = FakeKrxHttp()
        policy = ks.KrxThrottlePolicy(base_delay_seconds=0, jitter_max_seconds=0, macro_pause_every=0)
        with tempfile.TemporaryDirectory() as tmp:
            session = ks.KrxSession(ks.KrxCredentials("a", "b"), policy, http, run_lock=ks.KrxRunLock(str(Path(tmp) / "l")))
            module = types.SimpleNamespace(requests="original")
            original = webio.Post.read
            session.login()
            with session.installed([module]):
                source = bulk.PykrxDailyBulkSource()
                snapshot = norm.normalize_market_day(source.market_prices("20260209"), "20260209")
                fundamentals, _ = norm.normalize_fundamental_day(source.fundamentals("20260209"), "20260209", snapshot.codes)
                investor, _ = norm.normalize_investor_day(bulk._fetch_investor_series(source, "20260209"), "20260209", snapshot.codes)
                short, _ = norm.normalize_short_balance_day(bulk._fetch_short_frames(source, "20260209"), "20260209", snapshot.codes)
                self.assertIsNot(module.requests, "original")
            session.close()
        self.assertIs(webio.Post.read, original)
        self.assertEqual(module.requests, "original")
        self.assertEqual(session.stats()["krx_calls"], 15)
        self.assertTrue(all(c[2].get("allow_redirects") is False for c in http.calls))
        self.assertEqual(sorted(snapshot.codes), ["005930", "035720"])
        self.assertEqual(int(fundamentals.iloc[0]["eps"]), 5000)
        self.assertEqual(sorted(investor["stock_code"]), ["005930", "035720"])
        self.assertEqual(int(short.set_index("stock_code").loc["005930", "short_balance_value"]), 1234000)


if __name__ == "__main__":
    unittest.main()
