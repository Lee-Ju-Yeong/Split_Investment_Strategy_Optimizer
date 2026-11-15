"""ETF 분류 메타데이터 규칙 테스트 (src/research/etf/metadata.py).

대표 상품의 상품명·네이버 탭·운용사 분류만으로 기대값을 확인한다. 외부 자료·DB·네트워크를 쓰지 않는다.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.research.etf import metadata as md  # noqa: E402


def kr(symbol, name, tab, **hints):
    return {"market": "KR", "symbol": symbol, "name": name, "raw_category_code": tab, "issuer": "",
            "issuer_category": "", "underlying_benchmark": "", "history_download_status": "downloaded",
            "explicit_etn_name": "0", "leveraged_name_hint": hints.get("lev", "0"),
            "inverse_name_hint": hints.get("inv", "0"), "options_strategy_name_hint": hints.get("opt", "0")}


def us(symbol, name, issuer, issuer_category="", **hints):
    return {"market": "US", "symbol": symbol, "name": name, "raw_category_code": "", "issuer": issuer,
            "issuer_category": issuer_category, "underlying_benchmark": "",
            "history_download_status": hints.get("status", "downloaded"),
            "explicit_etn_name": hints.get("etn", "0"), "leveraged_name_hint": hints.get("lev", "0"),
            "inverse_name_hint": hints.get("inv", "0"), "options_strategy_name_hint": hints.get("opt", "0")}


C = md.classify


class TestDomesticListed(unittest.TestCase):
    def test_kospi200_family_spans_plain_leverage_inverse(self):
        plain = C(kr("069500", "KODEX 200", "1"))
        lev = C(kr("122630", "KODEX 레버리지", "3", lev="1"))
        inv2 = C(kr("252670", "KODEX 200선물인버스2X", "3", lev="1", inv="1"))
        inv1 = C(kr("114800", "KODEX 인버스", "3", inv="1"))
        krx300 = C(kr("292190", "KODEX KRX300", "1"))
        msci = C(kr("310970", "TIGER MSCI Korea TR", "1"))
        for r in (plain, lev, inv2, inv1, krx300, msci):
            self.assertEqual(r["index_family"], "KR_EQ_LARGE", r["name"])
            self.assertEqual(r["asset_class"], "국내주식")
            self.assertEqual(r["kr_tax_class"], "국내주식형")
            self.assertEqual(r["region"], "한국")
        self.assertEqual(plain["leverage_multiple"], 1.0)
        self.assertTrue(plain["d1_eligible"])
        self.assertEqual(lev["leverage_multiple"], 2.0)
        self.assertTrue(lev["d1_eligible"])
        self.assertEqual(inv2["leverage_multiple"], -2.0)
        self.assertTrue(inv2["is_inverse"])
        self.assertTrue(inv2["is_futures_based"])
        self.assertFalse(inv2["d1_eligible"])
        self.assertEqual(inv2["d1_exclusion_reasons"], "inverse")
        self.assertEqual(inv1["leverage_multiple"], -1.0)
        self.assertFalse(inv1["needs_review"])

    def test_kosdaq150_and_kosdaq_global_is_domestic(self):
        self.assertEqual(C(kr("233740", "KODEX 코스닥150레버리지", "3", lev="1"))["index_family"], "KR_KOSDAQ150")
        g = C(kr("461450", "KODEX 코스닥글로벌", "1"))
        self.assertEqual(g["asset_class"], "국내주식")
        self.assertFalse(g["needs_review"])

    def test_us_sp500_listed_in_korea_shares_family_with_spy(self):
        tiger = C(kr("360750", "TIGER 미국S&P500", "4"))
        lev = C(kr("225040", "TIGER 미국S&P500레버리지(합성 H)", "4", lev="1"))
        mix = C(kr("0080X0", "SOL 미국S&P500미국채혼합50", "7"))
        spy = C(us("SPY", "State Street SPDR S&P 500 ETF Trust", "State Street SPDR"))
        for r in (tiger, lev, mix, spy):
            self.assertEqual(r["index_family"], "US_EQ_LARGE_BLEND", r["name"])
        self.assertEqual(tiger["asset_class"], "해외주식")
        self.assertEqual(tiger["region"], "미국")
        self.assertEqual(tiger["kr_tax_class"], "기타")
        self.assertEqual(lev["leverage_multiple"], 2.0)
        self.assertTrue(lev["d1_eligible"])
        self.assertEqual(mix["asset_class"], "혼합")
        self.assertEqual(spy["kr_tax_class"], "")

    def test_options_strategies_excluded(self):
        cc = C(kr("482730", "TIGER 미국S&P500타겟데일리커버드콜", "4", opt="1"))
        buf = C(kr("0026E0", "KODEX 미국S&P500버퍼3월액티브", "4"))
        hedge = C(kr("0084E0", "KIWOOM 미국대형주500월간목표헤지액티브", "4"))
        for r in (cc, buf, hedge):
            self.assertTrue(r["is_options_strategy"], r["name"])
            self.assertFalse(r["d1_eligible"])
            self.assertIn("options_strategy", r["d1_exclusion_reasons"])
            self.assertEqual(r["index_family"], "US_EQ_LARGE_BLEND")
        # 원천 힌트가 0이어도 규칙 키워드(버퍼·목표헤지)가 명시적이면 검토 표시를 하지 않는다.
        self.assertFalse(buf["needs_review"])

    def test_nasdaq100(self):
        r = C(kr("418660", "TIGER 미국나스닥100레버리지(합성)", "4", lev="1"))
        self.assertEqual(r["index_family"], "US_NASDAQ100")
        self.assertEqual(r["leverage_multiple"], 2.0)
        self.assertEqual(C(us("QQQ", "Invesco QQQ Trust, Series 1", "Invesco"))["index_family"], "US_NASDAQ100")

    def test_gold_spot_and_futures(self):
        spot = C(kr("0072R0", "TIGER KRX금현물", "5"))
        fut = C(kr("132030", "KODEX 골드선물(H)", "5"))
        self.assertEqual(spot["asset_class"], "금")
        self.assertEqual(spot["index_family"], "GOLD")
        self.assertEqual(spot["kr_tax_class"], "기타")
        self.assertFalse(spot["is_futures_based"])
        self.assertTrue(fut["is_futures_based"])
        self.assertEqual(fut["index_family"], "GOLD")
        self.assertEqual(C(us("GLD", "SPDR Gold Shares", "State Street SPDR"))["index_family"], "GOLD")

    def test_currency_cash_bonds(self):
        usd = C(kr("261240", "KODEX 미국달러선물", "7"))
        self.assertEqual((usd["asset_class"], usd["index_family"]), ("통화", "FX_USD_KRW"))
        cd = C(kr("459580", "KODEX CD금리액티브(합성)", "6"))
        self.assertEqual((cd["asset_class"], cd["index_family"]), ("단기금리", "KRW_CASH"))
        ktb = C(kr("114260", "KODEX 국고채3년", "6"))
        self.assertEqual((ktb["asset_class"], ktb["index_family"], ktb["kr_tax_class"]), ("채권", "KR_KTB_3Y", "기타"))
        long_target = C(kr("457700", "KODEX 53-09 국고채액티브", "6"))
        self.assertEqual((long_target["index_family"], long_target["index_family_basis"]), ("KR_KTB_30Y", "theme"))
        credit = C(kr("473290", "KODEX 26-12 회사채(AA-이상)액티브", "6"))
        self.assertEqual(credit["index_family"], "KR_TARGET_CREDIT_2026")

    def test_single_stock_leverage_needs_review(self):
        r = C(kr("0193W0", "KODEX 삼성전자단일종목레버리지", "2", lev="1"))
        self.assertTrue(r["is_single_stock"])
        self.assertEqual(r["leverage_multiple"], 2.0)
        self.assertEqual(r["index_family"], "STOCK_KR_SAMSUNG_ELEC")
        self.assertTrue(r["needs_review"])
        self.assertIn("단일종목", r["review_reasons"])

    def test_long_short_spread_has_unknown_leverage(self):
        r = C(kr("360140", "KODEX 200롱코스닥150숏선물", "2"))
        self.assertIsNone(r["leverage_multiple"])
        self.assertFalse(r["d1_eligible"])
        self.assertEqual(r["d1_exclusion_reasons"], "leverage_unknown")
        self.assertTrue(r["needs_review"])

    def test_domestic_theme_and_regressions(self):
        self.assertEqual(C(kr("150460", "TIGER 중국소비테마", "2"))["asset_class"], "국내주식")
        # '필수소비재'의 '수소'가 수소 테마로 잡히면 안 된다.
        self.assertEqual(C(kr("266410", "KODEX 필수소비재", "2"))["index_family"], "KR_SECTOR_CONSUMER")
        semi = {C(kr(s, n, "2"))["index_family"] for s, n in
                [("091160", "KODEX 반도체"), ("395160", "KODEX AI반도체TOP2플러스"), ("0182S0", "1Q K반도체TOP2채권혼합50")]}
        self.assertEqual(semi, {"KR_THEME_SEMICONDUCTOR"})
        reit = C(kr("329200", "TIGER 리츠부동산인프라", "2"))
        self.assertEqual((reit["asset_class"], reit["index_family"]), ("리츠", "KR_REIT"))
        self.assertTrue(reit["needs_review"])

    def test_foreign_families(self):
        self.assertEqual(C(kr("283580", "KODEX 차이나CSI300", "4"))["index_family"], "CHINA_A_LARGE")
        hst = C(kr("371160", "TIGER 차이나항셍테크", "4"))
        self.assertEqual((hst["index_family"], hst["region"]), ("CHINA_TECH", "중국"))
        self.assertEqual(C(kr("390390", "KODEX 미국반도체", "4"))["index_family"], "US_GLOBAL_SEMIS")
        self.assertEqual(C(us("SOXX", "iShares PHLX SOX Semiconductor Sector Index Fund", "iShares"))["index_family"],
                         "US_GLOBAL_SEMIS")
        self.assertEqual(C(kr("488500", "TIGER 미국S&P500동일가중", "4"))["index_family"], "US_EQ_SP500_EQUAL_WEIGHT")
        self.assertEqual(C(us("RSP", "Invesco S&P 500 Equal Weight ETF", "Invesco"))["index_family"],
                         "US_EQ_SP500_EQUAL_WEIGHT")
        self.assertEqual(C(kr("458730", "TIGER 미국배당다우존스", "4"))["index_family"], "US_DIV_DJ_US_DIVIDEND100")
        self.assertEqual(C(us("SCHD", "Schwab US Dividend Equity ETF", "Schwab"))["index_family"],
                         "US_DIV_DJ_US_DIVIDEND100")

    def test_tab_name_conflict_is_flagged(self):
        r = C(kr("0060H0", "TIGER 토탈월드스탁액티브", "1"))
        self.assertEqual(r["asset_class"], "해외주식")
        self.assertTrue(r["needs_review"])


class TestUsListed(unittest.TestCase):
    def test_proshares_multiples(self):
        cases = {
            ("SSO", "ProShares Ultra S&P500"): (2.0, False, True),
            ("UPRO", "ProShares UltraPro S&P 500"): (3.0, False, False),
            ("SH", "ProShares Short S&P500"): (-1.0, True, False),
            ("SDS", "ProShares UltraShort S&P500"): (-2.0, True, False),
            ("SPXU", "ProShares UltraPro Short S&P500"): (-3.0, True, False),
        }
        for (sym, name), (lev, inv, eligible) in cases.items():
            r = C(us(sym, name, "ProShares"))
            self.assertEqual(r["leverage_multiple"], lev, name)
            self.assertEqual(r["is_inverse"], inv, name)
            self.assertEqual(r["d1_eligible"], eligible, name)
            self.assertEqual(r["index_family"], "US_EQ_LARGE_BLEND", name)
        self.assertEqual(C(us("UPRO", "ProShares UltraPro S&P 500", "ProShares"))["d1_exclusion_reasons"],
                         "leverage_3x_or_more")
        sdow = C(us("SDOW", "UltraPro Short Dow30", "ProShares"))
        self.assertEqual((sdow["leverage_multiple"], sdow["index_family"]), (-3.0, "US_DOW30"))
        self.assertEqual(C(us("TQQQ", "ProShares UltraPro QQQ", "ProShares"))["leverage_multiple"], 3.0)

    def test_broad_us_indices_merge(self):
        for sym, name, issuer in [("VOO", "Vanguard S&P 500 ETF", "Vanguard"),
                                  ("IVV", "iShares Core S&P 500 ETF", "iShares"),
                                  ("VTI", "Vanguard Morningstar Total Stock Market ETF", "Vanguard"),
                                  ("IWB", "iShares Russell 1000 ETF", "iShares")]:
            self.assertEqual(C(us(sym, name, issuer))["index_family"], "US_EQ_LARGE_BLEND", name)
        self.assertEqual(C(us("IWF", "iShares Russell 1000 Growth Fund", "iShares"))["index_family"], "US_EQ_LARGE_GROWTH")
        self.assertEqual(C(us("IWM", "iShares Russell 2000 Index Fund", "iShares"))["index_family"], "US_EQ_SMALL_BLEND")
        self.assertEqual(C(us("SPLV", "Invesco S&P 500 Low Volatility ETF", "Invesco"))["index_family_basis"], "name")

    def test_korea_etf_listed_in_us(self):
        r = C(us("EWY", "iShares MSCI South Korea ETF", "iShares"))
        self.assertEqual((r["asset_class"], r["region"], r["index_family"]), ("국내주식", "한국", "KR_EQ_LARGE"))

    def test_gold_and_treasury_leverage(self):
        ugl = C(us("UGL", "ProShares Ultra Gold", "ProShares"))
        self.assertEqual((ugl["leverage_multiple"], ugl["index_family"], ugl["is_futures_based"]), (2.0, "GOLD", True))
        tlt = C(us("TLT", "iShares 20+ Year Treasury Bond ETF", "iShares"))
        self.assertEqual((tlt["asset_class"], tlt["index_family"]), ("채권", "US_TREASURY_LONG"))
        tbt = C(us("TBT", "ProShares UltraShort Lehman 20  Year Treasury", "ProShares"))
        self.assertEqual((tbt["leverage_multiple"], tbt["index_family"]), (-2.0, "US_TREASURY_LONG"))
        self.assertEqual(C(kr("476760", "ACE 미국30년국채액티브", "6"))["index_family"], "US_TREASURY_LONG")

    def test_cash_like(self):
        self.assertEqual(C(us("BIL", "State Street SPDR Bloomberg 1-3 Month T-Bill ETF", "State Street SPDR"))["index_family"],
                         "USD_CASH")
        tbll = C(us("TBLL", "Invesco Short Term Treasury ETF", "Invesco", "Fixed Income / Treasury / Ultrashort Bond"))
        self.assertEqual((tbll["asset_class"], tbll["index_family"]), ("단기금리", "USD_CASH"))
        gsy = C(us("GSY", "Invesco Ultra Short Duration ETF", "Invesco", lev="1"))
        self.assertEqual(gsy["leverage_multiple"], 1.0)
        self.assertFalse(gsy["is_inverse"])
        self.assertFalse(gsy["needs_review"])
        sofr = C(kr("455030", "KODEX 미국달러SOFR금리액티브(합성)", "6"))
        self.assertEqual((sofr["asset_class"], sofr["index_family"]), ("단기금리", "USD_CASH"))

    def test_vix_leverage_unknown(self):
        r = C(us("UVXY", "ProShares Ultra VIX Short Term Futures ETF", "ProShares", lev="1"))
        self.assertIsNone(r["leverage_multiple"])
        self.assertFalse(r["d1_eligible"])
        self.assertTrue(r["needs_review"])
        self.assertEqual(r["asset_class"], "변동성")
        vixy = C(us("VIXY", "VIX Short-Term Futures ETF", "ProShares"))
        self.assertEqual(vixy["leverage_multiple"], 1.0)
        self.assertTrue(vixy["is_futures_based"])
        self.assertTrue(vixy["needs_review"])

    def test_options_inverse_currency(self):
        self.assertTrue(C(us("PBP", "Invesco S&P 500 BuyWrite ETF", "Invesco"))["is_options_strategy"])
        udn = C(us("UDN", "Invesco DB USD Index Bearish ETF", "Invesco"))
        self.assertTrue(udn["is_inverse"])
        self.assertTrue(udn["is_futures_based"])
        self.assertEqual((udn["asset_class"], udn["index_family"]), ("통화", "FX_USD_INDEX"))
        self.assertTrue(C(us("EMTY", "ProShares Decline of the Retail Store ETF", "ProShares"))["is_inverse"])

    def test_us_regressions(self):
        # PROPERTY(보험 이름)·MUNI(COMMUNICATION 부분 문자열)·5-10년 회사채 오분류 방지
        self.assertEqual(C(us("KBWP", "Invesco KBW Property & Casualty Insurance ETF", "Invesco"))["asset_class"], "해외주식")
        self.assertEqual(C(us("PSCU", "Invesco S&P SmallCap Utilities & Communication Services ETF", "Invesco"))["asset_class"],
                         "해외주식")
        self.assertEqual(C(us("PEY", "Invesco High Yield Equity Dividend Achievers ETF", "Invesco"))["asset_class"], "해외주식")
        self.assertEqual(C(us("IGIB", "iShares 5-10 Year Investment Grade Corporate Bond ETF", "iShares"))["index_family"],
                         "US_CORP_IG")
        self.assertEqual(C(us("IGLB", "iShares 10  Year Investment Grade Corporate Bond ETF", "iShares"))["index_family"],
                         "US_CORP_IG_LONG")
        self.assertEqual(C(us("BSJQ", "Invesco BulletShares 2026 High Yield Corporate Bond ETF", "Invesco"))["index_family"],
                         "US_TARGET_HY_2026")
        self.assertEqual(C(us("IBDR", "iShares iBonds Dec 2026 Term Corporate ETF", "iShares"))["index_family"],
                         "US_TARGET_CORP_2026")
        self.assertEqual(C(us("AAXJ", "iShares MSCI All Country Asia ex Japan ETF", "iShares"))["region"], "아시아태평양")
        self.assertEqual(C(us("IGHG", "ProShares Investment Grade-Interest Rate Hedged", "ProShares"))["asset_class"], "채권")
        rdiv = C(us("RDIV", "Invesco S&P Ultra Dividend Revenue ETF", "Invesco", lev="1"))
        self.assertEqual(rdiv["leverage_multiple"], 1.0)
        self.assertFalse(rdiv["needs_review"])

    def test_buffer_variants_are_options_and_excluded(self):
        # Codex 검토 회귀: \bBUFFER\b가 Buffer10·Buffer100·Buffered·DynaBuffer를 놓쳐 D1을 통과했다.
        for sym, name in [("APRT", "AllianzIM U.S. Large Cap Buffer10 Apr ETF"),
                          ("AIOO", "AllianzIM U.S. Equity Buffer100 Protection ETF"),
                          ("ACJL", "Buffered Bulldog ETF by C.J. Lawrence, a division of Apollon"),
                          ("MPDB", "m+ DynaBuffer ETF"),
                          ("SIXD", "AllianzIM U.S. Large Cap 6 Month Buffer10 Jun/Dec ETF")]:
            r = C(us(sym, name, "", opt="1", status="not_selected_in_this_price_stage"))
            self.assertTrue(r["is_options_strategy"], name)
            self.assertFalse(r["d1_eligible"], name)
            self.assertEqual(r["d1_exclusion_reasons"], "options_strategy", name)
            self.assertFalse(r["needs_review"], name)

    def test_unresolved_source_hint_fails_closed(self):
        # 원천 힌트는 옵션 전략인데 규칙 어휘가 못 잡는 이름: 검토 표시와 함께 D1에서도 뺀다.
        r = C(us("ZZZZ", "Example Shield Strategy ETF", "", opt="1", status="not_selected_in_this_price_stage"))
        self.assertFalse(r["is_options_strategy"])
        self.assertTrue(r["needs_review"])
        self.assertFalse(r["d1_eligible"])
        self.assertEqual(r["d1_exclusion_reasons"], "options_hint_unresolved")
        inv = C(us("ZZZY", "Example Downside Tracker ETF", "", inv="1", status="not_selected_in_this_price_stage"))
        self.assertEqual(inv["d1_exclusion_reasons"], "inverse_hint_unresolved")

    def test_source_hints_already_explained_by_rule_do_not_fail_closed(self):
        ns = "not_selected_in_this_price_stage"
        # 이름의 배수 숫자를 규칙이 읽은 경우(1X 인버스, 1.25x)
        bear1 = C(us("AAPD", "Direxion Daily AAPL Bear 1X ETF", "", lev="1", inv="1", status=ns))
        self.assertEqual((bear1["leverage_multiple"], bear1["d1_exclusion_reasons"]), (-1.0, "inverse"))
        self.assertFalse(bear1["needs_review"])
        x125 = C(us("TSLL", "GraniteShares 1.25x Long TSLA Daily ETF", "", lev="1", status=ns))
        self.assertEqual((x125["leverage_multiple"], x125["d1_eligible"]), (1.25, True))
        # 초단기 채권의 Short는 인버스가 아님
        bond = C(us("VUSB", "Vanguard Ultra-Short Bond ETF", "Vanguard", inv="1", status=ns))
        self.assertTrue(bond["d1_eligible"])
        self.assertFalse(bond["needs_review"])
        # 롱숏 주식은 배수 미확인으로 제외하고 검토 표시
        ls = C(us("FTLS", "First Trust Long/Short Equity", "", inv="1", status=ns))
        self.assertIsNone(ls["leverage_multiple"])
        self.assertEqual(ls["d1_exclusion_reasons"], "leverage_unknown")
        self.assertTrue(ls["needs_review"])

    def test_generic_other_issuers(self):
        bull = C(us("TSMX", "Direxion Daily TSM Bull 2X ETF", "", status="not_selected_in_this_price_stage"))
        self.assertEqual(bull["leverage_multiple"], 2.0)
        self.assertFalse(bull["has_price_history"])
        short2 = C(us("AMZO", "Tradr 2X Short AMZN Daily ETF", "", status="not_selected_in_this_price_stage"))
        self.assertEqual((short2["leverage_multiple"], short2["is_inverse"]), (-2.0, True))
        self.assertFalse(short2["needs_review"])
        bond = C(us("BAMU", "Brookstone Ultra-Short Bond ETF", "", status="not_selected_in_this_price_stage"))
        self.assertFalse(bond["is_inverse"])
        etn = C(us("SMHU", "MicroSectors 3x Long Semiconductor ETNs", "", status="not_selected_in_this_price_stage"))
        self.assertTrue(etn["is_etn"])
        self.assertEqual(etn["leverage_multiple"], 3.0)
        self.assertEqual(etn["d1_exclusion_reasons"], "etn;leverage_3x_or_more")


class TestNormalizationAndIO(unittest.TestCase):
    def test_kr_base_strips_wrappers(self):
        self.assertEqual(md.kr_base(md.kr_rest("TIGER 미국S&P500레버리지(합성 H)")), "미국S&P500")
        self.assertEqual(md.kr_base(md.kr_rest("KODEX 200TR")), "200")
        self.assertEqual(md.kr_base(md.kr_rest("KODEX TRF3070")), "TRF3070")
        self.assertEqual(md.kr_base(md.kr_rest("KODEX 미국나스닥100미국채혼합50")), "미국나스닥100")

    def test_us_core_strips_brand_and_multiplier(self):
        self.assertEqual(md.us_core("ProShares UltraPro Short QQQ"), "QQQ")
        self.assertEqual(md.us_core("State Street Technology Select Sector SPDR ETF"), "TECHNOLOGY SELECT SECTOR")
        self.assertEqual(md.us_core("iShares Core S&P 500 ETF"), "S&P 500")

    def _write(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False)
        tmp.write(text.encode("utf-8-sig"))
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return Path(tmp.name)

    def _csv(self, rows):
        header = ",".join(md.REQUIRED_COLUMNS)
        lines = [header] + [",".join(r) for r in rows]
        return "\n".join(lines) + "\n"

    def test_load_universe_keeps_codes_and_hangul(self):
        rows = [
            ["KR", "069500", "KODEX 200", "1", "", "", "", "downloaded", "0", "0", "0", "0"],
            ["KR", "0000D0", "TIGER 엔비디아미국채커버드콜밸런스(합성)", "7", "", "", "", "downloaded", "0", "0", "0", "1"],
        ]
        df = md.load_universe(self._write(self._csv(rows)))
        self.assertEqual(df["symbol"].tolist(), ["069500", "0000D0"])
        self.assertEqual(df["name"].iloc[1], "TIGER 엔비디아미국채커버드콜밸런스(합성)")
        meta = md.build_metadata(df)
        self.assertEqual(list(meta.columns), md.OUTPUT_COLUMNS)
        self.assertEqual(meta.loc[1, "asset_class"], "혼합")
        self.assertTrue(meta.loc[1, "is_options_strategy"])

    def test_load_universe_rejects_replacement_char(self):
        rows = [["KR", "069500", "KODEX �200", "1", "", "", "", "downloaded", "0", "0", "0", "0"]]
        with self.assertRaises(ValueError):
            md.load_universe(self._write(self._csv(rows)))

    def test_load_universe_rejects_header_only(self):
        with self.assertRaises(ValueError):
            md.load_universe(self._write(self._csv([])))

    def test_main_failure_keeps_previous_output(self):
        # Codex 검토 회귀: 헤더만 있는 입력이 기존 산출물을 먼저 덮어쓴 뒤 실패했다.
        out = Path(tempfile.mkdtemp()) / "etf_metadata.csv"
        self.addCleanup(lambda: __import__("shutil").rmtree(out.parent, ignore_errors=True))
        out.write_text("previous good output\n", encoding="utf-8")
        empty = self._write(self._csv([]))
        with self.assertRaises(ValueError):
            md.main(["--universe", str(empty), "--output", str(out)])
        self.assertEqual(out.read_text(encoding="utf-8"), "previous good output\n")
        self.assertEqual([p.name for p in out.parent.iterdir()], ["etf_metadata.csv"])

    def test_write_atomic_failure_keeps_previous_output(self):
        out = Path(tempfile.mkdtemp()) / "etf_metadata.csv"
        self.addCleanup(lambda: __import__("shutil").rmtree(out.parent, ignore_errors=True))
        out.write_text("previous good output\n", encoding="utf-8")
        meta = md.build_metadata(pd.DataFrame([kr("069500", "KODEX 200", "1")]))
        with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                md.write_atomic(meta, out)
        self.assertEqual(out.read_text(encoding="utf-8"), "previous good output\n")
        self.assertEqual([p.name for p in out.parent.iterdir()], ["etf_metadata.csv"])
        with self.assertRaises(ValueError):
            md.write_atomic(meta.iloc[0:0], out)
        md.write_atomic(meta, out)
        self.assertEqual(pd.read_csv(out, dtype={"symbol": str}, encoding="utf-8-sig")["symbol"].tolist(), ["069500"])

    def test_summarize_counts(self):
        df = pd.DataFrame([
            kr("069500", "KODEX 200", "1"), kr("122630", "KODEX 레버리지", "3", lev="1"),
            us("SSO", "ProShares Ultra S&P500", "ProShares"), us("SPY", "State Street SPDR S&P 500 ETF Trust", "State Street SPDR"),
        ])
        s = md.summarize(md.build_metadata(df))
        self.assertEqual(s["priced_rows"], 4)
        self.assertEqual(s["leverage_2x_priced_by_market"], {"KR": 1, "US": 1})
        self.assertEqual(s["families_priced"], 2)


if __name__ == "__main__":
    unittest.main()
