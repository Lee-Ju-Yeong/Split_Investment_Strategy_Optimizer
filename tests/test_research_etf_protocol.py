"""ETF 연구 프로토콜 JSON과 탐색용·보류용 분할 검사. 네트워크·실제 자료 없이 합성 자료로 돈다."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.etf import holdout as ho
from src.research.etf import protocol as pr


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.proto = pr.load()

    def test_shipped_protocol_is_valid_draft(self):
        self.assertEqual(self.proto["status"], "draft")
        self.assertEqual(self.proto["universe"]["liquidity"]["floor"], {"KR": 100000000, "US": 1000000})
        self.assertEqual(self.proto["schedule"]["evaluation_time_kst"], "18:00")
        self.assertIn("baskets.k_values", pr.open_items(self.proto))

    def test_hash_ignores_key_order_and_whitespace(self):
        reordered = json.loads(json.dumps(self.proto, sort_keys=False), object_pairs_hook=lambda kv: dict(reversed(kv)))
        self.assertEqual(pr.protocol_hash(reordered), pr.protocol_hash(self.proto))
        changed = copy.deepcopy(self.proto)
        changed["holdout"]["fraction"] = 0.31
        self.assertNotEqual(pr.protocol_hash(changed), pr.protocol_hash(self.proto))

    def test_frozen_rejects_open_items(self):
        frozen = copy.deepcopy(self.proto)
        frozen["status"] = "frozen"
        with self.assertRaisesRegex(ValueError, "미정 값"):
            pr.validate(frozen)

    def test_rejects_null_and_out_of_range(self):
        bad = copy.deepcopy(self.proto)
        bad["fx"]["conversion"]["cost_one_way"] = None
        with self.assertRaisesRegex(ValueError, "비어"):
            pr.validate(bad)
        bad = copy.deepcopy(self.proto)
        bad["holdout"]["fraction"] = 1.5
        with self.assertRaises(ValueError):
            pr.validate(bad)
        bad = copy.deepcopy(self.proto)
        bad["success"]["bootstrap"]["resamples"] = True  # bool은 숫자로 보지 않는다
        with self.assertRaises(ValueError):
            pr.validate(bad)
        bad = copy.deepcopy(self.proto)
        del bad["tax"]
        with self.assertRaisesRegex(ValueError, "절이 없"):
            pr.validate(bad)

    def test_rejects_weakened_contracts(self):
        cases = [
            (("success", "min_effect"), lambda d: d["success"]["min_effect"].pop("dMDD_pp")),
            (("empty open",), lambda d: d["baskets"].__setitem__("k_values", "")),
            (("bare decide_by",), lambda d: d["baskets"].__setitem__("k_values", "decide_by: ")),
            (("negative fee",), lambda d: d["execution"]["costs_one_way"].__setitem__("US_commission", -0.001)),
            (("fractional seed",), lambda d: d["holdout"].__setitem__("seed", 1.5)),
            (("fractional resamples",), lambda d: d["success"]["bootstrap"].__setitem__("resamples", 2000.5)),
            (("bad time",), lambda d: d["schedule"].__setitem__("evaluation_time_kst", "99:99")),
            (("infinite",), lambda d: d["execution"]["fill"].__setitem__("slippage_ticks", float("inf"))),
            (("missing text",), lambda d: d["success"].pop("population")),
            (("unknown enum",), lambda d: d["universe"].__setitem__("gold_family_etfs", "maybe")),
            (("stress below main",), lambda d: d["fx"]["conversion"].__setitem__("cost_stress_one_way", 0.0001)),
            (("stress below fee",), lambda d: d["execution"]["costs_one_way"].__setitem__("stress_one_way_all", 0.001)),
            (("bad stress ref",), lambda d: d["success"]["stress_applies"].append("fx.nope")),
            (("window order",), lambda d: d["holdout"]["correlation_merge"]["window"].__setitem__("end", "1990-01-01")),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                bad = copy.deepcopy(self.proto)
                mutate(bad)
                with self.assertRaises(ValueError):
                    pr.validate(bad)

    def test_load_rejects_nan_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            text = json.dumps(self.proto, ensure_ascii=False)
            path.write_text(text.replace('"interest_primary": 0.0', '"interest_primary": NaN'), encoding="utf-8")
            with self.assertRaises(ValueError):
                pr.load(path)


def _meta(rows):
    base = {"name": "", "has_price_history": True, "d1_eligible": True, "is_single_stock": False,
            "leverage_multiple": 1.0, "asset_class": "EQ"}
    return pd.DataFrame([{**base, **r} for r in rows])


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.proto = pr.load()
        self.rng = np.random.default_rng(7)

    def test_eligible_excludes_gold_family_single_stock_leverage_and_d1(self):
        meta = _meta([
            {"market": "KR", "symbol": "000001", "index_family": "A"},
            {"market": "KR", "symbol": "000002", "index_family": "GOLD"},
            {"market": "KR", "symbol": "000003", "index_family": "B", "is_single_stock": True, "leverage_multiple": 2.0},
            {"market": "KR", "symbol": "000004", "index_family": "C", "is_single_stock": True},
            {"market": "US", "symbol": "XYZ", "index_family": "D", "d1_eligible": False},
            {"market": "US", "symbol": "NOP", "index_family": "E", "has_price_history": False},
        ])
        self.assertEqual(ho.eligible(meta, self.proto)["symbol"].tolist(), ["000001", "000004"])
        meta.loc[0, "index_family"] = " "
        with self.assertRaisesRegex(ValueError, "index_family"):
            ho.eligible(meta, self.proto)

    def test_pairwise_correlation_matches_pandas_with_gaps(self):
        frame = pd.DataFrame(self.rng.normal(size=(300, 4)), columns=list("abcd"))
        frame.iloc[:50, 1] = np.nan
        frame.iloc[100:180, 2] = np.nan
        corr, n = ho.pairwise_correlation(frame)
        expected = frame.corr(min_periods=1).to_numpy()
        np.testing.assert_allclose(corr, expected, atol=1e-10)
        self.assertEqual(n[1, 2], frame.iloc[:, [1, 2]].dropna().shape[0])

    def test_correlation_merge_needs_threshold_and_overlap(self):
        base = self.rng.normal(size=400)
        frame = pd.DataFrame({
            "x1": base,
            "x2": 2 * base + self.rng.normal(scale=0.01, size=400),   # 같은 지수의 2배
            "y": self.rng.normal(size=400),
            "short": np.r_[np.full(300, np.nan), base[300:]],      # 겹친 표본 100개뿐
        })
        pairs = ho.correlated_pairs(frame, threshold=0.98, min_overlap=250)
        self.assertEqual([(a, b) for a, b, *_ in pairs], [("x1", "x2")])

    def test_constant_and_near_constant_series_do_not_merge(self):
        frame = pd.DataFrame({"a": np.full(300, 1e-4), "b": np.full(300, 2e-4),
                              "c": 1e-4 + np.r_[0.0, np.full(299, 1e-15)]})
        corr, _ = ho.pairwise_correlation(frame)
        self.assertTrue(np.isnan(corr[0, 1]))
        self.assertEqual(ho.correlated_pairs(frame, 0.98, 250), [])

    def test_zero_volume_return_is_missing_regardless_of_other_symbols(self):
        def daily(rows):
            return pd.DataFrame(rows, columns=["symbol", "date", "close", "volume"]).assign(
                date=lambda f: pd.to_datetime(f["date"]))
        alone = daily([("A", "2024-01-02", 100, 5), ("A", "2024-01-03", 100, 0), ("A", "2024-01-04", 110, 5)])
        other = pd.concat([alone, daily([("B", d, 50, 5) for d in ("2024-01-02", "2024-01-03", "2024-01-04")])])
        for frame in (alone, other):
            returns = ho.daily_returns(frame, frame["symbol"].unique())
            self.assertTrue(np.isnan(returns.loc[pd.Timestamp("2024-01-04"), "A"]))

    def test_merge_labels_do_not_depend_on_pair_order(self):
        cands = _meta([{"market": "US", "symbol": s, "index_family": f}
                       for s, f in (("A", "FA"), ("B", "FB"), ("C", "FC"), ("C2", "FC"))])
        pairs = [("A", "B", 0.99, 300), ("B", "C", 0.99, 300)]
        first, _ = ho.merge_families(cands, pairs)
        second, _ = ho.merge_families(cands, list(reversed(pairs)))
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(set(first), {"FC"})

    def test_small_strata_are_coarsened_before_assignment(self):
        rows, first = [], {}
        for i in range(12):
            rows.append({"market": "US", "symbol": f"S{i}", "index_family": f"F{i}",
                         "asset_class": "EQ" if i < 11 else "BOND"})
            year = 2010 if i < 6 else (2020 if i < 9 else 2015)
            first[f"S{i}"] = pd.Timestamp(f"{year}-01-01")
        frame = _meta(rows).assign(group=lambda f: f["index_family"], first_date=lambda f: f["symbol"].map(first))
        names = ho.strata(frame, self.proto)
        self.assertEqual(names["F0"], "US|EQ|long")      # long 6개라 그대로
        self.assertEqual(names["F7"], "US|EQ|*")         # short 3개 + mid 2개 → 자산군 단위 5개
        self.assertEqual(names["F10"], "US|EQ|*")
        self.assertEqual(names["F11"], "US|*|*")         # BOND 1개 → 시장 단위(작은 층으로 남음)

    def test_merge_keeps_larger_family_name(self):
        cands = _meta([
            {"market": "US", "symbol": "A1", "index_family": "BIG"},
            {"market": "US", "symbol": "A2", "index_family": "BIG"},
            {"market": "US", "symbol": "Z1", "index_family": "AAA_SMALL"},
            {"market": "US", "symbol": "C1", "index_family": "OTHER"},
        ])
        group, log = ho.merge_families(cands, [("A1", "Z1", 0.99, 500)])
        self.assertEqual(group.tolist(), ["BIG", "BIG", "BIG", "OTHER"])
        self.assertEqual(len(log), 1)

    def _synthetic(self, n_families=60):
        rows, first, kr, us = [], {}, {}, {}
        days = pd.bdate_range("2010-01-01", periods=600)
        for i in range(n_families):
            market = "KR" if i % 2 else "US"
            family = f"F{i:03d}"
            factor = self.rng.normal(size=len(days))
            for j in range(2):
                symbol = f"{market}{i:03d}{j}"
                rows.append({"market": market, "symbol": symbol, "index_family": family,
                             "asset_class": "EQ" if i % 3 else "BOND"})
                first[symbol] = pd.Timestamp("2010-01-01") if i % 4 else pd.Timestamp("2020-01-01")
                (kr if market == "KR" else us)[symbol] = factor + self.rng.normal(scale=2.0, size=len(days))
        # 이름은 다르지만 수익률이 같은 두 묶음(F000·F002, 둘 다 미국)
        us["US0020"] = us["US0000"] + self.rng.normal(scale=1e-3, size=len(days))
        # 시장을 가로지른 묶음: F001(국내) 상품 하나를 미국 상장으로
        rows.append({"market": "US", "symbol": "XUS001", "index_family": "F001", "asset_class": "EQ"})
        first["XUS001"] = pd.Timestamp("2010-01-01")
        us["XUS001"] = self.rng.normal(size=len(days))
        returns = {"KR": pd.DataFrame(kr, index=days), "US": pd.DataFrame(us, index=days)}
        return _meta(rows), pd.Series(first), returns

    def test_split_keeps_groups_together_and_is_deterministic(self):
        cands, first, returns = self._synthetic()
        result, summary = ho.split(cands, first, returns, self.proto)
        again, _ = ho.split(cands, first, returns, self.proto)
        pd.testing.assert_frame_equal(result, again)
        etfs = result[result["market"] != "GOLD"]
        self.assertTrue((etfs.groupby("group")["split"].nunique() == 1).all())     # 묶음째 한쪽
        self.assertEqual(etfs.loc[etfs["symbol"] == "US0020", "group"].item(),
                         etfs.loc[etfs["symbol"] == "US0000", "group"].item())    # 상관 병합
        self.assertEqual(etfs.loc[etfs["symbol"] == "XUS001", "split"].item(),
                         etfs.loc[etfs["symbol"] == "KR0010", "split"].item())    # 시장 교차 묶음
        self.assertTrue(etfs.loc[etfs["symbol"] == "XUS001", "stratum"].item().startswith("KR+US|"))
        gold = result[result["market"] == "GOLD"]
        self.assertEqual(gold["split"].tolist(), ["both"])
        share = summary["groups_by_split"]["holdout"] / sum(summary["groups_by_split"].values())
        self.assertGreater(share, 0.15)
        self.assertLess(share, 0.45)
        self.assertEqual(summary["merges"], 1)

    def test_seed_changes_assignment(self):
        cands, first, returns = self._synthetic()
        other = copy.deepcopy(self.proto)
        other["holdout"]["seed"] = 1
        a, _ = ho.split(cands, first, returns, self.proto)
        b, _ = ho.split(cands, first, returns, other)
        self.assertFalse(a["split"].equals(b["split"]))

    def test_missing_first_date_fails(self):
        cands, first, returns = self._synthetic(n_families=4)
        with self.assertRaisesRegex(ValueError, "첫 거래일"):
            ho.split(cands, first.drop(first.index[0]), returns, self.proto)

    def test_build_writes_split_and_manifest(self):
        cands, first, _ = self._synthetic(n_families=6)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            meta_csv = tmp / "etf_metadata.csv"
            cands.to_csv(meta_csv, index=False, encoding="utf-8-sig")
            bundle = tmp / "bundle"
            bundle.mkdir()
            days = pd.bdate_range("2012-01-02", periods=300)
            for market in ("KR", "US"):
                rows = []
                for symbol in cands.loc[cands["market"] == market, "symbol"]:
                    close = 100 * np.exp(np.cumsum(self.rng.normal(scale=0.01, size=len(days))))
                    rows.append(pd.DataFrame({"symbol": symbol, "date": days.strftime("%Y-%m-%d"),
                                              "close": close, "volume": 10}))
                with gzip.open(bundle / f"{market.lower()}_daily.csv.gz", "wt", encoding="utf-8") as handle:
                    pd.concat(rows).to_csv(handle, index=False)
            markets = [{"market": m, "daily_file": f"{m.lower()}_daily.csv.gz",
                        "daily_sha256": ho._sha256(bundle / f"{m.lower()}_daily.csv.gz")} for m in ("KR", "US")]
            (bundle / "manifest.json").write_text(json.dumps({"markets": markets}), encoding="utf-8")
            manifest = ho.build(meta_csv, bundle, tmp / "out")
            self.assertEqual(ho.build(meta_csv, bundle, tmp / "out")["split_sha256"], manifest["split_sha256"])
            split_csv = tmp / "out" / manifest["split_csv"]
            self.assertTrue(split_csv.exists())
            saved = pd.read_csv(split_csv, dtype={"symbol": str})
            self.assertEqual(list(saved.columns), ho.SPLIT_COLUMNS)
            self.assertEqual(manifest["protocol_sha256"], pr.protocol_hash(self.proto))
            self.assertEqual(len(list((tmp / "out").glob(".*.tmp"))), 0)
            split_csv.write_text("tampered", encoding="utf-8")   # 고정된 분할은 덮어쓰지 않는다
            with self.assertRaises(FileExistsError):
                ho.build(meta_csv, bundle, tmp / "out")
            (bundle / "us_daily.csv.gz").write_bytes(b"changed")      # manifest 해시와 다른 가격 파일
            with self.assertRaisesRegex(RuntimeError, "해시"):
                ho.build(meta_csv, bundle, tmp / "out2")


if __name__ == "__main__":
    unittest.main()
