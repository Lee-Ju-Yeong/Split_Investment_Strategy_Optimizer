"""합성 파일 묶음→해시 고정→M5 작은 실행. 실제 거래 달력이나 실제 성과가 아니다."""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import csv
import gzip
import json
from pathlib import Path
import tempfile

from src.research.etf import protocol
from src.research.etf.engine_inputs import _sha
from src.research.etf.m5_inputs import pin_inputs
from src.research.etf.m5_pilot import run_pilot


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _at(day, hour):
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).isoformat()


def fixture(root, *, dense=False, us_only=False):
    """빠른 연결 검사는 2020년 이후 세션을 희소하게 둔다. 실제 달력이 아니다."""
    root = Path(root)
    days, day = [], date(2015, 1, 1)
    while day <= date(2023, 1, 6):
        sample = day.year < 2020 or day.day <= 4 or day.month == 12 and day.day >= 30 or day.year >= 2023
        if day.weekday() < 5 and (dense or sample):
            days.append(day)
        day += timedelta(days=1)
    for name, opening, closing in (("krx", 0, 6), ("nyse", 13, 20)):
        _csv(root / "fx" / f"calendar_{name}.csv", [
            {"date": day.isoformat(), "open_utc": _at(day, opening), "close_utc": _at(day, closing)} for day in days])
    _csv(root / "fx" / "fx_usdkrw_smbs_mar.csv", [
        {"available_at_utc": _at(day, 0), "value": "1300"} for day in days])
    metadata = [{"market": market, "symbol": symbol, "d1_eligible": "True", "kr_tax_class": tax,
                 "leverage_multiple": "1.0", "index_family": f"SYNTHETIC_{market}"}
                for market, symbol, tax in (("KR", "SYNTHETIC_KR", "국내주식형"), ("US", "SYNTHETIC_US", ""))]
    if us_only:
        metadata.append(dict(metadata[1], symbol="SYNTHETIC_US_2", index_family="SYNTHETIC_US_2"))
    _csv(root / "metadata" / "etf_metadata.csv", metadata)
    split = root / "protocol" / "holdout_split_synthetic.csv"
    split_rows = [{"market": r["market"], "symbol": r["symbol"], "group": r["index_family"], "split": "explore"}
                  for r in metadata]
    if us_only:
        split_rows[0]["split"] = "holdout"
    split_rows.append({"market": "KR", "symbol": "SYNTHETIC_HIDDEN", "group": "HIDDEN", "split": "holdout"})
    _csv(split, split_rows)
    bundle = root / "prices" / "runs" / "synthetic"
    hashes = {}
    for market, symbol in (("KR", "SYNTHETIC_KR"), ("US", "SYNTHETIC_US")):
        rows = [_price(market, symbol, day, index) for index, day in enumerate(days)]
        if us_only and market == "US":
            rows.extend(_price(market, "SYNTHETIC_US_2", day, index + 5) for index, day in enumerate(days))
        path = bundle / f"{market.lower()}_daily.csv.gz"
        _csv(path, rows)
        hashes[market] = _sha(path)
    (bundle / "manifest.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    (root / "prices" / "CURRENT").write_text("synthetic\n", encoding="utf-8")
    record = {"split_csv": split.name, "split_sha256": _sha(split), "protocol_sha256": protocol.protocol_hash(protocol.load()),
              "inputs": {"metadata_sha256": _sha(root / "metadata" / "etf_metadata.csv"),
                         "price_manifest_sha256": _sha(bundle / "manifest.json"), "daily_sha256": hashes}}
    split.with_suffix(".json").write_text(json.dumps(record), encoding="utf-8")
    pin_inputs(root, split, root / "m5-inputs.json", synthetic=True)
    return root / "m5-inputs.json"


def _price(market, symbol, day, index):
    unit = Decimal(10 if market == "KR" else 1)
    opening = (Decimal(100) + Decimal(index % 17 - 8) * 2) * unit
    closing = opening + (Decimal(index % 3) - 1) * unit
    return {"market": market, "symbol": symbol, "date": day.isoformat(), "open": str(opening),
            "high": str(max(opening, closing) + unit), "low": str(min(opening, closing) - unit),
            "close": str(closing), "volume": "2000000", "dividend": "0", "split_ratio": "1",
            "corp_action_ok": "True", "price_basis": "raw"}


def demo(directory, *, build_dir=None):
    directory = Path(directory)
    manifest = fixture(directory / "inputs")
    result = run_pilot(directory / "inputs", manifest, ["SYNTHETIC_KR", "SYNTHETIC_US"],
                       directory / "run", build_dir=build_dir)
    assert result["status"] == "passed", result
    assert result["synthetic"] is True and result["succeeded"] == 4
    return result


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        print(json.dumps(demo(temporary), ensure_ascii=False, indent=2))
