"""M5 입력 고정/운반 목록. 원본·보류 분할을 다시 만들거나 성과를 읽지 않는다."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import zipfile

from .engine_inputs import _check_hash, _extra_paths, _rows, _sha, _snapshot, load_m1_inputs
from .engine_lifecycle import DelistingEvent
from .engine_types import aware


SCHEMA = "etf-m5-inputs-v1"


def _relative(root, path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("M5 input files must stay inside data_root")
    return resolved.relative_to(root).as_posix()


def pin_inputs(data_root, split_csv, output, *, synthetic=False):
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be bool")
    root, split = Path(data_root).resolve(), Path(split_csv).resolve()
    symbols = [row["symbol"] for row in _rows(split) if row["split"] == "explore"
               or (row["market"] == "GOLD" and row["split"] == "both")]
    record, selected, _, bundle = _snapshot(root, split, symbols)
    files = [split, split.with_suffix(".json"), root / "metadata" / "etf_metadata.csv",
             root / "prices" / "CURRENT", bundle / "manifest.json"]
    for market, expected in record["inputs"]["daily_sha256"].items():
        daily = bundle / f"{market.lower()}_daily.csv.gz"
        _check_hash(daily, expected)
        files.append(daily)
    extras = _extra_paths(root, any(row["market"] == "GOLD" for row in selected.values()))
    files.extend(extras.values())
    controls = root / "protocol" / "m5_controls.json"
    if controls.exists():
        files.append(controls)
    result = {"schema": SCHEMA, "synthetic": synthetic, "split_csv": _relative(root, split),
              "controls": _relative(root, controls) if controls.exists() else None,
              "files": {_relative(root, path): _sha(path) for path in files},
              "extra_inputs_sha256": {name: _sha(path) for name, path in extras.items()},
              "original_split_protocol_sha256": record["protocol_sha256"]}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def read_manifest(data_root, manifest_path, symbols):
    root = Path(data_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or type(manifest.get("synthetic")) is not bool:
        raise ValueError("invalid M5 input manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or any(not isinstance(name, str) for name in files):
        raise ValueError("M5 input manifest needs a file/hash map")
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
           for value in files.values()):
        raise ValueError("input hashes must be SHA256 hex digests")
    for name in files:
        if name != _relative(root, root / name):
            raise ValueError("input manifest paths must be canonical relative paths")
    split_name = manifest.get("split_csv")
    if split_name not in files:
        raise ValueError("split file is missing from the input manifest")
    split = root / split_name
    _check_hash(split, files[split_name])
    rows = list(_rows(split))
    membership = {row["symbol"]: row for row in rows}
    if len(membership) != len(rows):
        raise ValueError("split symbols must be unique")
    for symbol in symbols:
        row = membership.get(symbol)
        if row is None or (row["split"] != "explore" and not (row["market"] == "GOLD" and row["split"] == "both")):
            raise ValueError(f"M5 may not read holdout/unassigned prices: {symbol}")
    for name, expected in files.items():
        _check_hash(root / name, expected)
    _manifest_closure(root, split, manifest, membership)
    return manifest


def _manifest_closure(root, split, manifest, membership):
    allowed = [row["symbol"] for row in membership.values() if row["split"] == "explore"
               or (row["market"] == "GOLD" and row["split"] == "both")]
    record, selected, _, bundle = _snapshot(root, split, allowed)
    paths = [split, split.with_suffix(".json"), root / "metadata" / "etf_metadata.csv",
             root / "prices" / "CURRENT", bundle / "manifest.json"]
    paths.extend(bundle / f"{market.lower()}_daily.csv.gz" for market in record["inputs"]["daily_sha256"])
    extras = _extra_paths(root, any(row["market"] == "GOLD" for row in selected.values()))
    paths.extend(extras.values())
    if manifest.get("controls") is not None:
        if manifest["controls"] != "protocol/m5_controls.json":
            raise ValueError("invalid controls path")
        paths.append(root / manifest["controls"])
    if not {_relative(root, path) for path in paths} <= set(manifest["files"]):
        raise ValueError("input manifest is missing required source files")
    expected = {name: manifest["files"][_relative(root, path)] for name, path in extras.items()}
    if manifest.get("extra_inputs_sha256") != expected:
        raise ValueError("extra input hashes differ from pinned file hashes")
    if manifest.get("original_split_protocol_sha256") != record["protocol_sha256"]:
        raise ValueError("original split protocol hash differs")


def _controls(root, manifest, symbols, start, end):
    name = manifest.get("controls")
    if name is None:
        return {}, {}, ()
    if name not in manifest["files"]:
        raise ValueError("controls must be pinned in the input manifest")
    values = json.loads((Path(root) / name).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - {"locked_status", "payment_dates", "delistings"}:
        raise ValueError("invalid M5 controls")
    locks = _dated_controls(values.get("locked_status", []), "locked", symbols, start, end)
    payments = _dated_controls(values.get("payment_dates", []), "pay_at", symbols, start, end)
    notices = []
    seen = set()
    for row in values.get("delistings", []):
        event = DelistingEvent(row["symbol"], datetime.fromisoformat(row["known_at"]),
                              date.fromisoformat(row["last_session"]), date.fromisoformat(row["delisting_date"]),
                              row["source_sha256"])
        if event.symbol in seen:
            raise ValueError("duplicate delisting controls")
        seen.add(event.symbol)
        if event.symbol in symbols and start <= event.last_session <= end:
            notices.append(event)
    return locks, payments, tuple(notices)


def _dated_controls(rows, field, symbols, start, end):
    result, seen = {}, set()
    for row in rows:
        key = (row["symbol"], date.fromisoformat(row["date"]))
        if key in seen:
            raise ValueError("duplicate dated controls")
        seen.add(key)
        value = row[field]
        if field == "locked" and type(value) is not bool:
            raise ValueError("locked must be bool")
        if field == "pay_at":
            value = aware(datetime.fromisoformat(value))
        if key[0] in symbols and start <= key[1] <= end:
            result[key] = value
    return result


def load_m5_inputs(data_root, manifest_path, symbols, start, end, *, max_fx_age_days=7, fx_closure_age_days=None):
    manifest = read_manifest(data_root, manifest_path, symbols)
    locks, payments, notices = _controls(data_root, manifest, symbols, start, end)
    result = load_m1_inputs(data_root, symbols, start, end, Path(data_root) / manifest["split_csv"],
                            locked_status=locks, payment_dates=payments, delistings=notices,
                            input_hashes=manifest["extra_inputs_sha256"], domestic_settlement="t_plus_2",
                            max_fx_age_days=max_fx_age_days, fx_closure_age_days=fx_closure_age_days)
    result.provenance.update(m5_input_manifest_sha256=_sha(manifest_path), synthetic=manifest["synthetic"],
                             delistings=[asdict(event) for event in notices])
    return result


def pack_inputs(data_root, manifest_path, output):
    """최소 원본 파일+manifest를 운반한다. ZIP 내부 바이트도 고정 해시와 대조한다."""
    root, manifest_path = Path(data_root).resolve(), Path(manifest_path).resolve()
    manifest = read_manifest(root, manifest_path, [])
    if "m5-inputs.json" in manifest["files"]:
        raise ValueError("reserved archive manifest name")
    expected = dict(manifest["files"], **{"m5-inputs.json": _sha(manifest_path)})
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for name in manifest["files"]:
            archive.write(root / name, name)
        archive.write(manifest_path, "m5-inputs.json")
    with zipfile.ZipFile(output) as archive:
        for name, digest in expected.items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"input changed while packaging: {name}; archive is invalid")
    return {"archive": str(output), "sha256": _sha(Path(output)), "files": len(expected)}


def main():
    parser = argparse.ArgumentParser(description="고정 M1 자료에서 M5 입력 해시와 전송 파일 목록 생성")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="새 JSON 경로")
    parser.add_argument("--synthetic", action="store_true", help="합성 fixture임을 명시")
    parser.add_argument("--archive", type=Path, help="선택: 새 ZIP 경로. 원본 파일 해시 검증 후 포장")
    args = parser.parse_args()
    manifest = pin_inputs(args.data_root, args.split_csv, args.output, synthetic=args.synthetic)
    if args.archive is not None:
        print(json.dumps(pack_inputs(args.data_root, args.output, args.archive), ensure_ascii=False))
    print(json.dumps({"manifest": str(args.output), "sha256": _sha(args.output),
                      "files": list(manifest["files"]), "synthetic": manifest["synthetic"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
