from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ASSEMBLY_CONTRACT = "time_major_intraday_post_impact_days_v1"
METADATA_KEYS = (
    "tickers",
    "populated_tickers",
    "feature_names",
    "horizons_minutes",
    "horizon_labels",
    "target_names",
    "systemic_target_names",
)
CONTRACT_KEYS = tuple(key for key in METADATA_KEYS if key != "populated_tickers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge immutable intraday day releases using hard-linked shards."
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--incremental-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{label} day-release manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("assembly_contract") != ASSEMBLY_CONTRACT
        or payload.get("transactional_publish") is not True
        or payload.get("portable_payload_paths") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("live_orders_allowed") is not False
    ):
        raise ValueError(f"{label} day-release contract is invalid")
    records = payload.get("day_shards")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} day release has no day shards")
    dates = [str(record["date"]) for record in records]
    if dates != sorted(set(dates)) or int(payload.get("days", -1)) != len(records):
        raise ValueError(f"{label} day-release dates are invalid")
    return path, payload


def _release_path(root: Path, relative: object) -> Path:
    path = Path(str(relative))
    if path.is_absolute():
        raise ValueError("day-release payload path must be relative")
    result = (root / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("day-release payload path escapes its root") from exc
    return result


def _load_metadata(root: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    record = manifest["metadata"]
    path = _release_path(root, record["path"])
    if file_sha256(path) != str(record["sha256"]):
        raise ValueError("day-release metadata checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(METADATA_KEYS):
            raise ValueError("day-release metadata keys changed")
        return {name: archive[name].copy() for name in METADATA_KEYS}


def _verify_record(root: Path, record: Mapping[str, Any]) -> Path:
    path = _release_path(root, record["path"])
    if not path.is_file():
        raise ValueError(f"day shard is missing: {record['date']}")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"day shard byte count differs: {record['date']}")
    if file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"day shard checksum differs: {record['date']}")
    return path


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def merge_day_releases(base_dir: Path, incremental_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() or Path(str(output_dir) + ".tmp").exists():
        raise FileExistsError(f"immutable merge output already exists: {output_dir}")
    base_manifest_path, base = _load_manifest(base_dir, "base")
    incremental_manifest_path, incremental = _load_manifest(
        incremental_dir, "incremental"
    )
    base_metadata = _load_metadata(base_dir, base)
    incremental_metadata = _load_metadata(incremental_dir, incremental)
    for key in CONTRACT_KEYS:
        if not np.array_equal(base_metadata[key], incremental_metadata[key]):
            raise ValueError(f"day-release metadata contract differs: {key}")
    base_dates = [str(record["date"]) for record in base["day_shards"]]
    incremental_dates = [
        str(record["date"]) for record in incremental["day_shards"]
    ]
    if incremental_dates[0] <= base_dates[-1]:
        raise ValueError("incremental day release does not strictly follow the base")

    temporary = Path(str(output_dir) + ".tmp")
    temporary.mkdir(parents=True)
    link_modes: dict[str, int] = {"hardlink": 0, "copy": 0}
    merged_records: list[dict[str, Any]] = []
    try:
        for root, records in (
            (base_dir, base["day_shards"]),
            (incremental_dir, incremental["day_shards"]),
        ):
            for record in records:
                source = _verify_record(root, record)
                destination = temporary / "days" / f"{record['date']}.npz"
                mode = _link_or_copy(source, destination)
                link_modes[mode] += 1
                merged = dict(record)
                merged["path"] = str(Path("days") / destination.name)
                merged_records.append(merged)

        populated_set = set(
            str(value) for value in base_metadata["populated_tickers"].tolist()
        ) | set(
            str(value)
            for value in incremental_metadata["populated_tickers"].tolist()
        )
        populated = [
            str(value)
            for value in base_metadata["tickers"].tolist()
            if str(value) in populated_set
        ]
        if len(populated) != len(populated_set):
            raise ValueError("populated ticker is absent from the stock axis")
        metadata_path = temporary / "metadata.npz"
        with metadata_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{
                    **{key: base_metadata[key] for key in CONTRACT_KEYS},
                    "populated_tickers": np.asarray(populated, dtype="U6"),
                },
            )
        merged_manifest = dict(base)
        merged_manifest.pop("h5_endpoint_target_node_coverage", None)
        merged_manifest.update(
            {
                "days": len(merged_records),
                "first_date": merged_records[0]["date"],
                "last_date": merged_records[-1]["date"],
                "populated_stocks": len(populated),
                "metadata": {
                    "path": metadata_path.name,
                    "sha256": file_sha256(metadata_path),
                },
                "day_shards": merged_records,
                "merge_provenance": {
                    "contract": "append_only_intraday_day_release_merge_v1",
                    "merged_at_utc": datetime.now(timezone.utc).isoformat(),
                    "base_manifest": str(base_manifest_path),
                    "base_manifest_sha256": file_sha256(base_manifest_path),
                    "incremental_manifest": str(incremental_manifest_path),
                    "incremental_manifest_sha256": file_sha256(
                        incremental_manifest_path
                    ),
                    "base_days": len(base_dates),
                    "incremental_days": len(incremental_dates),
                    "payload_materialization": link_modes,
                    "merger_sha256": file_sha256(Path(__file__)),
                },
                "transactional_publish": True,
                "portable_payload_paths": True,
                "promotion_eligible": False,
                "live_orders_allowed": False,
            }
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                merged_manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "output": str(output_dir),
        "manifest_sha256": file_sha256(output_dir / "manifest.json"),
        "days": len(merged_records),
        "first_date": merged_records[0]["date"],
        "last_date": merged_records[-1]["date"],
        "payload_materialization": link_modes,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }


def main() -> int:
    args = parse_args()
    result = merge_day_releases(
        Path(args.base_dir), Path(args.incremental_dir), Path(args.output_dir)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
