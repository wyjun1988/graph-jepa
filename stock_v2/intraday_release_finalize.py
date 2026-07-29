from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stock_v2.intraday_release_audit import canonical_sha256, file_sha256


FINALIZATION_CONTRACT = "portable_intraday_trajectory_release_finalization_v1"


def _relative_output(
    root: Path,
    value: str,
    label: str,
    *,
    legacy_path_base: Path | None,
) -> str:
    resolved_root = root.resolve()
    candidate = Path(str(value))
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        release_relative = (root / candidate).resolve()
        legacy_relative = (
            (legacy_path_base / candidate).resolve()
            if legacy_path_base is not None
            else None
        )
        if release_relative.is_file():
            resolved = release_relative
        elif legacy_relative is not None and legacy_relative.is_file():
            resolved = legacy_relative
        else:
            resolved = release_relative
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the release directory") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    return relative.as_posix()


def finalize_intraday_trajectory_release(
    release_dir: str | Path,
    *,
    code_files: Mapping[str, str | Path],
    legacy_path_base: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(release_dir)
    path_base = Path(legacy_path_base) if legacy_path_base is not None else None
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("trajectory release manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("live_orders_allowed") is not False:
        raise ValueError("trajectory release does not prohibit live orders")
    if manifest.get("promotion_eligible") is not False:
        raise ValueError("trajectory release unexpectedly enables promotion")

    existing = manifest.get("finalization")
    if existing is not None:
        if existing.get("contract") != FINALIZATION_CONTRACT:
            raise ValueError("trajectory release has an unknown finalization contract")
        return manifest

    original_manifest_sha256 = file_sha256(manifest_path)
    outputs = manifest["outputs"]
    index_path = Path(
        _relative_output(
            root,
            outputs["timestamp_index"],
            "timestamp index",
            legacy_path_base=path_base,
        )
    )
    if file_sha256(root / index_path) != outputs["timestamp_index_sha256"]:
        raise ValueError("timestamp index checksum mismatch before finalization")
    outputs["timestamp_index"] = index_path.as_posix()

    for record in outputs["shards"]:
        ticker = str(record["ticker"])
        relative = Path(
            _relative_output(
                root,
                record["path"],
                f"ticker shard {ticker}",
                legacy_path_base=path_base,
            )
        )
        if file_sha256(root / relative) != record["sha256"]:
            raise ValueError(f"ticker shard checksum mismatch before finalization: {ticker}")
        record["path"] = relative.as_posix()
    outputs["shards_sha256"] = canonical_sha256(outputs["shards"])

    provenance: dict[str, Any] = {}
    for name, value in sorted(code_files.items()):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"code provenance file is missing: {name}")
        provenance[str(name)] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    if not provenance:
        raise ValueError("at least one code provenance file is required")
    manifest["code_provenance"] = {
        "files": provenance,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest["finalization"] = {
        "contract": FINALIZATION_CONTRACT,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_manifest_sha256": original_manifest_sha256,
        "portable_output_paths": True,
        "output_payloads_unchanged": True,
    }

    temporary = Path(str(manifest_path) + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest
