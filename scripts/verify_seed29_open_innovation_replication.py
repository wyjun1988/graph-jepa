from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.open_innovation_replication import (
    evaluate_open_innovation_replication,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labeled_paths(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = str(value).partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"{label} values must use NAME=PATH")
        name = name.strip()
        if name in result:
            raise ValueError(f"duplicate {label}: {name}")
        result[name] = Path(raw_path.strip())
    return result


def source_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {"fold": fold, "path": str(path), "sha256": sha256_file(path)}
        for fold, path in sorted(paths.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a frozen seed-29 schema-4 open-innovation replication."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--stability-summary", required=True)
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--aggregate-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract)
    stability_path = Path(args.stability_summary)
    aggregate_path = Path(args.aggregate_summary)
    fold_paths = labeled_paths(args.fold, "fold")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source_hashes = {
        str(path): sha256_file(ROOT / str(path))
        for path in contract["source_sha256"]
    }
    payload = evaluate_open_innovation_replication(
        contract,
        json.loads(stability_path.read_text(encoding="utf-8")),
        {
            fold: json.loads(path.read_text(encoding="utf-8"))
            for fold, path in fold_paths.items()
        },
        json.loads(aggregate_path.read_text(encoding="utf-8")),
        source_hashes,
    )
    payload["inputs"] = {
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "stability": {"path": str(stability_path), "sha256": sha256_file(stability_path)},
        "folds": source_rows(fold_paths),
        "aggregate": {"path": str(aggregate_path), "sha256": sha256_file(aggregate_path)},
        "sources": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(source_hashes.items())
        ],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "DIAGNOSTIC_ONLY").touch()
    print(json.dumps(payload["gate"], ensure_ascii=False), flush=True)
    return 0 if payload["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
