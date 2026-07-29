from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.seed_stability import evaluate_seed_stability


def labeled_paths(values: list[str], role: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = str(value).partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError(f"{role} values must use LABEL=PATH")
        label = label.strip()
        if label in result:
            raise ValueError(f"duplicate {role} label: {label}")
        result[label] = Path(raw_path.strip())
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {"fold": fold, "path": str(path), "sha256": file_sha256(path)}
        for fold, path in sorted(paths.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paired five-fold Graph-JEPA encoder seed stability."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reference-fold", action="append", required=True)
    parser.add_argument("--candidate-fold", action="append", required=True)
    parser.add_argument("--direct-comparison", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    reference_paths = labeled_paths(args.reference_fold, "reference fold")
    candidate_paths = labeled_paths(args.candidate_fold, "candidate fold")
    direct_paths = labeled_paths(args.direct_comparison, "direct comparison")
    payload = evaluate_seed_stability(
        contract,
        {fold: pd.read_csv(path) for fold, path in reference_paths.items()},
        {fold: pd.read_csv(path) for fold, path in candidate_paths.items()},
        {
            fold: json.loads(path.read_text(encoding="utf-8"))
            for fold, path in direct_paths.items()
        }
        if direct_paths
        else None,
    )
    payload["inputs"] = {
        "contract": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "reference": source_rows(reference_paths),
        "candidate": source_rows(candidate_paths),
        "direct_comparisons": source_rows(direct_paths),
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
