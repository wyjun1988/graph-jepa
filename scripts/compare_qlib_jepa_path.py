from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_direct_state_mlp import compare_metric


METRICS = {
    "entry_path_ic": ("return_path_ic", "entry_path_ic"),
    "entry_path_ic_top300": ("return_path_ic_top300", "entry_path_ic_top300"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_qlib_to_jepa(
    qlib_daily: pd.DataFrame,
    jepa_daily: pd.DataFrame,
    horizons: list[int],
    superiority_t: float = 1.96,
) -> dict[str, Any]:
    if "split" not in qlib_daily.columns:
        raise ValueError("Qlib daily metrics must identify the validation/test split")
    qlib_test = qlib_daily[qlib_daily["split"] == "test"].copy()
    if qlib_test.empty:
        raise ValueError("Qlib daily metrics contain no test rows")
    result: dict[str, Any] = {
        "role": "research_only_qlib_vs_jepa_path_challenge",
        "test_used_for_selection": False,
        "live_orders_allowed": False,
        "gate_rule": (
            "pass unless Qlib minus JEPA paired daily IC is positive with "
            f"Newey-West t >= {float(superiority_t):.2f}"
        ),
        "horizons": {},
    }
    passed = True
    for horizon in horizons:
        row: dict[str, Any] = {}
        for name, (qlib_column, jepa_column) in METRICS.items():
            comparison = compare_metric(
                qlib_test,
                jepa_daily,
                int(horizon),
                qlib_column,
                jepa_column,
            )
            delta = comparison.pop("delta_direct_minus_jepa")
            qlib_metrics = comparison.pop("direct")
            qlib_superiority = bool(
                np.isfinite(float(delta["mean"]))
                and np.isfinite(float(delta["newey_west_t"]))
                and float(delta["mean"]) > 0.0
                and float(delta["newey_west_t"]) >= float(superiority_t)
            )
            row[name] = {
                "qlib": qlib_metrics,
                "jepa": comparison["jepa"],
                "delta_qlib_minus_jepa": delta,
                "qlib_significantly_superior": qlib_superiority,
                "passed_non_inferiority_challenge": not qlib_superiority,
            }
            passed = passed and not qlib_superiority
        result["horizons"][str(horizon)] = row
    result["status"] = "pass" if passed else "blocked"
    return result


def _parse_horizons(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise ValueError("horizons must be unique positive integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired test-only comparison of Qlib LightGBM and a JEPA latent path head."
    )
    parser.add_argument("--qlib-daily", required=True)
    parser.add_argument("--jepa-daily", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--superiority-t", type=float, default=1.96)
    args = parser.parse_args()

    qlib_path = Path(args.qlib_daily)
    jepa_path = Path(args.jepa_daily)
    result = compare_qlib_to_jepa(
        pd.read_csv(qlib_path),
        pd.read_csv(jepa_path),
        _parse_horizons(args.horizons),
        superiority_t=float(args.superiority_t),
    )
    result["inputs"] = {
        "qlib_daily": str(qlib_path),
        "qlib_daily_sha256": sha256_file(qlib_path),
        "jepa_daily": str(jepa_path),
        "jepa_daily_sha256": sha256_file(jepa_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
