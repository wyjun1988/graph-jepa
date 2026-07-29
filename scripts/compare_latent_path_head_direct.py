from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_direct_state_mlp import compare_metric


PATH_COLUMNS = {
    "entry_path_ic": ("return_path_ic", "entry_path_ic"),
    "entry_path_ic_top300": ("return_path_ic_top300", "entry_path_ic_top300"),
}


def compare_head_to_direct(
    original_combined: dict[str, Any],
    head_daily: pd.DataFrame,
    challengers: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if not challengers:
        raise ValueError("at least one direct challenger is required")
    result = deepcopy(original_combined)
    for horizon, row in result["horizons"].items():
        for metric, (direct_column, head_column) in PATH_COLUMNS.items():
            candidates = []
            for label, frame in challengers.items():
                comparison = compare_metric(
                    frame,
                    head_daily,
                    int(horizon),
                    direct_column,
                    head_column,
                )
                delta = float(comparison["delta_direct_minus_jepa"]["mean"])
                candidates.append((delta, label, comparison))
            _delta, selected_label, selected = max(
                candidates, key=lambda item: (item[0], item[1])
            )
            row[metric] = selected
            row[f"selected_direct_challenger_for_{metric}"] = selected_label
            row[f"challenger_{metric}_deltas"] = {
                label: float(comparison["delta_direct_minus_jepa"]["mean"])
                for _value, label, comparison in candidates
            }
    result["path_reference"] = "latent_trajectory_residual_head"
    result["direct_path_challengers"] = sorted(challengers)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-pair direct path IC with a latent trajectory path head."
    )
    parser.add_argument("--original-combined", required=True)
    parser.add_argument("--head-daily", required=True)
    parser.add_argument("--challenger", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    challengers = {}
    for value in args.challenger:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("challengers must use LABEL=DAILY_CSV")
        challengers[label] = pd.read_csv(raw_path)
    if not challengers:
        raise ValueError("at least one direct challenger is required")
    original = json.loads(Path(args.original_combined).read_text(encoding="utf-8"))
    result = compare_head_to_direct(
        original,
        pd.read_csv(args.head_daily),
        challengers,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
