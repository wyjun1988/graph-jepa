from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


def parse_challenger(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("challengers must use LABEL=COMPARISON_JSON")
    return label.strip(), Path(raw_path.strip())


def combine(challengers: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    if len(challengers) < 2:
        raise ValueError("at least two direct challengers are required")
    horizon_sets = [set(data.get("horizons") or {}) for _label, data in challengers]
    if not horizon_sets[0] or any(values != horizon_sets[0] for values in horizon_sets[1:]):
        raise ValueError("direct challengers must contain identical non-empty horizons")

    selected_horizons: dict[str, Any] = {}
    for horizon in sorted(horizon_sets[0], key=int):
        ranked = []
        for label, data in challengers:
            row = data["horizons"][horizon]
            delta = float(
                row["state_skill"]["delta_direct_minus_jepa"]["mean"]
            )
            ranked.append((delta, label, row))
        _delta, selected_label, selected_row = max(
            ranked,
            key=lambda item: (item[0], item[1]),
        )
        combined_row = deepcopy(selected_row)
        combined_row["selected_direct_challenger"] = selected_label
        combined_row["challenger_state_skill_deltas"] = {
            label: float(row["state_skill"]["delta_direct_minus_jepa"]["mean"])
            for _value, label, row in ranked
        }
        for metric in ("entry_path_ic", "entry_path_ic_top300"):
            metric_ranked = []
            for label, data in challengers:
                metric_row = data["horizons"][horizon][metric]
                metric_delta = float(
                    metric_row["delta_direct_minus_jepa"]["mean"]
                )
                metric_ranked.append((metric_delta, label, metric_row))
            _metric_delta, metric_label, metric_row = max(
                metric_ranked,
                key=lambda item: (item[0], item[1]),
            )
            combined_row[metric] = deepcopy(metric_row)
            combined_row[f"selected_direct_challenger_for_{metric}"] = metric_label
            combined_row[f"challenger_{metric}_deltas"] = {
                label: float(row["delta_direct_minus_jepa"]["mean"])
                for _value, label, row in metric_ranked
            }
        selected_horizons[horizon] = combined_row
    return {
        "policy": "strongest_direct_challenger_per_horizon_and_gate_metric",
        "challengers": [label for label, _data in challengers],
        "horizons": selected_horizons,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Combined Direct State Challenge",
        "",
        "The gate receives the largest paired direct advantage independently for state skill and executable-path IC.",
        "",
        "| Horizon | State challenger | State delta | Path challenger | Path delta | Top300 challenger | Top300 delta |",
        "|---:|---|---:|---|---:|---|---:|",
    ]
    for horizon, row in result["horizons"].items():
        delta = row["state_skill"]["delta_direct_minus_jepa"]["mean"]
        path_delta = row["entry_path_ic"]["delta_direct_minus_jepa"]["mean"]
        top300_delta = row["entry_path_ic_top300"]["delta_direct_minus_jepa"]["mean"]
        lines.append(
            f"| {horizon} | {row['selected_direct_challenger']} | {delta:+.6f} | "
            f"{row['selected_direct_challenger_for_entry_path_ic']} | {path_delta:+.6f} | "
            f"{row['selected_direct_challenger_for_entry_path_ic_top300']} | "
            f"{top300_delta:+.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose the strongest direct state challenger per horizon."
    )
    parser.add_argument("--challenger", action="append", type=parse_challenger, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    loaded = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.challenger
    ]
    result = combine(loaded)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
