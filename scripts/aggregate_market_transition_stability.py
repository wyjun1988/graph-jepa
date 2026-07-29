from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.compare_market_transition_heads import historical_gate
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
)


def finite_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def selected_values(gate: Mapping[str, object]) -> dict[str, float]:
    values = gate["values"]
    family = values["family_auc"]
    return {
        "major_trajectory_auc": float(values["major_trajectory_auc"]),
        "weighted_systemic_auc": float(values["weighted_systemic_auc"]),
        "price_auc": float(family["price_co_movement"]),
        "activity_auc": float(family["market_activity"]),
        "node_state_auc": float(family["node_state"]),
        "topology_auc": float(family["topology"]),
        "broad_selloff_auc": float(values["weighted_broad_selloff_auc"]),
        "major_impact_mass_recall": float(
            values["major_systemic_impact_mass_recall"]
        ),
        "major_impact_mass_lift": float(
            values["major_systemic_impact_mass_lift"]
        ),
        "peak_horizon_accuracy": float(values["peak_horizon_accuracy"]),
    }


def aggregate_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metrics = sorted(records[0]["values"]) if records else []
    failures = Counter(
        failure for record in records for failure in record["failures"]
    )
    return {
        "runs": len(records),
        "passed_runs": int(sum(bool(record["passed"]) for record in records)),
        "all_runs_passed": bool(records) and all(
            bool(record["passed"]) for record in records
        ),
        "failure_counts": dict(sorted(failures.items())),
        "metrics": {
            name: finite_summary(
                [float(record["values"][name]) for record in records]
            )
            for name in metrics
        },
    }


def render_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# Market Transition Stability",
        "",
        f"- Target: `{payload['target_version']}`",
        f"- Seeds: `{payload['seeds']}`",
        f"- All seed/fold runs passed: `{str(payload['all_runs_passed']).lower()}`",
        "- Live orders allowed: `false`",
        "",
        "| Fold | Metric | Mean | Std | Worst |",
        "|---|---|---:|---:|---:|",
    ]
    for fold, aggregate in payload["by_fold"].items():
        for metric in (
            "major_trajectory_auc",
            "activity_auc",
            "broad_selloff_auc",
            "node_state_auc",
            "major_impact_mass_recall",
            "peak_horizon_accuracy",
        ):
            values = aggregate["metrics"][metric]
            lines.append(
                f"| {fold} | {metric} | {values['mean']:.3f} | "
                f"{values['std']:.3f} | {values['minimum']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate fixed-architecture market-transition seed stability."
    )
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", default="2701,4301,7301")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique seeds are required")
    records = []
    for seed in seeds:
        for fold in ("fold1", "fold2"):
            root = Path(f"{args.run_prefix}_seed{seed}_20260714") / fold
            summary = json.loads((root / "summary.json").read_text())
            major = json.loads(
                (root / "major_trajectory" / "summary.json").read_text()
            )
            if summary.get("target_version") != MARKET_TRANSITION_TARGET_VERSION:
                raise ValueError("stability target version does not match")
            if (
                summary.get("impact_metric_version")
                != MARKET_TRANSITION_IMPACT_METRIC_VERSION
            ):
                raise ValueError("stability impact metric version does not match")
            gate = historical_gate(summary, major)
            records.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "passed": bool(gate["passed"]),
                    "failures": list(gate["failures"]),
                    "values": selected_values(gate),
                }
            )
    by_fold = {
        fold: aggregate_records(
            [record for record in records if record["fold"] == fold]
        )
        for fold in ("fold1", "fold2")
    }
    payload = {
        "status": "complete",
        "role": "fixed_architecture_seed_stability",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "run_prefix": args.run_prefix,
        "seeds": seeds,
        "records": records,
        "by_fold": by_fold,
        "overall": aggregate_records(records),
        "all_runs_passed": all(bool(record["passed"]) for record in records),
        "test_used_for_within_run_selection": False,
        "development_fold_reuse_warning": True,
        "decision": "research_only_requires_new_confirmation_period",
        "live_orders_allowed": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "all_runs_passed": payload["all_runs_passed"],
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
