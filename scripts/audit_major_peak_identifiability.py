from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stock_v2.market_transition_head import MARKET_FAMILY_TARGETS


def peak_region(horizon: int) -> str:
    if int(horizon) <= 2:
        return "early_1_2"
    if int(horizon) <= 5:
        return "middle_3_5"
    return "late_10"


def build_paths(
    rows: pd.DataFrame,
    calibrations: dict[str, object],
    *,
    event_quantile: float = 0.90,
) -> tuple[pd.DataFrame, float]:
    scored = rows.copy()
    salience = np.zeros(len(scored), dtype=np.float64)
    for horizon, positions in scored.groupby("horizon").groups.items():
        threshold = calibrations[str(int(horizon))]["family_event_threshold"]
        family = np.stack(
            [
                scored.loc[positions, f"family:{name}"].to_numpy(dtype=np.float64)
                / max(float(threshold[name]), 1e-8)
                for name in MARKET_FAMILY_TARGETS
            ],
            axis=1,
        )
        salience[np.asarray(positions, dtype=np.int64)] = np.max(family, axis=1)
    scored["horizon_salience"] = salience
    path_rows = []
    for (split, date), selected in scored.groupby(["split", "date"], sort=True):
        selected = selected.sort_values("horizon")
        values = selected["horizon_salience"].to_numpy(dtype=np.float64)
        horizons = selected["horizon"].to_numpy(dtype=np.int64)
        order = np.argsort(values, kind="stable")[::-1]
        top = float(values[order[0]])
        second = float(values[order[1]]) if len(order) > 1 else float("nan")
        path_rows.append(
            {
                "split": split,
                "date": date,
                "path_salience": top,
                "peak_horizon": int(horizons[order[0]]),
                "peak_region": peak_region(int(horizons[order[0]])),
                "second_horizon": int(horizons[order[1]]) if len(order) > 1 else -1,
                "relative_top2_margin": (
                    float((top - second) / top) if top > 1e-12 else 0.0
                ),
            }
        )
    paths = pd.DataFrame(path_rows)
    fit = paths.loc[paths["split"] == "fit", "path_salience"].to_numpy(
        dtype=np.float64
    )
    if len(fit) < 20:
        raise ValueError("at least twenty fit paths are required")
    event_threshold = float(np.quantile(fit, float(event_quantile)))
    paths["major_event"] = paths["path_salience"] >= event_threshold
    return paths, event_threshold


def summarize(paths: pd.DataFrame, event_threshold: float) -> dict[str, object]:
    fit_major = paths[(paths["split"] == "fit") & paths["major_event"]]
    fit_exact_peak = int(fit_major["peak_horizon"].mode().iloc[0])
    fit_peak_region = str(fit_major["peak_region"].mode().iloc[0])
    output = {}
    for split in ("fit", "validation", "test"):
        selected = paths[paths["split"] == split]
        major = selected[selected["major_event"]]
        exact = major["peak_horizon"].value_counts(normalize=True).sort_index()
        region = major["peak_region"].value_counts(normalize=True).sort_index()
        margin = major["relative_top2_margin"].to_numpy(dtype=np.float64)
        output[split] = {
            "rows": len(selected),
            "major_events": len(major),
            "major_event_rate": float(selected["major_event"].mean()),
            "exact_peak_distribution": {
                str(int(key)): float(value) for key, value in exact.items()
            },
            "exact_oracle_majority_rate": (
                float(exact.max()) if len(exact) else float("nan")
            ),
            "fit_majority_peak": fit_exact_peak,
            "fit_majority_peak_accuracy": (
                float(np.mean(major["peak_horizon"] == fit_exact_peak))
                if len(major)
                else float("nan")
            ),
            "region_distribution": {str(key): float(value) for key, value in region.items()},
            "region_oracle_majority_rate": (
                float(region.max()) if len(region) else float("nan")
            ),
            "fit_majority_region": fit_peak_region,
            "fit_majority_region_accuracy": (
                float(np.mean(major["peak_region"] == fit_peak_region))
                if len(major)
                else float("nan")
            ),
            "relative_top2_margin": {
                "median": float(np.median(margin)) if len(margin) else float("nan"),
                "q25": float(np.quantile(margin, 0.25)) if len(margin) else float("nan"),
                "q75": float(np.quantile(margin, 0.75)) if len(margin) else float("nan"),
            },
            "near_tie_rate": {
                "within_5_percent": float(np.mean(margin <= 0.05)) if len(margin) else float("nan"),
                "within_10_percent": float(np.mean(margin <= 0.10)) if len(margin) else float("nan"),
                "within_20_percent": float(np.mean(margin <= 0.20)) if len(margin) else float("nan"),
            },
        }
    return {
        "fit_major_event_threshold": float(event_threshold),
        "splits": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether exact major-path peak horizons are identifiable."
    )
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--target-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--event-quantile", type=float, default=0.90)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.target_csv)
    target_summary = json.loads(Path(args.target_summary).read_text())
    paths, threshold = build_paths(
        rows,
        target_summary["calibrations"],
        event_quantile=float(args.event_quantile),
    )
    summary = {
        "status": "complete",
        "role": "major_path_peak_identifiability_audit",
        "target_version": target_summary["target_version"],
        "horizons": target_summary["horizons"],
        **summarize(paths, threshold),
        "interpretation_contract": {
            "exact_peak_is_unstable_when_top2_margin_is_small": True,
            "diagnostic_does_not_change_preregistered_gate": True,
            "test_used_for_selection": False,
        },
        "live_orders_allowed": False,
    }
    paths.to_csv(output_dir / "daily_peak_identifiability.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fit_threshold": threshold,
                "test": summary["splits"]["test"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
