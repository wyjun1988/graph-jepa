from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean


HORIZONS = (1, 2, 3, 5, 10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_summary(left: Sequence[float], right: Sequence[float], lag: int) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if not valid.any():
        raise ValueError("paired comparison contains no finite rows")
    return {
        "left": newey_west_mean(left_array[valid], lag=int(lag)),
        "right": newey_west_mean(right_array[valid], lag=int(lag)),
        "delta_left_minus_right": newey_west_mean(
            left_array[valid] - right_array[valid], lag=int(lag)
        ),
    }


def normalize_inputs(
    qlib: pd.DataFrame,
    raw_jepa: pd.DataFrame,
    latent_head: pd.DataFrame,
) -> pd.DataFrame:
    required_qlib = {
        "date",
        "horizon",
        "split",
        "return_path_ic",
        "return_path_ic_top300",
    }
    required_raw = {
        "date",
        "horizon",
        "realized_entry_path_ic",
        "realized_entry_path_ic_top300",
    }
    required_head = {"date", "horizon", "entry_path_ic", "entry_path_ic_top300"}
    for label, frame, required in (
        ("qlib", qlib, required_qlib),
        ("raw_jepa", raw_jepa, required_raw),
        ("latent_head", latent_head, required_head),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")

    qlib_test = qlib.loc[qlib["split"] == "test", list(required_qlib)].copy()
    qlib_test = qlib_test.rename(
        columns={
            "return_path_ic": "qlib_all",
            "return_path_ic_top300": "qlib_top300",
        }
    ).drop(columns="split")
    raw = raw_jepa.loc[
        :,
        [
            "date",
            "horizon",
            "realized_entry_path_ic",
            "realized_entry_path_ic_top300",
        ],
    ].rename(
        columns={
            "realized_entry_path_ic": "raw_jepa_all",
            "realized_entry_path_ic_top300": "raw_jepa_top300",
        }
    )
    head = latent_head.loc[
        :, ["date", "horizon", "entry_path_ic", "entry_path_ic_top300"]
    ].rename(
        columns={
            "entry_path_ic": "latent_head_all",
            "entry_path_ic_top300": "latent_head_top300",
        }
    )
    for label, frame in (("qlib", qlib_test), ("raw_jepa", raw), ("latent_head", head)):
        frame["date"] = frame["date"].astype(str)
        frame["horizon"] = pd.to_numeric(frame["horizon"], errors="raise").astype(int)
        if frame.duplicated(["date", "horizon"]).any():
            raise ValueError(f"{label} contains duplicate date/horizon rows")
        if set(frame["horizon"]) != set(HORIZONS):
            raise ValueError(f"{label} horizon set does not match {list(HORIZONS)}")

    merged = qlib_test.merge(
        raw, on=["date", "horizon"], how="inner", validate="one_to_one"
    ).merge(head, on=["date", "horizon"], how="inner", validate="one_to_one")
    expected_rows = len(HORIZONS) * 194
    if len(merged) != expected_rows:
        raise ValueError(f"expected {expected_rows} aligned rows, found {len(merged)}")
    if any(len(merged.loc[merged["horizon"] == horizon]) != 194 for horizon in HORIZONS):
        raise ValueError("each horizon must contain exactly 194 aligned dates")
    return merged.sort_values(["horizon", "date"]).reset_index(drop=True)


def robustness(frame: pd.DataFrame, lag: int) -> dict[str, Any]:
    ordered = frame.sort_values("date").reset_index(drop=True).copy()
    delta = ordered["qlib_top300"] - ordered["raw_jepa_top300"]
    midpoint = len(ordered) // 2
    monthly = ordered.assign(month=ordered["date"].str[:7]).groupby("month", sort=True)
    month_rows = []
    for month, group in monthly:
        month_rows.append(
            {
                "month": str(month),
                "rows": int(len(group)),
                "qlib_mean": float(group["qlib_top300"].mean()),
                "raw_jepa_mean": float(group["raw_jepa_top300"].mean()),
                "latent_head_mean": float(group["latent_head_top300"].mean()),
                "qlib_minus_raw_jepa_mean": float(
                    (group["qlib_top300"] - group["raw_jepa_top300"]).mean()
                ),
            }
        )
    leave_one_month_out = []
    month_values = ordered["date"].str[:7]
    for month in sorted(month_values.unique()):
        selected = delta.loc[month_values != month]
        leave_one_month_out.append(
            {"excluded_month": month, **newey_west_mean(selected.to_numpy(), lag=lag)}
        )
    winsorized = {}
    for quantile in (0.01, 0.05):
        lower, upper = delta.quantile([quantile, 1.0 - quantile])
        winsorized[f"q{int(quantile * 100):02d}"] = newey_west_mean(
            delta.clip(lower=lower, upper=upper).to_numpy(), lag=lag
        )
    largest_days = ordered.assign(delta=delta).loc[
        :, ["date", "qlib_top300", "raw_jepa_top300", "latent_head_top300", "delta"]
    ]
    largest_days = largest_days.iloc[
        np.argsort(np.abs(largest_days["delta"].to_numpy()))[::-1][:10]
    ]
    return {
        "full": newey_west_mean(delta.to_numpy(), lag=lag),
        "first_half": newey_west_mean(delta.iloc[:midpoint].to_numpy(), lag=lag),
        "second_half": newey_west_mean(delta.iloc[midpoint:].to_numpy(), lag=lag),
        "winsorized": winsorized,
        "monthly": month_rows,
        "positive_month_fraction": float(
            np.mean([row["qlib_minus_raw_jepa_mean"] > 0.0 for row in month_rows])
        ),
        "leave_one_month_out": leave_one_month_out,
        "leave_one_month_out_min_t": float(
            min(row["newey_west_t"] for row in leave_one_month_out)
        ),
        "leave_one_month_out_max_t": float(
            max(row["newey_west_t"] for row in leave_one_month_out)
        ),
        "largest_absolute_delta_days": largest_days.to_dict(orient="records"),
        "daily_ic_correlations": ordered[
            ["qlib_top300", "raw_jepa_top300", "latent_head_top300"]
        ].corr().to_dict(),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Fold4 Qlib Versus JEPA Gap Audit",
        "",
        "- Scope: retrospective diagnostic; not unbiased promotion evidence.",
        "- Live orders allowed: `false`.",
        "- Metric: paired daily entry-path IC on the top 300 liquid stocks.",
        "",
        "| H | Qlib | Raw JEPA | Qlib-Raw | t | Latent head | Qlib-Head | t |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        row = result["horizons"][str(horizon)]["top300"]
        raw = row["qlib_vs_raw_jepa"]
        head = row["qlib_vs_latent_head"]
        lines.append(
            "| {h} | {q:+.4f} | {r:+.4f} | {d:+.4f} | {t:+.2f} | "
            "{j:+.4f} | {hd:+.4f} | {ht:+.2f} |".format(
                h=horizon,
                q=raw["left"]["mean"],
                r=raw["right"]["mean"],
                d=raw["delta_left_minus_right"]["mean"],
                t=raw["delta_left_minus_right"]["newey_west_t"],
                j=head["right"]["mean"],
                hd=head["delta_left_minus_right"]["mean"],
                ht=head["delta_left_minus_right"]["newey_west_t"],
            )
        )
    robust = result["h5_top300_robustness"]
    lines.extend(
        [
            "",
            "## H5 Robustness",
            "",
            f"- First-half Qlib-Raw: `{robust['first_half']['mean']:+.4f}` "
            f"(t=`{robust['first_half']['newey_west_t']:+.2f}`).",
            f"- Second-half Qlib-Raw: `{robust['second_half']['mean']:+.4f}` "
            f"(t=`{robust['second_half']['newey_west_t']:+.2f}`).",
            f"- Positive month fraction: `{robust['positive_month_fraction']:.3f}`.",
            f"- Leave-one-month-out t range: `{robust['leave_one_month_out_min_t']:+.2f}` "
            f"to `{robust['leave_one_month_out_max_t']:+.2f}`.",
            "",
            "Top Qlib h5 gain features are recorded in `audit.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the fold4 Qlib versus JEPA path gap.")
    parser.add_argument("--qlib-daily", required=True)
    parser.add_argument("--raw-jepa-daily", required=True)
    parser.add_argument("--latent-head-daily", required=True)
    parser.add_argument("--qlib-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = {
        "qlib_daily": Path(args.qlib_daily),
        "raw_jepa_daily": Path(args.raw_jepa_daily),
        "latent_head_daily": Path(args.latent_head_daily),
        "qlib_summary": Path(args.qlib_summary),
    }
    merged = normalize_inputs(
        pd.read_csv(paths["qlib_daily"]),
        pd.read_csv(paths["raw_jepa_daily"]),
        pd.read_csv(paths["latent_head_daily"]),
    )
    qlib_summary = json.loads(paths["qlib_summary"].read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "role": "retrospective_test_informed_fold4_qlib_gap_diagnostic",
        "test_used_for_hypothesis_generation": True,
        "eligible_as_unbiased_promotion_evidence": False,
        "live_orders_allowed": False,
        "rows_per_horizon": 194,
        "horizons": {},
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in paths.items()
        },
        "qlib_h5_top_features_by_gain": qlib_summary["horizons"]["5"][
            "top_features_by_gain"
        ],
    }
    for horizon in HORIZONS:
        frame = merged.loc[merged["horizon"] == horizon]
        result["horizons"][str(horizon)] = {}
        for suffix in ("all", "top300"):
            result["horizons"][str(horizon)][suffix] = {
                "qlib_vs_raw_jepa": paired_summary(
                    frame[f"qlib_{suffix}"], frame[f"raw_jepa_{suffix}"], lag=horizon
                ),
                "qlib_vs_latent_head": paired_summary(
                    frame[f"qlib_{suffix}"], frame[f"latent_head_{suffix}"], lag=horizon
                ),
                "raw_jepa_vs_latent_head": paired_summary(
                    frame[f"raw_jepa_{suffix}"],
                    frame[f"latent_head_{suffix}"],
                    lag=horizon,
                ),
            }
    result["h5_top300_robustness"] = robustness(
        merged.loc[merged["horizon"] == 5], lag=5
    )
    h5_delta = result["horizons"]["5"]["top300"]["qlib_vs_raw_jepa"][
        "delta_left_minus_right"
    ]
    result["current_gate"] = {
        "rule": "blocked when Qlib-Raw JEPA mean is positive and Newey-West t >= 1.96",
        "blocked": bool(h5_delta["mean"] > 0.0 and h5_delta["newey_west_t"] >= 1.96),
        "promotion_eligible_from_this_audit_alone": False,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "audit.md").write_text(render_markdown(result), encoding="utf-8")
    merged.to_csv(output_dir / "aligned_daily.csv", index=False)
    (output_dir / "AUDIT_COMPLETE").touch()
    print(json.dumps(result["current_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
