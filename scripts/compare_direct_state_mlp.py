from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean


METRIC_COLUMNS = {
    "state_skill": (
        "all_state_skill_vs_persistence",
        "mse_skill_vs_persistence",
    ),
    "target_corr": ("all_state_target_corr", "target_corr"),
    "delta_corr": ("all_state_delta_corr", "delta_corr"),
    "target_sign_accuracy": (
        "all_state_target_sign_accuracy_abs_ge_0_10",
        "target_sign_accuracy_abs_ge_0_10",
    ),
    "delta_sign_accuracy": (
        "all_state_delta_sign_accuracy_abs_ge_0_10",
        "delta_sign_accuracy_abs_ge_0_10",
    ),
    "entry_path_ic": ("return_path_ic", "realized_entry_path_ic"),
    "entry_path_ic_top300": (
        "return_path_ic_top300",
        "realized_entry_path_ic_top300",
    ),
}
REQUIRED_HORIZONS = (1, 2, 3, 5, 10)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compare_metric(
    direct: pd.DataFrame,
    jepa: pd.DataFrame,
    horizon: int,
    direct_column: str,
    jepa_column: str,
) -> dict[str, object]:
    left = direct.loc[
        direct["horizon"] == int(horizon), ["date", direct_column]
    ].rename(columns={direct_column: "direct"})
    right = jepa.loc[
        jepa["horizon"] == int(horizon), ["date", jepa_column]
    ].rename(columns={jepa_column: "jepa"})
    if left["date"].duplicated().any() or right["date"].duplicated().any():
        raise ValueError(f"duplicate daily row at horizon {horizon}")
    direct_dates = set(left["date"].astype(str))
    jepa_dates = set(right["date"].astype(str))
    if direct_dates != jepa_dates or not direct_dates:
        raise ValueError(
            f"direct and JEPA dates differ at horizon {horizon}: "
            f"direct={len(direct_dates)} jepa={len(jepa_dates)}"
        )
    merged = left.merge(
        right, on="date", how="inner", validate="one_to_one"
    ).sort_values("date")
    delta = (
        merged["direct"].to_numpy(dtype=np.float64)
        - merged["jepa"].to_numpy(dtype=np.float64)
    )
    return {
        "direct": newey_west_mean(merged["direct"].to_numpy(), lag=int(horizon)),
        "jepa": newey_west_mean(merged["jepa"].to_numpy(), lag=int(horizon)),
        "delta_direct_minus_jepa": newey_west_mean(delta, lag=int(horizon)),
    }


def compare_frames(
    direct: pd.DataFrame,
    jepa: pd.DataFrame,
    required_horizons: tuple[int, ...] | None = None,
    required_state_target_scope: str | None = None,
) -> dict[str, object]:
    state_target_feature_count = None
    if required_state_target_scope is not None:
        required_columns = {"state_target_scope", "state_target_feature_count"}
        if not required_columns.issubset(direct.columns) or not required_columns.issubset(
            jepa.columns
        ):
            raise ValueError(
                "required state target scope metadata is missing from comparison inputs"
            )
        direct_scopes = set(direct["state_target_scope"].astype(str))
        jepa_scopes = set(jepa["state_target_scope"].astype(str))
        expected = {str(required_state_target_scope)}
        if direct_scopes != expected or jepa_scopes != expected:
            raise ValueError(
                "direct and JEPA state target scopes do not match the required scope"
            )
        direct_counts = set(
            pd.to_numeric(direct["state_target_feature_count"], errors="raise").astype(int)
        )
        jepa_counts = set(
            pd.to_numeric(jepa["state_target_feature_count"], errors="raise").astype(int)
        )
        if len(direct_counts) != 1 or direct_counts != jepa_counts:
            raise ValueError(
                "direct and JEPA state target feature counts do not match"
            )
        state_target_feature_count = next(iter(direct_counts))
    direct_horizons = set(
        pd.to_numeric(direct["horizon"], errors="raise").astype(int)
    )
    jepa_horizons = set(pd.to_numeric(jepa["horizon"], errors="raise").astype(int))
    if direct_horizons != jepa_horizons:
        raise ValueError("direct and JEPA horizon sets differ")
    if required_horizons is not None and direct_horizons != set(required_horizons):
        raise ValueError(
            f"comparison requires horizons {list(required_horizons)}, "
            f"found {sorted(direct_horizons)}"
        )
    horizons = sorted(direct_horizons)
    return {
        "state_target_scope": required_state_target_scope or "unverified_legacy",
        "state_target_feature_count": state_target_feature_count,
        "horizons": {
            str(horizon): {
                name: compare_metric(
                    direct,
                    jepa,
                    horizon,
                    direct_column,
                    jepa_column,
                )
                for name, (direct_column, jepa_column) in METRIC_COLUMNS.items()
            }
            for horizon in horizons
        }
    }


def render_markdown(comparison: dict[str, object]) -> str:
    temporal_scope = comparison.get("state_target_scope") == "checkpoint_temporal"
    title = (
        "# Direct Temporal-Target MLP Versus Graph-JEPA"
        if temporal_scope
        else "# Direct All-State MLP Versus Graph-JEPA"
    )
    scope_note = (
        "Paired daily comparison on the checkpoint's identical non-zero temporal "
        "training targets, dates, stocks, observed cells, and horizons."
        if temporal_scope
        else "Paired daily comparison on identical dates, stocks, observed cells, and horizons."
    )
    lines = [
        title,
        "",
        scope_note,
        "Delta is direct residual MLP minus Graph-JEPA; t is Newey-West on daily deltas.",
        "",
        "| H | Direct skill | JEPA skill | Delta | t | Direct target corr | JEPA | Delta | t | Direct delta corr | JEPA | Delta | t |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, metrics in comparison["horizons"].items():
        skill = metrics["state_skill"]
        target = metrics["target_corr"]
        delta = metrics["delta_corr"]
        lines.append(
            "| {h} | {sd:+.4f} | {sj:+.4f} | {sx:+.4f} | {st:+.2f} | "
            "{td:+.4f} | {tj:+.4f} | {tx:+.4f} | {tt:+.2f} | "
            "{dd:+.4f} | {dj:+.4f} | {dx:+.4f} | {dt:+.2f} |".format(
                h=horizon,
                sd=skill["direct"]["mean"],
                sj=skill["jepa"]["mean"],
                sx=skill["delta_direct_minus_jepa"]["mean"],
                st=skill["delta_direct_minus_jepa"]["newey_west_t"],
                td=target["direct"]["mean"],
                tj=target["jepa"]["mean"],
                tx=target["delta_direct_minus_jepa"]["mean"],
                tt=target["delta_direct_minus_jepa"]["newey_west_t"],
                dd=delta["direct"]["mean"],
                dj=delta["jepa"]["mean"],
                dx=delta["delta_direct_minus_jepa"]["mean"],
                dt=delta["delta_direct_minus_jepa"]["newey_west_t"],
            )
        )
    lines.extend(
        [
            "",
            "| H | Direct target sign | JEPA | Delta | t | Direct delta sign | JEPA | Delta | t |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, metrics in comparison["horizons"].items():
        target = metrics["target_sign_accuracy"]
        delta = metrics["delta_sign_accuracy"]
        lines.append(
            "| {h} | {td:.4f} | {tj:.4f} | {tx:+.4f} | {tt:+.2f} | "
            "{dd:.4f} | {dj:.4f} | {dx:+.4f} | {dt:+.2f} |".format(
                h=horizon,
                td=target["direct"]["mean"],
                tj=target["jepa"]["mean"],
                tx=target["delta_direct_minus_jepa"]["mean"],
                tt=target["delta_direct_minus_jepa"]["newey_west_t"],
                dd=delta["direct"]["mean"],
                dj=delta["jepa"]["mean"],
                dx=delta["delta_direct_minus_jepa"]["mean"],
                dt=delta["delta_direct_minus_jepa"]["newey_west_t"],
            )
        )
    lines.extend(
        [
            "",
            "| H | Direct entry-path IC | JEPA | Delta | t | Direct top300 IC | JEPA | Delta | t |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon, metrics in comparison["horizons"].items():
        path = metrics["entry_path_ic"]
        top300 = metrics["entry_path_ic_top300"]
        lines.append(
            "| {h} | {pd:+.4f} | {pj:+.4f} | {px:+.4f} | {pt:+.2f} | "
            "{td:+.4f} | {tj:+.4f} | {tx:+.4f} | {tt:+.2f} |".format(
                h=horizon,
                pd=path["direct"]["mean"],
                pj=path["jepa"]["mean"],
                px=path["delta_direct_minus_jepa"]["mean"],
                pt=path["delta_direct_minus_jepa"]["newey_west_t"],
                td=top300["direct"]["mean"],
                tj=top300["jepa"]["mean"],
                tx=top300["delta_direct_minus_jepa"]["mean"],
                tt=top300["delta_direct_minus_jepa"]["newey_west_t"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired all-state comparison of direct residual MLP and Graph-JEPA."
    )
    parser.add_argument("--direct-daily", required=True)
    parser.add_argument("--jepa-daily", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--required-state-target-scope",
        choices=["all", "checkpoint_temporal"],
        default=None,
    )
    args = parser.parse_args()

    direct = pd.read_csv(args.direct_daily)
    jepa = pd.read_csv(args.jepa_daily)
    comparison = compare_frames(
        direct,
        jepa,
        REQUIRED_HORIZONS,
        required_state_target_scope=args.required_state_target_scope,
    )
    comparison.update(
        {
            "role": "research_only_paired_direct_state_comparison",
            "test_used_for_selection": False,
            "live_orders_allowed": False,
            "direct_daily": args.direct_daily,
            "direct_daily_sha256": sha256_file(Path(args.direct_daily)),
            "jepa_daily": args.jepa_daily,
            "jepa_daily_sha256": sha256_file(Path(args.jepa_daily)),
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        render_markdown(comparison), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
