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


REQUIRED_HORIZONS = (1, 2, 3, 5, 10)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(values: np.ndarray, lag: int) -> dict[str, float | int]:
    return newey_west_mean(np.asarray(values, dtype=np.float64), lag=int(lag))


def compare_transition_energy_strata(
    direct: pd.DataFrame,
    jepa: pd.DataFrame,
    *,
    impact_quantile: float = 0.75,
    required_horizons: tuple[int, ...] | None = REQUIRED_HORIZONS,
    gate_horizons: tuple[int, ...] = (2, 3),
) -> dict[str, object]:
    if not 0.0 < impact_quantile < 1.0:
        raise ValueError("impact_quantile must be between 0 and 1")
    direct_required = {
        "date",
        "horizon",
        "all_state_skill_vs_persistence",
    }
    jepa_required = {
        "date",
        "horizon",
        "mse_skill_vs_persistence",
        "persistence_sse",
        "observed_cells",
    }
    if missing := sorted(direct_required - set(direct.columns)):
        raise ValueError(f"direct daily metrics missing columns: {missing}")
    if missing := sorted(jepa_required - set(jepa.columns)):
        raise ValueError(f"JEPA daily metrics missing columns: {missing}")

    direct = direct.copy()
    jepa = jepa.copy()
    direct["horizon"] = pd.to_numeric(direct["horizon"], errors="raise").astype(int)
    jepa["horizon"] = pd.to_numeric(jepa["horizon"], errors="raise").astype(int)
    direct["date"] = direct["date"].astype(str)
    jepa["date"] = jepa["date"].astype(str)
    horizons = tuple(sorted(set(direct["horizon"])))
    if set(horizons) != set(jepa["horizon"]):
        raise ValueError("direct and JEPA horizon sets differ")
    if required_horizons is not None and set(horizons) != set(required_horizons):
        raise ValueError(
            f"comparison requires horizons {list(required_horizons)}, "
            f"found {list(horizons)}"
        )

    results: dict[str, object] = {}
    for horizon in horizons:
        direct_h = direct.loc[
            direct["horizon"] == horizon,
            ["date", "all_state_skill_vs_persistence"],
        ].rename(columns={"all_state_skill_vs_persistence": "direct_skill"})
        jepa_h = jepa.loc[
            jepa["horizon"] == horizon,
            [
                "date",
                "mse_skill_vs_persistence",
                "persistence_sse",
                "observed_cells",
            ],
        ].rename(columns={"mse_skill_vs_persistence": "jepa_skill"})
        if direct_h["date"].duplicated().any() or jepa_h["date"].duplicated().any():
            raise ValueError(f"duplicate daily row at horizon {horizon}")
        if set(direct_h["date"]) != set(jepa_h["date"]) or direct_h.empty:
            raise ValueError(f"direct and JEPA dates differ at horizon {horizon}")
        merged = direct_h.merge(
            jepa_h,
            on="date",
            how="inner",
            validate="one_to_one",
        ).sort_values("date")
        observed_cells = pd.to_numeric(
            merged["observed_cells"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        persistence_sse = pd.to_numeric(
            merged["persistence_sse"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if (
            not np.isfinite(observed_cells).all()
            or not np.isfinite(persistence_sse).all()
            or (observed_cells <= 0.0).any()
            or (persistence_sse < 0.0).any()
        ):
            raise ValueError(f"invalid transition energy inputs at horizon {horizon}")
        transition_energy = persistence_sse / observed_cells
        threshold = float(np.quantile(transition_energy, impact_quantile))
        top_mask = transition_energy >= threshold
        if not top_mask.any() or top_mask.all():
            raise ValueError(f"transition energy split is degenerate at horizon {horizon}")
        direct_skill = pd.to_numeric(
            merged["direct_skill"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        jepa_skill = pd.to_numeric(
            merged["jepa_skill"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(direct_skill).all() or not np.isfinite(jepa_skill).all():
            raise ValueError(f"non-finite state skill at horizon {horizon}")

        strata = {
            "all": np.ones(len(merged), dtype=bool),
            "top_impact": top_mask,
            "lower_impact": ~top_mask,
        }
        result_strata: dict[str, object] = {}
        for name, mask in strata.items():
            delta = jepa_skill[mask] - direct_skill[mask]
            result_strata[name] = {
                "rows": int(mask.sum()),
                "direct_skill": _metric(direct_skill[mask], horizon),
                "jepa_skill": _metric(jepa_skill[mask], horizon),
                "delta_jepa_minus_direct": _metric(delta, horizon),
                "transition_energy_mean": float(transition_energy[mask].mean()),
            }
        results[str(horizon)] = {
            "impact_quantile": float(impact_quantile),
            "transition_energy_threshold": threshold,
            "strata": result_strata,
        }

    gate_rows = {}
    for horizon in gate_horizons:
        if str(horizon) not in results:
            raise ValueError(f"gate horizon {horizon} is missing")
        horizon_result = results[str(horizon)]
        all_margin = horizon_result["strata"]["all"][
            "delta_jepa_minus_direct"
        ]["mean"]
        top_margin = horizon_result["strata"]["top_impact"][
            "delta_jepa_minus_direct"
        ]["mean"]
        gate_rows[str(horizon)] = {
            "all_margin_jepa_minus_direct": float(all_margin),
            "top_impact_margin_jepa_minus_direct": float(top_margin),
            "passed": bool(all_margin >= 0.0 and top_margin >= 0.0),
        }
    return {
        "metric": "state_skill_vs_persistence",
        "transition_energy": "persistence_sse / observed_cells",
        "impact_quantile": float(impact_quantile),
        "horizons": results,
        "predeclared_gate": {
            "horizons": list(gate_horizons),
            "rule": "JEPA-minus-direct state-skill margin >= 0 for all and top-impact rows",
            "by_horizon": gate_rows,
            "passed": all(row["passed"] for row in gate_rows.values()),
        },
        "live_orders_allowed": False,
    }


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# State Skill by Transition Energy",
        "",
        "Delta is Graph-JEPA minus direct residual MLP. Top impact is selected "
        "within each horizon using target transition energy.",
        "",
        "| H | Stratum | Rows | Direct | JEPA | Delta | NW t | Energy |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, horizon_result in report["horizons"].items():
        for name in ("all", "top_impact", "lower_impact"):
            row = horizon_result["strata"][name]
            lines.append(
                "| {h} | {name} | {rows} | {direct:+.4f} | {jepa:+.4f} | "
                "{delta:+.4f} | {t:+.2f} | {energy:.6f} |".format(
                    h=horizon,
                    name=name,
                    rows=row["rows"],
                    direct=row["direct_skill"]["mean"],
                    jepa=row["jepa_skill"]["mean"],
                    delta=row["delta_jepa_minus_direct"]["mean"],
                    t=row["delta_jepa_minus_direct"]["newey_west_t"],
                    energy=row["transition_energy_mean"],
                )
            )
    gate = report["predeclared_gate"]
    lines.extend(
        [
            "",
            f"Predeclared h2/h3 gate passed: **{gate['passed']}**.",
            "",
            "This is retrospective, test-informed diagnostic evidence and does "
            "not authorize live orders.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-daily", type=Path, required=True)
    parser.add_argument("--jepa-daily", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--impact-quantile", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_transition_energy_strata(
        pd.read_csv(args.direct_daily),
        pd.read_csv(args.jepa_daily),
        impact_quantile=args.impact_quantile,
    )
    report["inputs"] = {
        "direct_daily": str(args.direct_daily),
        "direct_daily_sha256": sha256_file(args.direct_daily),
        "jepa_daily": str(args.jepa_daily),
        "jepa_daily_sha256": sha256_file(args.jepa_daily),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "comparison.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["predeclared_gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
