from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import _correlation, _decile_spread, newey_west_mean


def top_liquidity_mask(frame: pd.DataFrame, size: int = 300) -> np.ndarray:
    liquidity = pd.to_numeric(frame["current_value_ma20_log"], errors="coerce").to_numpy()
    finite = np.flatnonzero(np.isfinite(liquidity))
    selected = finite[np.argsort(liquidity[finite], kind="stable")[-min(size, len(finite)) :]]
    mask = np.zeros(len(frame), dtype=bool)
    mask[selected] = True
    return mask


def reference_daily_rows(forecast_path: Path) -> pd.DataFrame:
    columns = [
        "date",
        "horizon",
        "prediction_return_1d",
        "target_return_1d",
        "current_return_1d",
        "realized_path_return",
        "current_value_ma20_log",
    ]
    frame = pd.read_csv(forecast_path, usecols=columns)
    rows: list[dict[str, float | int | str]] = []
    score_specs = {
        "jepa": ("prediction_return_1d", 1.0),
        "current_return": ("current_return_1d", 1.0),
        "one_day_reversal": ("current_return_1d", -1.0),
    }
    for (date, horizon), group in frame.groupby(["date", "horizon"], sort=True):
        state = pd.to_numeric(group["target_return_1d"], errors="coerce").to_numpy()
        path = pd.to_numeric(group["realized_path_return"], errors="coerce").to_numpy()
        liquid = top_liquidity_mask(group)
        for name, (column, direction) in score_specs.items():
            score = direction * pd.to_numeric(group[column], errors="coerce").to_numpy()
            rows.append(
                {
                    "model": name,
                    "horizon": int(horizon),
                    "date": str(date),
                    "state_ic": _correlation(score, state),
                    "state_ic_top300": _correlation(score[liquid], state[liquid]),
                    "path_ic": _correlation(score, path),
                    "path_ic_top300": _correlation(score[liquid], path[liquid]),
                    "next_return_decile_spread": _decile_spread(score, state),
                    "path_decile_spread": _decile_spread(score, path),
                }
            )
    return pd.DataFrame(rows)


def compare_metric(
    baseline: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str,
    horizon: int,
) -> dict[str, object]:
    left = baseline.loc[baseline["horizon"] == horizon, ["date", metric]].rename(
        columns={metric: "baseline"}
    )
    right = reference.loc[
        (reference["model"] == "jepa") & (reference["horizon"] == horizon),
        ["date", metric],
    ].rename(columns={metric: "jepa"})
    merged = left.merge(right, on="date", how="inner").sort_values("date")
    delta = merged["baseline"].to_numpy(dtype=np.float64) - merged["jepa"].to_numpy(dtype=np.float64)
    return {
        "baseline": newey_west_mean(merged["baseline"].to_numpy(), lag=horizon),
        "jepa": newey_west_mean(merged["jepa"].to_numpy(), lag=horizon),
        "delta_baseline_minus_jepa": newey_west_mean(delta, lag=horizon),
    }


def render_markdown(comparison: dict[str, object]) -> str:
    lines = [
        "# Direct Baseline Comparison",
        "",
        "Paired daily comparison on identical dates and stocks. Delta is direct baseline minus JEPA.",
        "",
        "| Model | H | State IC | JEPA IC | Delta | Delta NW t | Path IC | JEPA path IC | Path delta | Path delta NW t |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    models = comparison["models"]
    for model_name in sorted(models):
        for horizon in sorted(models[model_name], key=int):
            metrics = models[model_name][horizon]
            state = metrics["state_ic"]
            path = metrics["path_ic"]
            lines.append(
                "| {model} | {h} | {si:+.4f} | {ji:+.4f} | {sd:+.4f} | {st:+.2f} | "
                "{pi:+.4f} | {jpi:+.4f} | {pd:+.4f} | {pt:+.2f} |".format(
                    model=model_name,
                    h=horizon,
                    si=state["baseline"]["mean"],
                    ji=state["jepa"]["mean"],
                    sd=state["delta_baseline_minus_jepa"]["mean"],
                    st=state["delta_baseline_minus_jepa"]["newey_west_t"],
                    pi=path["baseline"]["mean"],
                    jpi=path["jepa"]["mean"],
                    pd=path["delta_baseline_minus_jepa"]["mean"],
                    pt=path["delta_baseline_minus_jepa"]["newey_west_t"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct baselines with paired JEPA forecasts.")
    parser.add_argument("--baseline-daily", required=True)
    parser.add_argument("--jepa-forecasts", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_daily)
    reference = reference_daily_rows(Path(args.jepa_forecasts))
    metrics = [
        "state_ic",
        "state_ic_top300",
        "path_ic",
        "path_ic_top300",
        "next_return_decile_spread",
        "path_decile_spread",
    ]
    result: dict[str, object] = {
        "baseline_daily": args.baseline_daily,
        "jepa_forecasts": args.jepa_forecasts,
        "models": {},
        "simple_references": {},
    }
    for model_name, model_frame in baseline.groupby("model", sort=True):
        model_result: dict[str, object] = {}
        for horizon in sorted(model_frame["horizon"].unique()):
            horizon_result = {
                metric: compare_metric(model_frame, reference, metric, int(horizon))
                for metric in metrics
            }
            model_result[str(int(horizon))] = horizon_result
        result["models"][str(model_name)] = model_result
    for model_name, model_frame in reference.groupby("model", sort=True):
        result["simple_references"][str(model_name)] = {
            str(int(horizon)): {
                metric: newey_west_mean(
                    horizon_frame[metric].to_numpy(dtype=np.float64), lag=int(horizon)
                )
                for metric in metrics
            }
            for horizon, horizon_frame in model_frame.groupby("horizon", sort=True)
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(render_markdown(result), encoding="utf-8")
    reference.to_csv(output_dir / "reference_daily_metrics.csv", index=False)
    print(f"wrote {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
