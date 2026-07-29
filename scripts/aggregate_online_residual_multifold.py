from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REQUIRED_HORIZONS = (1, 2, 3, 5, 10)
TARGETED_HORIZONS = (2, 3)


def parse_fold(value: str) -> tuple[str, Path]:
    name, separator, raw_path = str(value).partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("folds must use NAME=SUMMARY_JSON")
    return name.strip(), Path(raw_path.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int = 1701, samples: int = 100_000
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap values require at least two finite observations")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "worst_fold": float(array.min()),
        "positive_folds": int(np.sum(array > 0.0)),
    }


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _resolve_project_path(summary_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    current = summary_path.resolve().parent
    for parent in (current, *current.parents):
        candidate = parent / path
        if candidate.exists():
            return candidate
    raise ValueError(f"cannot resolve state metadata path {raw_path}")


def summarize_fold(name: str, summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1 or summary.get("mode") != "transfer":
        raise ValueError(f"{name} is not an online residual transfer report")
    required_false = (
        "live_orders_allowed",
        "target_parameters_selected",
        "action_outputs_fed_back",
    )
    for key in required_false:
        if summary.get(key) is not False:
            raise ValueError(f"{name} violates the {key}=false contract")
    if summary.get("state_updates_use_matured_errors_only") is not True:
        raise ValueError(f"{name} does not use matured residuals only")

    selection = summary.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"{name} lacks transfer selection metrics")
    if tuple(int(value) for value in selection.get("selection_horizons", [])) != (
        TARGETED_HORIZONS
    ):
        raise ValueError(f"{name} uses unexpected targeted horizons")
    raw_horizons = selection.get("horizons")
    if not isinstance(raw_horizons, Mapping) or {
        int(value) for value in raw_horizons
    } != set(REQUIRED_HORIZONS):
        raise ValueError(f"{name} lacks the required rollout horizons")

    horizons = {}
    for horizon in REQUIRED_HORIZONS:
        raw = raw_horizons[str(horizon)]
        if int(raw.get("rows", 0)) < 1:
            raise ValueError(f"{name} h{horizon} has no scored rows")
        horizons[str(horizon)] = {
            "pooled_delta_vs_baseline": _finite(
                raw["pooled_delta_vs_baseline"], f"{name} h{horizon} pooled baseline"
            ),
            "pooled_delta_vs_bias_only": _finite(
                raw["pooled_delta_vs_bias_only"], f"{name} h{horizon} pooled bias"
            ),
            "daily_delta_vs_baseline": _finite(
                raw["daily_delta_vs_baseline"]["mean"],
                f"{name} h{horizon} daily baseline",
            ),
            "top_impact_delta_vs_baseline": _finite(
                raw["top_impact_delta_vs_baseline"]["mean"],
                f"{name} h{horizon} top-impact baseline",
            ),
            "post_cold_start_delta_vs_baseline": _finite(
                raw["post_cold_start_delta_vs_baseline"]["mean"],
                f"{name} h{horizon} post-cold-start baseline",
            ),
        }

    state_path = _resolve_project_path(summary_path, str(summary["state_metadata"]))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("live_orders_allowed") is not False:
        raise ValueError(f"{name} state metadata permits live orders")
    if state.get("action_outputs_fed_back") is not False:
        raise ValueError(f"{name} state metadata feeds back actions")
    if state.get("causal_update") != (
        "at origin t update only with residuals matured by t, then predict"
    ):
        raise ValueError(f"{name} state metadata has an unknown update contract")
    if summary.get("state_artifact_sha256") != state.get("artifact_sha256"):
        raise ValueError(f"{name} state artifact hash differs")
    chain = state.get("source_chain")
    if not isinstance(chain, list) or not chain:
        raise ValueError(f"{name} state metadata lacks a source chain")

    return {
        "fold": name,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "state_metadata": str(state_path),
        "state_metadata_sha256": sha256_file(state_path),
        "source_state_metadata_sha256": str(
            summary["source_state_metadata_sha256"]
        ),
        "checkpoint_sha256": str(summary["target_checkpoint_sha256"]),
        "eval_start": str(chain[-1]["eval_start"]),
        "eval_end": str(chain[-1]["eval_end"]),
        "source_chain_length": len(chain),
        "pooled_floor_passed": bool(selection["pooled_floor_passed"]),
        "targeted_gate_passed": bool(selection["targeted_gate_passed"]),
        "reported_gate_passed": bool(selection["transfer_gate_passed"]),
        "horizons": horizons,
    }


def aggregate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(folds) != 3:
        raise ValueError("online residual transfer requires exactly three held-out folds")
    if len({str(fold["checkpoint_sha256"]) for fold in folds}) != len(folds):
        raise ValueError("online residual folds must use distinct checkpoints")
    starts = [str(fold["eval_start"]) for fold in folds]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError("online residual folds must be chronological and unique")
    for previous, current in zip(folds, folds[1:]):
        if current["source_state_metadata_sha256"] != previous[
            "state_metadata_sha256"
        ]:
            raise ValueError("online residual state chain is discontinuous")
        if int(current["source_chain_length"]) != int(previous["source_chain_length"]) + 1:
            raise ValueError("online residual source chain length is discontinuous")

    horizon_summary = {}
    for horizon in REQUIRED_HORIZONS:
        key = str(horizon)
        horizon_summary[key] = {
            metric: bootstrap_mean_interval(
                [float(fold["horizons"][key][metric]) for fold in folds],
                seed=1701 + horizon * 10 + ordinal,
            )
            for ordinal, metric in enumerate(
                (
                    "pooled_delta_vs_baseline",
                    "pooled_delta_vs_bias_only",
                    "daily_delta_vs_baseline",
                    "top_impact_delta_vs_baseline",
                    "post_cold_start_delta_vs_baseline",
                )
            )
        }

    all_reported = all(bool(fold["reported_gate_passed"]) for fold in folds)
    targeted_lower_bounds = all(
        horizon_summary[str(horizon)][metric]["lower_95"] > 0.0
        for horizon in TARGETED_HORIZONS
        for metric in (
            "pooled_delta_vs_baseline",
            "daily_delta_vs_baseline",
            "top_impact_delta_vs_baseline",
        )
    )
    no_large_pooled_regression = all(
        horizon_summary[str(horizon)]["pooled_delta_vs_baseline"]["worst_fold"]
        >= -0.002
        for horizon in REQUIRED_HORIZONS
    )
    qualification_passed = bool(
        all_reported and targeted_lower_bounds and no_large_pooled_regression
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "evidence_role": "retrospective_hypothesis_test",
        "promotion_eligible": False,
        "folds": list(folds),
        "fold_gate_passes": int(
            sum(bool(fold["reported_gate_passed"]) for fold in folds)
        ),
        "horizons": horizon_summary,
        "qualification_contract": {
            "all_fold_gates_passed": all_reported,
            "targeted_95pct_lower_bounds_positive": targeted_lower_bounds,
            "worst_pooled_delta_at_least_minus_0_002": no_large_pooled_regression,
        },
        "qualification_passed": qualification_passed,
        "live_orders_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the fixed online residual adapter across held-out folds."
    )
    parser.add_argument("--fold", action="append", type=parse_fold, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folds = [summarize_fold(name, path) for name, path in args.fold]
    report = aggregate(folds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker = "ONLINE_MULTIFOLD_PASSED" if report["qualification_passed"] else "ONLINE_MULTIFOLD_FAILED"
    (args.output_dir / marker).touch()
    print(
        json.dumps(
            {
                "qualification_passed": report["qualification_passed"],
                "promotion_eligible": report["promotion_eligible"],
                "output": str(output),
                "live_orders_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
