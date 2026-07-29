from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _labeled_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise ValueError("folds must use unique NAME=PATH values")
        result[name] = Path(raw_path)
    return result


def _ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size < 2 or np.unique(labels).size != 2 or not np.isfinite(scores).all():
        raise ValueError("major-event ranking requires finite scores and two classes")
    event_rate = float(np.mean(labels))
    average_precision = float(average_precision_score(labels, scores))
    return {
        "rows": int(labels.size),
        "events": int(labels.sum()),
        "event_rate": event_rate,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": average_precision,
        "average_precision_lift": average_precision / max(event_rate, 1e-12),
    }


def _bool_series(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} contains non-boolean values")
    return normalized.map(mapping).astype(bool)


def _architecture_matches(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return all(observed.get(key) == value for key, value in expected.items())


def _load_fold(
    name: str,
    root: Path,
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = root / "summary.json"
    head_path = root / "market_transition_head.pt"
    major_path = root / "major_trajectory" / "summary.json"
    daily_path = root / "major_trajectory" / "daily_major_trajectory.csv"
    for path in (summary_path, head_path, major_path, daily_path):
        if not path.is_file():
            raise ValueError(f"missing family-query fold artifact for {name}: {path}")
    if name != contract["development_fold"] and not (root / "FOLD_COMPLETE").is_file():
        raise ValueError(f"confirmation fold lacks immutable completion marker: {name}")

    summary = _load(summary_path)
    major = _load(major_path)
    if summary.get("status") != "complete" or summary.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe or incomplete family-query fold: {name}")
    if summary.get("test_used_for_selection") is not False:
        raise ValueError(f"test split influenced family-query selection: {name}")
    if major.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe major-event fold: {name}")
    if major.get("test_used_for_threshold") is not False:
        raise ValueError(f"test split influenced the major-event threshold: {name}")
    if summary.get("target_version") != contract["target_version"]:
        raise ValueError(f"target version mismatch: {name}")
    if summary.get("impact_metric_version") != contract["impact_metric_version"]:
        raise ValueError(f"impact metric version mismatch: {name}")
    if major.get("target_version") != contract["target_version"]:
        raise ValueError(f"major-event target version mismatch: {name}")
    if major.get("impact_metric_version") != contract["impact_metric_version"]:
        raise ValueError(f"major-event metric version mismatch: {name}")

    expected = contract["folds"][name]
    architecture = summary.get("architecture", {})
    if not _architecture_matches(architecture, contract["architecture"]):
        raise ValueError(f"family-query architecture mismatch: {name}")
    if str(summary.get("train_end")) != str(expected["train_end"]):
        raise ValueError(f"fold train_end mismatch: {name}")

    daily = pd.read_csv(daily_path)
    required = {
        "date",
        "actual_normalized_salience",
        "predicted_normalized_salience",
        "major_trajectory_event",
    }
    if not required.issubset(daily.columns):
        raise ValueError(f"major-event daily schema mismatch: {name}")
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    if daily["date"].duplicated().any():
        raise ValueError(f"duplicate major-event dates: {name}")
    daily = daily.sort_values("date", kind="mergesort").reset_index(drop=True)
    for column in ("actual_normalized_salience", "predicted_normalized_salience"):
        daily[column] = pd.to_numeric(daily[column], errors="raise")
        if not np.isfinite(daily[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"non-finite {column}: {name}")
    daily["major_trajectory_event"] = _bool_series(
        daily["major_trajectory_event"], name=f"{name}.major_trajectory_event"
    )
    expected_test_rows = int(summary["split_dates"]["test"])
    if len(daily) != expected_test_rows:
        raise ValueError(f"daily/test row count mismatch: {name}")
    if len(daily) < int(contract["minimum_test_dates_per_fold"]):
        raise ValueError(f"insufficient test dates: {name}")
    if daily["major_trajectory_event"].nunique() != 2:
        raise ValueError(f"major-event test labels need two classes: {name}")
    daily["fold"] = name

    manifest_input = None
    if name != contract["development_fold"]:
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing confirmation run manifest: {name}")
        manifest = _load(manifest_path)
        expected_manifest = {
            "schema_version": 1,
            "role": "family_query_magnitude_risk_gate_fold_manifest",
            "fold": name,
            "contract_sha256": contract_sha256,
            "model_dir_name": expected["model_dir_name"],
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "training_recipe": contract["training_recipe"],
            "source_pins": contract["source_pins"],
            "promotion_eligible": False,
            "live_orders_allowed": False,
        }
        for key, value in expected_manifest.items():
            if manifest.get(key) != value:
                raise ValueError(f"confirmation manifest mismatch for {name}: {key}")
        local_artifacts = {
            "summary": summary_path,
            "head": head_path,
            "daily_test": root / "daily_test.csv",
            "major_summary": major_path,
            "major_daily": daily_path,
        }
        for artifact_name, path in local_artifacts.items():
            expected_sha = manifest.get("artifacts", {}).get(artifact_name, {}).get(
                "sha256"
            )
            if not path.is_file() or expected_sha != _sha256(path):
                raise ValueError(
                    f"confirmation artifact differs from manifest: {name}.{artifact_name}"
                )
        manifest_input = {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        }

    payload = {
        "parent_model_sha256": str(summary["parent_model_sha256"]),
        "train_end": str(summary["train_end"]),
        "split_dates": summary["split_dates"],
        "epochs_ran": len(summary.get("history", [])),
        "best_validation_score": float(summary["best_validation_score"]),
        "trajectory_auc": float(summary["metrics"]["test"]["trajectory"]["roc_auc"]),
        "major_auc": float(major["roc_auc"]),
        "major_average_precision_lift": float(major["average_precision_lift"]),
        "major_mass_lift": float(major["systemic_impact_mass_lift_at_major_rate"]),
        "major_peak_accuracy": float(major["peak_horizon_accuracy_on_major_events"]),
        "inputs": {
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "head": {"path": str(head_path), "sha256": _sha256(head_path)},
            "major": {"path": str(major_path), "sha256": _sha256(major_path)},
            "daily": {"path": str(daily_path), "sha256": _sha256(daily_path)},
            "manifest": manifest_input,
        },
    }
    return payload, daily


def _group_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(name): group.index.to_numpy(dtype=np.int64)
        for name, group in frame.groupby("fold", sort=True)
    }


def _moving_block_sample(
    groups: Mapping[str, np.ndarray],
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = []
    offsets = np.arange(int(block_length), dtype=np.int64)
    for values in groups.values():
        size = len(values)
        blocks = int(math.ceil(size / float(block_length)))
        starts = rng.integers(0, size, size=blocks)
        local = np.concatenate([(start + offsets) % size for start in starts])[:size]
        sampled.append(values[local])
    return np.concatenate(sampled)


def _circular_shift_placebo(
    scores: np.ndarray,
    groups: Mapping[str, np.ndarray],
    *,
    minimum_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    shifted = scores.copy()
    for values in groups.values():
        size = len(values)
        candidates = np.arange(1, size, dtype=np.int64)
        distance = np.minimum(candidates, size - candidates)
        eligible = candidates[distance >= int(minimum_shift)]
        if eligible.size == 0:
            raise ValueError("placebo minimum shift is too large for a confirmation fold")
        shift = int(rng.choice(eligible))
        shifted[values] = np.roll(scores[values], shift)
    return shifted


def _uncertainty(
    frame: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_samples: int,
    placebo_samples: int,
    placebo_minimum_shift: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    labels = frame["major_trajectory_event"].to_numpy(dtype=bool)
    scores = frame["fold_percentile_score"].to_numpy(dtype=np.float64)
    groups = _group_indices(frame)
    bootstrap_rng = np.random.default_rng(int(seed))
    bootstrap_auc = []
    for _ in range(int(bootstrap_samples)):
        indices = _moving_block_sample(
            groups, block_length=int(block_length), rng=bootstrap_rng
        )
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size == 2:
            bootstrap_auc.append(float(roc_auc_score(sampled_labels, scores[indices])))
    if len(bootstrap_auc) < max(100, int(bootstrap_samples * 0.90)):
        raise ValueError("too few valid moving-block bootstrap samples")

    placebo_rng = np.random.default_rng(int(seed) + 1)
    placebo_auc = [
        float(
            roc_auc_score(
                labels,
                _circular_shift_placebo(
                    scores,
                    groups,
                    minimum_shift=int(placebo_minimum_shift),
                    rng=placebo_rng,
                ),
            )
        )
        for _ in range(int(placebo_samples))
    ]
    observed = _ranking(labels, scores)
    placebo_array = np.asarray(placebo_auc, dtype=np.float64)
    return observed, {
        "method": "within_fold_circular_moving_block_bootstrap_and_shift_placebo",
        "block_length_days": int(block_length),
        "bootstrap_samples_requested": int(bootstrap_samples),
        "bootstrap_samples_valid": len(bootstrap_auc),
        "pooled_rank_auc_lower_95": float(np.quantile(bootstrap_auc, 0.025)),
        "pooled_rank_auc_upper_95": float(np.quantile(bootstrap_auc, 0.975)),
        "placebo_samples": int(placebo_samples),
        "placebo_minimum_shift_days": int(placebo_minimum_shift),
        "placebo_auc_99": float(np.quantile(placebo_array, 0.99)),
        "placebo_empirical_p": float(
            (1 + np.sum(placebo_array >= float(observed["roc_auc"])))
            / (len(placebo_array) + 1)
        ),
        "seed": int(seed),
    }


def evaluate_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    fold_roots: Mapping[str, Path],
    source_root: Path,
    bootstrap_samples: int | None = None,
    placebo_samples: int | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("unsupported family-query risk-gate contract schema")
    if contract.get("role") != "retrospective_family_query_magnitude_risk_gate":
        raise ValueError("invalid family-query risk-gate contract")
    if contract.get("test_used_for_hypothesis_generation") is not True:
        raise ValueError("development-fold reuse must be declared")
    if contract.get("promotion_eligible") is not False or contract.get(
        "live_orders_allowed"
    ) is not False:
        raise ValueError("risk-gate contract must remain research-only")
    expected_folds = contract["folds"]
    if set(fold_roots) != set(expected_folds):
        raise ValueError("risk-gate fold set differs from the contract")

    uncertainty_contract = contract["uncertainty"]
    resolved_bootstrap = int(uncertainty_contract["bootstrap_samples"])
    resolved_placebo = int(uncertainty_contract["placebo_samples"])
    resolved_seed = int(uncertainty_contract["seed"])
    if bootstrap_samples is not None and int(bootstrap_samples) != resolved_bootstrap:
        raise ValueError("bootstrap sample count differs from preregistration")
    if placebo_samples is not None and int(placebo_samples) != resolved_placebo:
        raise ValueError("placebo sample count differs from preregistration")
    if seed is not None and int(seed) != resolved_seed:
        raise ValueError("uncertainty seed differs from preregistration")

    source_pins = {
        relative: {
            "expected_sha256": str(expected),
            "observed_sha256": (
                _sha256(source_root / relative)
                if (source_root / relative).is_file()
                else None
            ),
        }
        for relative, expected in contract["source_pins"].items()
    }
    source_match = all(
        row["expected_sha256"] == row["observed_sha256"]
        for row in source_pins.values()
    )

    folds = {}
    frames = []
    fold_thresholds = contract["fold_checks"]
    contract_sha256 = _sha256(contract_path)
    for name in sorted(fold_roots):
        payload, daily = _load_fold(
            name,
            fold_roots[name],
            contract=contract,
            contract_sha256=contract_sha256,
        )
        expected = expected_folds[name]
        payload["checkpoint_matches_contract"] = (
            payload["parent_model_sha256"] == expected["checkpoint_sha256"]
        )
        payload["checks"] = {
            "checkpoint_matches_contract": payload["checkpoint_matches_contract"],
            "major_auc_at_least": (
                payload["major_auc"] >= float(fold_thresholds["major_auc_at_least"])
            ),
            "major_ap_lift_at_least": (
                payload["major_average_precision_lift"]
                >= float(fold_thresholds["major_average_precision_lift_at_least"])
            ),
            "major_mass_lift_at_least": (
                payload["major_mass_lift"]
                >= float(fold_thresholds["major_mass_lift_at_least"])
            ),
            "major_peak_accuracy_at_least": (
                payload["major_peak_accuracy"]
                >= float(fold_thresholds["major_peak_accuracy_at_least"])
            ),
        }
        payload["passed"] = all(payload["checks"].values())
        folds[name] = payload
        frames.append(daily)

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["fold", "date"], kind="mergesort"
    ).reset_index(drop=True)
    combined["fold_percentile_score"] = combined.groupby("fold", sort=True)[
        "predicted_normalized_salience"
    ].rank(method="average", pct=True)

    confirmation_names = tuple(contract["confirmation_folds"])
    if contract["development_fold"] in confirmation_names:
        raise ValueError("development fold cannot be a confirmation fold")
    if set(confirmation_names) != set(expected_folds) - {contract["development_fold"]}:
        raise ValueError("confirmation folds must be every non-development fold")
    confirmation = (
        combined[combined["fold"].isin(confirmation_names)]
        .copy()
        .reset_index(drop=True)
    )
    pooled, uncertainty = _uncertainty(
        confirmation,
        block_length=int(uncertainty_contract["block_length_days"]),
        bootstrap_samples=resolved_bootstrap,
        placebo_samples=resolved_placebo,
        placebo_minimum_shift=int(
            uncertainty_contract["placebo_minimum_shift_days"]
        ),
        seed=resolved_seed,
    )

    confirmation_passes = int(sum(folds[name]["passed"] for name in confirmation_names))
    aggregate_checks = contract["aggregate_checks"]
    mean_ap_lift = float(
        np.mean([folds[name]["major_average_precision_lift"] for name in confirmation_names])
    )
    mean_mass_lift = float(
        np.mean([folds[name]["major_mass_lift"] for name in confirmation_names])
    )
    development = contract["development_fold"]
    checks = {
        "source_pins_match_contract": source_match,
        "development_artifacts_match": (
            folds[development]["inputs"]["summary"]["sha256"]
            == expected_folds[development]["summary_sha256"]
            and folds[development]["inputs"]["head"]["sha256"]
            == expected_folds[development]["head_sha256"]
            and folds[development]["inputs"]["major"]["sha256"]
            == expected_folds[development]["major_summary_sha256"]
        ),
        "confirmation_folds_pass_at_least": (
            confirmation_passes
            >= int(aggregate_checks["confirmation_folds_pass_at_least"])
        ),
        "confirmation_pooled_rank_auc_at_least": (
            pooled["roc_auc"]
            >= float(aggregate_checks["confirmation_pooled_rank_auc_at_least"])
        ),
        "confirmation_block_auc_lower_above": (
            float(uncertainty["pooled_rank_auc_lower_95"])
            > float(aggregate_checks["confirmation_block_auc_lower_above"])
        ),
        "confirmation_mean_major_ap_lift_at_least": (
            mean_ap_lift
            >= float(aggregate_checks["confirmation_mean_major_ap_lift_at_least"])
        ),
        "confirmation_mean_major_mass_lift_at_least": (
            mean_mass_lift
            >= float(aggregate_checks["confirmation_mean_major_mass_lift_at_least"])
        ),
        "confirmation_auc_beats_99pct_shift_placebo": (
            pooled["roc_auc"] > float(uncertainty["placebo_auc_99"])
        ),
    }
    passed = all(checks.values())
    result = {
        "status": "complete",
        "role": "family_query_magnitude_risk_gate_multifold_audit",
        "contract": str(contract_path),
        "contract_sha256": contract_sha256,
        "source_pins": source_pins,
        "development_fold_excluded_from_confirmation_statistics": development,
        "confirmation_folds": list(confirmation_names),
        "folds": folds,
        "confirmation_passes": confirmation_passes,
        "confirmation_total": len(confirmation_names),
        "confirmation_pooled_rank_metrics": pooled,
        "confirmation_mean_major_average_precision_lift": mean_ap_lift,
        "confirmation_mean_major_mass_lift": mean_mass_lift,
        "uncertainty": uncertainty,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "passed": passed,
        "failures": [name for name, value in checks.items() if not value],
        "decision": (
            "magnitude_risk_gate_requires_comparator_exposure_backtest"
            if passed
            else "reject_family_query_magnitude_risk_gate"
        ),
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    return result, combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate preregistered family-query magnitude risk-gate folds."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--placebo-samples", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    payload, daily = evaluate_contract(
        _load(contract_path),
        contract_path=contract_path,
        fold_roots=_labeled_paths(args.fold),
        source_root=Path(args.source_root),
        bootstrap_samples=args.bootstrap_samples,
        placebo_samples=args.placebo_samples,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite risk-gate aggregate: {output_dir}")
    output_dir.mkdir(parents=True)
    daily.to_csv(output_dir / "pooled_daily_major_trajectory.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "DIAGNOSTIC_ONLY").touch()
    print(json.dumps({"decision": payload["decision"], "checks": payload["checks"]}))


if __name__ == "__main__":
    main()
