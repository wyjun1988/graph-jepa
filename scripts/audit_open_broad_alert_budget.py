from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stock_v2.systemic_transition import binary_ranking_metrics


DEFAULT_ALERT_RATES = (0.10, 0.125, 0.15, 0.175, 0.20)


def parse_fold(value: str) -> tuple[str, Path]:
    name, separator, raw_path = str(value).partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("folds must use NAME=REPORT_DIR")
    return name.strip(), Path(raw_path.strip())


def alert_rate_metrics(
    labels: Sequence[bool],
    scores: Sequence[float],
    rates: Sequence[float],
) -> dict[str, dict[str, Any]]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) < 20:
        raise ValueError("alert labels and scores must be aligned daily vectors")
    if not np.isfinite(scores).all() or labels.sum() == 0 or labels.all():
        raise ValueError("alert audit requires finite scores and both label classes")
    result = {}
    for rate in rates:
        rate = float(rate)
        if not 0.0 < rate <= 0.25:
            raise ValueError("alert rates must be in (0, 0.25]")
        result[f"{rate:.3f}"] = binary_ranking_metrics(
            labels,
            scores,
            selection_rate=rate,
        )
    return result


def select_validation_alert_rate(
    validation_metrics: Mapping[str, Mapping[str, Any]],
    *,
    minimum_recall: float = 0.50,
    minimum_lift: float = 1.50,
) -> str | None:
    qualified = [
        key
        for key, metrics in validation_metrics.items()
        if float(metrics["recall_at_selection_rate"]) >= float(minimum_recall)
        and float(metrics["lift_at_selection_rate"]) >= float(minimum_lift)
    ]
    return min(qualified, key=float) if qualified else None


def _boolean_column(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise ValueError("broad-selloff labels are not boolean")
    return normalized.isin(("true", "1")).to_numpy(dtype=bool)


def audit_fold(
    name: str,
    report_dir: Path,
    *,
    rates: Sequence[float] = DEFAULT_ALERT_RATES,
    prediction_mode: str = "candidate",
) -> dict[str, Any]:
    if prediction_mode not in ("candidate", "modular"):
        raise ValueError("prediction mode must be candidate or modular")
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or int(summary.get("schema_version", 0)) < 4:
        raise ValueError(f"{name} is not a completed schema-4 open-nowcast report")
    if summary.get("live_orders_allowed") is not False:
        raise ValueError(f"{name} does not explicitly prohibit live orders")
    required = {"split", "actual_broad_selloff", "score_broad_selloff"}
    metrics = {}
    prefix = "candidate" if prediction_mode == "candidate" else "modular"
    files = {
        "validation": f"{prefix}_selection_validation_daily.csv",
        "test": f"{prefix}_refit_test_daily.csv",
    }
    for split, filename in files.items():
        selected = pd.read_csv(report_dir / filename)
        if not required.issubset(selected.columns):
            raise ValueError(f"{name} {filename} lacks broad-alert columns")
        if set(selected["split"].astype(str)) != {split}:
            raise ValueError(f"{name} {filename} contains the wrong split")
        expected = int(summary["split_dates"][split])
        if len(selected) != expected:
            raise ValueError(f"{name} {split} daily rows differ from the report")
        metrics[split] = alert_rate_metrics(
            _boolean_column(selected["actual_broad_selloff"]),
            selected["score_broad_selloff"].to_numpy(dtype=np.float64),
            rates,
        )
    selected_rate = select_validation_alert_rate(metrics["validation"])
    selected_test = metrics["test"].get(selected_rate) if selected_rate else None
    checks = {
        "validation_selects_qualified_rate": selected_rate is not None,
        "selected_rate_at_most_0_20": selected_rate is not None
        and float(selected_rate) <= 0.20,
        "test_auc_at_least_0_52": selected_test is not None
        and float(selected_test["roc_auc"]) >= 0.52,
        "test_recall_at_least_0_25": selected_test is not None
        and float(selected_test["recall_at_selection_rate"]) >= 0.25,
        "test_lift_at_least_1_25": selected_test is not None
        and float(selected_test["lift_at_selection_rate"]) >= 1.25,
    }
    return {
        "fold": name,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "report_schema_version": int(summary["schema_version"]),
        "target_version": summary["target_version"],
        "open_sensor_contract": summary["open_sensor_contract"],
        "prediction_mode": prediction_mode,
        "rates": [float(value) for value in rates],
        "selection_rule": (
            "smallest validation-only rate with recall>=0.50 and lift>=1.50"
        ),
        "selected_rate": float(selected_rate) if selected_rate else None,
        "validation_metrics": metrics["validation"],
        "test_metrics": metrics["test"],
        "selected_test_metrics": selected_test,
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "failures": [key for key, passed in checks.items() if not passed],
        },
        "test_used_for_selection": False,
        "live_orders_allowed": False,
    }


def aggregate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(folds) != 5:
        raise ValueError("broad-alert audit requires exactly five folds")
    checkpoints = {str(row["checkpoint_sha256"]) for row in folds}
    contracts = {
        (
            int(row["report_schema_version"]),
            str(row["target_version"]),
            str(row["open_sensor_contract"]),
            str(row["prediction_mode"]),
            tuple(float(value) for value in row["rates"]),
        )
        for row in folds
    }
    if len(checkpoints) != len(folds) or len(contracts) != 1:
        raise ValueError("broad-alert folds must use distinct checkpoints and one contract")
    passes = sum(bool(row["gate"]["passed"]) for row in folds)
    selected_rates = [
        float(row["selected_rate"])
        for row in folds
        if row["selected_rate"] is not None
    ]
    checks = {
        "every_fold_passes_operational_alert_gate": passes == len(folds),
        "every_fold_selects_a_validation_rate": len(selected_rates) == len(folds),
        "mean_selected_rate_at_most_0_175": len(selected_rates) == len(folds)
        and float(np.mean(selected_rates)) <= 0.175,
    }
    return {
        "folds": len(folds),
        "passes": int(passes),
        "mean_selected_rate": (
            float(np.mean(selected_rates)) if selected_rates else None
        ),
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "failures": [key for key, passed in checks.items() if not passed],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit validation-only broad-selloff alert budgets across folds."
    )
    parser.add_argument("--fold", action="append", type=parse_fold, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rates", default="0.10,0.125,0.15,0.175,0.20")
    parser.add_argument(
        "--prediction-mode", choices=("candidate", "modular"), default="candidate"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rates = tuple(float(value.strip()) for value in args.rates.split(",") if value.strip())
    fold_rows = [
        audit_fold(name, path, rates=rates, prediction_mode=args.prediction_mode)
        for name, path in args.fold
    ]
    result = aggregate(fold_rows)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "role": "research_only_validation_selected_broad_alert_budget_audit",
        "prediction_mode": args.prediction_mode,
        "folds": fold_rows,
        "aggregate": result,
        "decision": "shadow_candidate" if result["gate"]["passed"] else "research_only",
        "live_orders_allowed": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "AUDIT_COMPLETE").touch()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
