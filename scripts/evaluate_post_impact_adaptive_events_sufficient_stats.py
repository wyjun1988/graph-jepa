from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_post_impact_adaptive_events as evaluator
from stock_v2.post_impact_reforecast import RegressionMetricAccumulator


ROLE = "post_impact_adaptive_event_sufficient_statistics_diagnostic"
CONTRACT = "regression_daily_sufficient_statistics_v1"
_ORIGINAL_METRICS = RegressionMetricAccumulator.metrics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_with_sufficient_statistics(
    accumulator: RegressionMetricAccumulator,
) -> dict[str, Any]:
    metrics = dict(_ORIGINAL_METRICS(accumulator))
    metrics["sufficient_statistics"] = {
        "count": int(accumulator.count),
        "prediction_sum": float(accumulator.prediction_sum),
        "target_sum": float(accumulator.target_sum),
        "prediction_squared_sum": float(accumulator.prediction_squared_sum),
        "target_squared_sum": float(accumulator.target_squared_sum),
        "prediction_target_cross_sum": float(accumulator.cross_sum),
        "squared_error_sum": float(accumulator.squared_error_sum),
    }
    return metrics


def main() -> int:
    args = evaluator.parse_args()
    if args.evaluation_scope != "validation_only":
        raise ValueError("sufficient-statistics diagnostic forbids test evaluation")
    RegressionMetricAccumulator.metrics = metrics_with_sufficient_statistics
    result = evaluator.main()
    output = Path(args.output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("test_evaluated") is not False or payload.get("test") is not None:
        raise ValueError("sufficient-statistics diagnostic evaluated a test split")
    payload["role"] = ROLE
    payload["sufficient_statistics_contract"] = CONTRACT
    payload["diagnostic_wrapper"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
        "wrapped_evaluator": str(Path(evaluator.__file__).resolve()),
        "wrapped_evaluator_sha256": file_sha256(Path(evaluator.__file__).resolve()),
    }
    payload["counts_as_primary_forward_evidence"] = False
    payload["promotion_eligible"] = False
    payload["live_orders_allowed"] = False
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "role": ROLE,
                "output": str(output),
                "output_sha256": file_sha256(output),
                "test_evaluated": False,
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
