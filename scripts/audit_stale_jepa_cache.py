from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_post_impact_reforecast import (
    STALE_CACHE_CONTRACT_V2,
    StaleCache,
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit a strict-OOS stale JEPA cache payload."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-causal-stock-graph", action="store_true")
    return parser.parse_args()


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    finite = 0
    total = int(np.prod(values.shape, dtype=np.int64))
    minimum: float | None = None
    maximum: float | None = None
    for row in range(values.shape[0]):
        block = np.asarray(values[row], dtype=np.float32)
        valid = np.isfinite(block)
        finite += int(valid.sum())
        if valid.any():
            block_minimum = float(block[valid].min())
            block_maximum = float(block[valid].max())
            minimum = block_minimum if minimum is None else min(minimum, block_minimum)
            maximum = block_maximum if maximum is None else max(maximum, block_maximum)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite_values": finite,
        "total_values": total,
        "finite_fraction": float(finite / total) if total else 0.0,
        "minimum": minimum,
        "maximum": maximum,
    }


def audit_cache(cache_dir: Path, *, require_graph: bool) -> dict[str, Any]:
    cache = StaleCache(cache_dir)
    target = pd.DatetimeIndex(pd.to_datetime(cache.dates, errors="raise"))
    context = pd.DatetimeIndex(pd.to_datetime(cache.context_dates, errors="raise"))
    graph = cache.graph_summary()
    failures: list[str] = []
    if require_graph and cache.cache_contract != STALE_CACHE_CONTRACT_V2:
        failures.append("cache_contract_is_not_graph_v2")
    if require_graph and not graph["available"]:
        failures.append("causal_stock_graph_is_absent")
    if not np.asarray(context < target).all():
        failures.append("context_date_is_not_strictly_before_target")
    arrays = {
        "context_latent": _array_summary(cache.context),
        "predicted_delta": _array_summary(cache.delta),
        "predicted_state": _array_summary(cache.state),
    }
    if any(record["finite_fraction"] != 1.0 for record in arrays.values()):
        failures.append("stale_cache_contains_non_finite_values")
    report = {
        "schema_version": 1,
        "audit_contract": "strict_oos_stale_daily_jepa_payload_audit_v1",
        "audited_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "cache_contract": cache.cache_contract,
        "manifest_sha256": file_sha256(cache.manifest_path),
        "dates": {
            "count": len(cache.dates),
            "target_start": cache.dates[0],
            "target_end": cache.dates[-1],
            "minimum_context_lag_days": int((target - context).days.min()),
            "maximum_context_lag_days": int((target - context).days.max()),
            "future_or_same_context_violations": int(
                np.count_nonzero(np.asarray(context >= target))
            ),
        },
        "tickers": {
            "count": len(cache.tickers),
            "unique": len(set(cache.tickers)),
        },
        "arrays": arrays,
        "stock_graph": graph,
        "files_verified_by_manifest_sha256": len(cache.manifest["files"]),
        "failures": failures,
        "integrity_gate_passed": not failures,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    return report


def main() -> int:
    args = parse_args()
    report = audit_cache(
        Path(args.cache_dir),
        require_graph=bool(args.require_causal_stock_graph),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed" if report["integrity_gate_passed"] else "failed",
                "dates": report["dates"]["count"],
                "tickers": report["tickers"]["count"],
                "stock_graph_edges": report["stock_graph"]["total_edges"],
                "failures": len(report["failures"]),
                "output": str(output),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["integrity_gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
