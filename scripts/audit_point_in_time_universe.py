from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.pit_universe_audit import audit_point_in_time_universe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a point-in-time KRX universe.")
    parser.add_argument("--universe", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--compare-universe")
    parser.add_argument("--evaluation-end")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit_point_in_time_universe(
            args.universe,
            failures_path=args.failures,
            comparison_universe_path=args.compare_universe,
            evaluation_end=args.evaluation_end,
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "schema_version": 1,
            "audit_contract": "point_in_time_liquidity_universe_audit_v1",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "promotion_eligible": False,
            "live_orders_allowed": False,
        }
        exit_code = 1
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed" if report["passed"] else "failed",
                "report": str(output),
                "stocks": report.get("stocks"),
                "overlap": (report.get("comparison") or {}).get("overlap"),
                "error": report.get("error"),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
