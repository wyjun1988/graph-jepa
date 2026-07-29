from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.intraday_release_audit import audit_intraday_trajectory_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit a portable intraday trajectory release."
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--minimum-shards", type=int, default=400)
    parser.add_argument("--require-input-files", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit_intraday_trajectory_release(
            args.release_dir,
            minimum_shards=args.minimum_shards,
            require_input_files=bool(args.require_input_files),
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "schema_version": 1,
            "audit_contract": "portable_intraday_trajectory_release_audit_v1",
            "release": str(Path(args.release_dir).resolve()),
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
    console = {
        "status": "passed" if report["passed"] else "failed",
        "report": str(output),
        "error": report.get("error"),
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    print(
        json.dumps(console, ensure_ascii=False, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
