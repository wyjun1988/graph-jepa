from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.rolling_validation import audit_rolling_sensor_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit structured sensor coverage for all rolling v6 folds."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = {}
    for value in args.report:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("reports must use LABEL=PATH")
        if label in reports:
            raise ValueError(f"duplicate report label: {label}")
        reports[label] = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    payload = audit_rolling_sensor_reports(contract, reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "blockers": len(payload["blockers"]),
                "live_orders_allowed": False,
            }
        )
    )
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
