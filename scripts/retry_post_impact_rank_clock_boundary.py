from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_post_impact_live_prospective_schedule import (
    _append_event,
    clock_timestamp,
    cycle_paths,
)
from scripts.run_post_impact_rank_adapter_live_session_v2 import (
    _committed,
    run_clock,
    validate_session_contract,
)
from scripts.run_post_impact_rank_adapter_live_shadow_v2 import load_rank_contract
from stock_v2.kiwoom_minute import KST


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def wait_past_boundary(
    target: pd.Timestamp,
    grace_seconds: float,
    *,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    not_before = target + pd.Timedelta(float(grace_seconds), unit="s")
    clock = now_fn or (lambda: pd.Timestamp.now(tz=KST))
    while True:
        remaining = float((not_before - clock()).total_seconds())
        if remaining <= 0.0:
            return
        sleep_fn(min(5.0, remaining + 0.25))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retry one frozen read-only rank clock only when the primary scheduler "
            "left no immutable snapshot or in-progress capture after the boundary."
        )
    )
    parser.add_argument("--capture-config", required=True)
    parser.add_argument("--rank-contract", required=True)
    parser.add_argument("--clock", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary-root", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--grace-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if float(args.grace_seconds) < 2.0:
        raise ValueError("boundary retry requires at least two seconds of grace")

    capture_path = _resolve(args.capture_config)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    rank_contract_path = _resolve(args.rank_contract)
    rank_contract = load_rank_contract(rank_contract_path)
    clocks, _evidence_class = validate_session_contract(capture, rank_contract)
    if str(args.clock) not in clocks:
        raise ValueError("retry clock is outside the frozen session contract")
    target = clock_timestamp(str(capture["session"]), str(args.clock))
    if str(pd.Timestamp.now(tz=KST).date()) != str(capture["session"]):
        raise ValueError("boundary retry config is not for today")
    wait_past_boundary(target, float(args.grace_seconds))

    artifact_root = _resolve(args.artifact_root)
    ledger = _resolve(args.ledger)
    summary_root = _resolve(args.summary_root)
    event_log = _resolve(args.event_log)
    paths = cycle_paths(capture, str(args.clock))
    snapshot = paths["snapshot"]
    temporary_pattern = f".{snapshot.name}.tmp-*"
    in_progress = sorted(snapshot.parent.glob(temporary_pattern))
    if _committed(
        ledger,
        artifact_root,
        session=str(capture["session"]),
        clock_hhmm=str(args.clock),
    ):
        event = "boundary_retry_not_needed_committed"
    elif in_progress:
        event = "boundary_retry_not_needed_capture_in_progress"
    else:
        _append_event(
            event_log,
            {
                "event": "boundary_retry_start",
                "clock_kst": str(args.clock),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "retry_reason": "no commit or in-progress immutable capture after boundary grace",
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )
        try:
            run_clock(
                capture,
                clock_hhmm=str(args.clock),
                rank_contract=rank_contract_path,
                artifact_root=artifact_root,
                ledger=ledger,
                summary_root=summary_root,
                event_log=event_log,
            )
        except Exception as exc:
            _append_event(
                event_log,
                {
                    "event": "boundary_retry_failed",
                    "clock_kst": str(args.clock),
                    "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "live_orders_allowed": False,
                    "broker_order_calls_executed": 0,
                },
            )
            raise
        event = "boundary_retry_complete"
    _append_event(
        event_log,
        {
            "event": event,
            "clock_kst": str(args.clock),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "live_orders_allowed": False,
            "broker_order_calls_executed": 0,
        },
    )


if __name__ == "__main__":
    main()
