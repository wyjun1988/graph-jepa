from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_post_impact_live_prospective_schedule import (
    _append_event,
    clock_timestamp,
    cycle_paths,
    validate_schedule_config,
)
from scripts.run_post_impact_rank_adapter_live_shadow import (
    load_rank_contract,
    prospective_scope,
)
from stock_v2.kiwoom_minute import KST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Follow an existing read-only capture schedule and commit rank-adapter "
            "shadow predictions without making broker calls."
        )
    )
    parser.add_argument("--capture-config", required=True)
    parser.add_argument("--rank-contract", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary-root", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--lock-file", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--snapshot-wait-minutes", type=int, default=20)
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def build_rank_inference_command(
    capture: Mapping[str, Any],
    *,
    rank_contract: Path,
    snapshot: Path,
    artifact_root: Path,
    ledger: Path,
    summary: Path,
) -> list[str]:
    command = [
        str(_resolve(capture["python"])),
        str(ROOT / "scripts/run_post_impact_rank_adapter_live_shadow.py"),
        "--contract",
        str(rank_contract),
        "--snapshot-dir",
        str(snapshot),
        "--historical-day-release-dir",
        str(_resolve(capture["historical_day_release_dir"])),
        "--prospective-stale-cache-dir",
        str(_resolve(capture["prospective_stale_cache_dir"])),
        "--lifecycle-release-dir",
        str(_resolve(capture["lifecycle_release_dir"])),
        "--device",
        str(capture["device"]),
        "--history-context-sessions",
        str(capture["history_context_sessions"]),
        "--minimum-latest-nodes",
        str(capture["minimum_latest_nodes"]),
        "--artifact-root",
        str(artifact_root),
        "--ledger",
        str(ledger),
        "--summary-output",
        str(summary),
    ]
    if any("capture" in Path(value).name.lower() for value in command):
        raise ValueError("rank follower command unexpectedly contains a capture script")
    if any("order" in Path(value).name.lower() for value in command):
        raise ValueError("rank follower command unexpectedly contains an order script")
    return command


def select_rank_clocks(
    capture_clocks: tuple[str, ...], contract: Mapping[str, Any]
) -> tuple[str, ...]:
    _first_session, primary_minutes = prospective_scope(contract)
    available = {
        int(value[:2]) * 60 + int(value[3:]): value for value in capture_clocks
    }
    missing = [minute for minute in primary_minutes if minute not in available]
    if missing:
        raise ValueError("capture schedule is missing frozen rank-adapter clocks")
    return tuple(available[minute] for minute in primary_minutes)


def _run(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "returncode": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - started),
        "output_tail": completed.stdout[-4000:],
    }


def main() -> int:
    args = parse_args()
    if float(args.poll_seconds) < 1.0:
        raise ValueError("rank follower poll interval is too short")
    if int(args.snapshot_wait_minutes) < 5:
        raise ValueError("rank follower snapshot wait is too short")
    capture_path = _resolve(args.capture_config)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture_clocks = validate_schedule_config(capture)
    if capture.get("live_orders_allowed") is not False:
        raise ValueError("capture schedule permits live orders")
    rank_contract = _resolve(args.rank_contract)
    contract = load_rank_contract(rank_contract)
    clocks = select_rank_clocks(capture_clocks, contract)
    now = pd.Timestamp.now(tz=KST)
    if str(now.date()) != str(capture["session"]):
        raise ValueError("rank follower capture session is not today")

    artifact_root = _resolve(args.artifact_root)
    ledger = _resolve(args.ledger)
    summary_root = _resolve(args.summary_root)
    event_log = _resolve(args.event_log)
    lock_path = _resolve(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("rank-adapter shadow follower is already running") from exc
        for clock in clocks:
            paths = cycle_paths(capture, clock)
            summary = summary_root / clock.replace(":", "") / "summary.json"
            if summary.is_file():
                _append_event(
                    event_log,
                    {
                        "event": "already_committed",
                        "clock_kst": clock,
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "live_orders_allowed": False,
                        "broker_order_calls_executed": 0,
                    },
                )
                continue
            target = clock_timestamp(str(capture["session"]), clock)
            current = pd.Timestamp.now(tz=KST)
            if current < target:
                time.sleep(float((target - current).total_seconds()) + 1.0)
            deadline = target + pd.Timedelta(
                int(args.snapshot_wait_minutes), unit="minute"
            )
            manifest = paths["snapshot"] / "manifest.json"
            upstream_summary = paths["summary"]
            while (
                not manifest.is_file() or not upstream_summary.is_file()
            ) and pd.Timestamp.now(tz=KST) <= deadline:
                time.sleep(float(args.poll_seconds))
            if not manifest.is_file() or not upstream_summary.is_file():
                _append_event(
                    event_log,
                    {
                        "event": "upstream_timeout",
                        "clock_kst": clock,
                        "snapshot_ready": manifest.is_file(),
                        "upstream_inference_ready": upstream_summary.is_file(),
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "live_orders_allowed": False,
                        "broker_order_calls_executed": 0,
                    },
                )
                continue
            command = build_rank_inference_command(
                capture,
                rank_contract=rank_contract,
                snapshot=paths["snapshot"],
                artifact_root=artifact_root,
                ledger=ledger,
                summary=summary,
            )
            result = _run(command)
            _append_event(
                event_log,
                {
                    "event": "rank_shadow_inference_complete",
                    "clock_kst": clock,
                    "result": result,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "live_orders_allowed": False,
                    "broker_order_calls_executed": 0,
                },
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
