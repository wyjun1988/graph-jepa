from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_post_impact_live_prospective_schedule import (
    _append_event,
    build_cycle_commands,
    clock_timestamp,
    cycle_paths,
    validate_schedule_config,
)
from scripts.run_post_impact_rank_adapter_live_shadow_v2 import (
    load_rank_contract,
    validate_prospective_scope,
    validate_runtime_inputs,
)
from stock_v2.kiwoom_minute import KST
from stock_v2.prospective_ledger import read_prediction_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture immutable completed-bar snapshots and commit zero-order "
            "rank-adapter v2 predictions for one frozen session."
        )
    )
    parser.add_argument("--capture-config", required=True)
    parser.add_argument("--rank-contract", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary-root", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--lock-file", required=True)
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_session_contract(
    capture: Mapping[str, Any], rank_contract: Mapping[str, Any]
) -> tuple[tuple[str, ...], str]:
    clocks = validate_schedule_config(capture)
    session = str(pd.Timestamp(str(capture["session"])).date())
    realized_minutes = tuple(
        int(clock[:2]) * 60 + int(clock[3:]) for clock in clocks
    )
    evidence_classes = {
        validate_prospective_scope(
            rank_contract, session=session, clock_minute=clock_minute
        )
        for clock_minute in realized_minutes
    }
    if len(evidence_classes) != 1:
        raise ValueError("rank session mixes primary and diagnostic clocks")
    validate_runtime_inputs(
        rank_contract,
        argparse.Namespace(
            historical_day_release_dir=capture["historical_day_release_dir"],
            prospective_stale_cache_dir=capture["prospective_stale_cache_dir"],
            lifecycle_release_dir=capture["lifecycle_release_dir"],
        ),
    )
    return clocks, next(iter(evidence_classes))


def build_capture_command(
    capture: Mapping[str, Any], clock_hhmm: str
) -> tuple[list[str], Path]:
    command, _unused_inference, paths = build_cycle_commands(capture, clock_hhmm)
    if Path(command[1]).name != "capture_kiwoom_live_minute_snapshot.py":
        raise ValueError("rank session capture command changed")
    if any("order" in Path(value).name.lower() for value in command):
        raise ValueError("rank session contains an order-capable command")
    return command, paths["snapshot"]


def build_rank_command(
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
        str(ROOT / "scripts/run_post_impact_rank_adapter_live_shadow_v2.py"),
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
    if Path(command[1]).name != "run_post_impact_rank_adapter_live_shadow_v2.py":
        raise ValueError("rank session inference command changed")
    if any("order" in Path(value).name.lower() for value in command):
        raise ValueError("rank session contains an order-capable command")
    return command


def _run(command: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return {
        "returncode": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - started),
        "output_tail": completed.stdout[-4000:],
    }


def _commit_id(session: str, clock_hhmm: str) -> str:
    return (
        f"{session}|{clock_hhmm.replace(':', '')}|"
        "post_impact_rank_adapter_live_v2"
    )


def _committed(
    ledger: Path, artifact_root: Path, *, session: str, clock_hhmm: str
) -> bool:
    if not ledger.is_file():
        return False
    records = read_prediction_ledger(ledger, artifact_root=artifact_root)
    expected = _commit_id(session, clock_hhmm)
    return any(record["commit_id"] == expected for record in records)


def run_clock(
    capture: Mapping[str, Any],
    *,
    clock_hhmm: str,
    rank_contract: Path,
    artifact_root: Path,
    ledger: Path,
    summary_root: Path,
    event_log: Path,
) -> None:
    session = str(capture["session"])
    summary = summary_root / clock_hhmm.replace(":", "") / "summary.json"
    committed = _committed(
        ledger, artifact_root, session=session, clock_hhmm=clock_hhmm
    )
    if committed:
        if not summary.is_file():
            raise ValueError("rank ledger commit exists without its immutable summary")
        _append_event(
            event_log,
            {
                "event": "already_committed",
                "clock_kst": clock_hhmm,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )
        return
    if summary.exists():
        raise ValueError("rank summary exists without its ledger commit")

    capture_command, snapshot = build_capture_command(capture, clock_hhmm)
    manifest = snapshot / "manifest.json"
    if snapshot.exists():
        if not manifest.is_file():
            raise ValueError("rank snapshot directory is incomplete")
        capture_result = {
            "returncode": 0,
            "wall_seconds": 0.0,
            "output_tail": "existing immutable completed-bar snapshot reused",
        }
    else:
        capture_result = _run(capture_command)
    _append_event(
        event_log,
        {
            "event": "capture_complete",
            "clock_kst": clock_hhmm,
            "result": capture_result,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "live_orders_allowed": False,
            "broker_order_calls_executed": 0,
        },
    )
    if capture_result["returncode"] != 0 or not manifest.is_file():
        raise RuntimeError(f"rank completed-bar capture failed at {clock_hhmm}")

    rank_command = build_rank_command(
        capture,
        rank_contract=rank_contract,
        snapshot=snapshot,
        artifact_root=artifact_root,
        ledger=ledger,
        summary=summary,
    )
    inference_result = _run(rank_command)
    _append_event(
        event_log,
        {
            "event": "rank_inference_complete",
            "clock_kst": clock_hhmm,
            "result": inference_result,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "live_orders_allowed": False,
            "broker_order_calls_executed": 0,
        },
    )
    if (
        inference_result["returncode"] != 0
        or not summary.is_file()
        or not _committed(
            ledger, artifact_root, session=session, clock_hhmm=clock_hhmm
        )
    ):
        raise RuntimeError(f"rank read-only inference failed at {clock_hhmm}")


def main() -> int:
    args = parse_args()
    capture_path = _resolve(args.capture_config)
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    rank_contract_path = _resolve(args.rank_contract)
    rank_contract = load_rank_contract(rank_contract_path)
    clocks, evidence_class = validate_session_contract(capture, rank_contract)
    now = pd.Timestamp.now(tz=KST)
    if str(now.date()) != str(capture["session"]):
        raise ValueError("rank live session config is not for today in KST")

    artifact_root = _resolve(args.artifact_root)
    ledger = _resolve(args.ledger)
    summary_root = _resolve(args.summary_root)
    event_log = _resolve(args.event_log)
    lock_path = _resolve(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("rank live session is already running") from exc
        for clock in clocks:
            target = clock_timestamp(str(capture["session"]), clock)
            current = pd.Timestamp.now(tz=KST)
            if current < target:
                time.sleep(float((target - current).total_seconds()) + 1.0)
            elif current - target > pd.Timedelta(
                int(capture["max_lateness_minutes"]), unit="minute"
            ):
                failures += 1
                _append_event(
                    event_log,
                    {
                        "event": "skipped_too_late",
                        "clock_kst": clock,
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "live_orders_allowed": False,
                        "broker_order_calls_executed": 0,
                    },
                )
                continue
            try:
                run_clock(
                    capture,
                    clock_hhmm=clock,
                    rank_contract=rank_contract_path,
                    artifact_root=artifact_root,
                    ledger=ledger,
                    summary_root=summary_root,
                    event_log=event_log,
                )
            except Exception as exc:
                failures += 1
                _append_event(
                    event_log,
                    {
                        "event": "clock_failed",
                        "clock_kst": clock,
                        "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "live_orders_allowed": False,
                        "broker_order_calls_executed": 0,
                    },
                )
        _append_event(
            event_log,
            {
                "event": "session_complete",
                "session": capture["session"],
                "prospective_evidence_class": evidence_class,
                "failed_clocks": failures,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
