from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_minute import KST
from stock_v2.prospective_ledger import read_prediction_ledger


CLOCK_PATTERN = re.compile(r"(?:0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a frozen one-session schedule of read-only completed-bar "
            "captures and prospective scientific-control commits."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--lock-file", required=True)
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_schedule_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    if config.get("schema_version") != 1 or config.get("role") != (
        "post_impact_live_prospective_schedule"
    ):
        raise ValueError("invalid live prospective schedule config")
    if config.get("live_orders_allowed") is not False:
        raise ValueError("live prospective schedule permits live orders")
    if config.get("broker_order_calls_allowed") is not False:
        raise ValueError("live prospective schedule permits broker order calls")
    session = str(config.get("session") or "")
    datetime.fromisoformat(session).date()
    clocks = tuple(str(value) for value in config.get("clocks_kst") or [])
    if not clocks or any(CLOCK_PATTERN.fullmatch(value) is None for value in clocks):
        raise ValueError("live prospective schedule clocks are invalid")
    if list(clocks) != sorted(set(clocks)):
        raise ValueError("live prospective schedule clocks must be unique and sorted")
    for name in (
        "python",
        "universe_manifest",
        "env_file",
        "contract",
        "historical_day_release_dir",
        "prospective_stale_cache_dir",
        "lifecycle_release_dir",
    ):
        path = _resolve(config.get(name))
        if not path.exists():
            raise FileNotFoundError(f"live prospective schedule input is missing: {name}")
    if int(config.get("interval_minutes", 0)) != 5:
        raise ValueError("live prospective schedule requires the frozen 5-minute grid")
    if str(config.get("timestamp_semantics")) != "start":
        raise ValueError("live prospective schedule timestamp semantics changed")
    if float(config.get("sleep_seconds", 0.0)) < 0.2:
        raise ValueError("live prospective schedule violates Kiwoom pacing")
    if int(config.get("minimum_populated_tickers", 0)) < 400:
        raise ValueError("live prospective schedule population gate is too low")
    if int(config.get("minimum_latest_nodes", 0)) < 400:
        raise ValueError("live prospective schedule node gate is too low")
    return clocks


def clock_timestamp(session: str, clock_hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{session} {clock_hhmm}", tz=KST)


def cycle_paths(config: Mapping[str, Any], clock_hhmm: str) -> dict[str, Path]:
    session = str(config["session"])
    compact = session.replace("-", "")
    hhmm = clock_hhmm.replace(":", "")
    return {
        "snapshot": _resolve(config["snapshot_root"])
        / f"krx500_{compact}_{hhmm}_v1",
        "summary": _resolve(config["summary_root"])
        / f"post_impact_live_prospective_scientific_v1_{compact}_{hhmm}"
        / "summary.json",
    }


def build_cycle_commands(
    config: Mapping[str, Any], clock_hhmm: str
) -> tuple[list[str], list[str], dict[str, Path]]:
    paths = cycle_paths(config, clock_hhmm)
    python = str(_resolve(config["python"]))
    capture = [
        python,
        str(ROOT / "scripts/capture_kiwoom_live_minute_snapshot.py"),
        "--universe-manifest",
        str(_resolve(config["universe_manifest"])),
        "--session",
        str(config["session"]),
        "--interval-minutes",
        str(config["interval_minutes"]),
        "--timestamp-semantics",
        str(config["timestamp_semantics"]),
        "--cutoff-hhmm",
        clock_hhmm,
        "--env-file",
        str(_resolve(config["env_file"])),
        "--server",
        "real",
        "--sleep-sec",
        str(config["sleep_seconds"]),
        "--max-pages",
        str(config["maximum_pages"]),
        "--minimum-populated-tickers",
        str(config["minimum_populated_tickers"]),
        "--output-dir",
        str(paths["snapshot"]),
    ]
    inference = [
        python,
        str(ROOT / "scripts/run_post_impact_live_prospective_inference.py"),
        "--contract",
        str(_resolve(config["contract"])),
        "--snapshot-dir",
        str(paths["snapshot"]),
        "--historical-day-release-dir",
        str(_resolve(config["historical_day_release_dir"])),
        "--prospective-stale-cache-dir",
        str(_resolve(config["prospective_stale_cache_dir"])),
        "--lifecycle-release-dir",
        str(_resolve(config["lifecycle_release_dir"])),
        "--device",
        str(config["device"]),
        "--history-context-sessions",
        str(config["history_context_sessions"]),
        "--minimum-latest-nodes",
        str(config["minimum_latest_nodes"]),
        "--artifact-root",
        str(_resolve(config["artifact_root"])),
        "--ledger",
        str(_resolve(config["ledger"])),
        "--summary-output",
        str(paths["summary"]),
    ]
    if any("order" in Path(value).name.lower() for value in (capture[1], inference[1])):
        raise ValueError("live prospective cycle includes an order-capable script")
    return capture, inference, paths


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _committed(config: Mapping[str, Any], clock_hhmm: str) -> bool:
    ledger = _resolve(config["ledger"])
    records = read_prediction_ledger(
        ledger, artifact_root=_resolve(config["artifact_root"])
    )
    commit_id = (
        f"{config['session']}|{clock_hhmm.replace(':', '')}|"
        "post_impact_scientific_live_v1"
    )
    return any(record["commit_id"] == commit_id for record in records)


def _run_command(command: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
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


def run_cycle(
    config: Mapping[str, Any],
    clock_hhmm: str,
    *,
    event_log: Path,
) -> None:
    if _committed(config, clock_hhmm):
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
    capture, inference, paths = build_cycle_commands(config, clock_hhmm)
    if paths["snapshot"].exists():
        capture_result = {
            "returncode": 0,
            "wall_seconds": 0.0,
            "output_tail": "existing immutable snapshot reused",
        }
    else:
        capture_result = _run_command(capture)
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
    if capture_result["returncode"] != 0:
        raise RuntimeError(f"live capture failed at {clock_hhmm}")
    inference_result = _run_command(inference)
    _append_event(
        event_log,
        {
            "event": "inference_complete",
            "clock_kst": clock_hhmm,
            "result": inference_result,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "live_orders_allowed": False,
            "broker_order_calls_executed": 0,
        },
    )
    if inference_result["returncode"] != 0 or not _committed(config, clock_hhmm):
        raise RuntimeError(f"live prospective inference failed at {clock_hhmm}")


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    clocks = validate_schedule_config(config)
    now = pd.Timestamp.now(tz=KST)
    if str(now.date()) != str(config["session"]):
        raise ValueError("live prospective schedule session is not today in KST")
    event_log = _resolve(args.event_log)
    lock_path = _resolve(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("live prospective schedule is already running") from exc
        for clock in clocks:
            target = clock_timestamp(str(config["session"]), clock)
            current = pd.Timestamp.now(tz=KST)
            lateness = current - target
            if lateness > pd.Timedelta(
                int(config["max_lateness_minutes"]), unit="minute"
            ):
                _append_event(
                    event_log,
                    {
                        "event": "skipped_too_late",
                        "clock_kst": clock,
                        "lateness_seconds": float(lateness.total_seconds()),
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "live_orders_allowed": False,
                        "broker_order_calls_executed": 0,
                    },
                )
                continue
            if current < target:
                time.sleep(float((target - current).total_seconds()) + 1.0)
            try:
                run_cycle(config, clock, event_log=event_log)
            except Exception as exc:
                _append_event(
                    event_log,
                    {
                        "event": "cycle_failed",
                        "clock_kst": clock,
                        "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                        "live_orders_allowed": False,
                        "broker_order_calls_executed": 0,
                    },
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
