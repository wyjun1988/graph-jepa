from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
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
    validate_schedule_config,
)
from scripts.run_post_impact_rank_adapter_live_shadow import (
    load_rank_contract,
    prospective_scope,
)
from scripts.train_post_impact_reforecast import StaleCache
from stock_v2.kiwoom_minute import KST
from stock_v2.prospective_ledger import file_sha256


ROLE = "post_impact_next_session_zero_order_cycle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one immutable next-session release, then run the shared "
            "read-only capture schedule and scoped rank-adapter follower."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--phase-log", required=True)
    parser.add_argument("--lock-file", required=True)
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"next-session path escapes project root: {path}") from exc


def _session(value: object) -> str:
    return str(pd.Timestamp(str(value)).date())


def validate_cycle_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("role") != ROLE:
        raise ValueError("invalid next-session cycle config")
    if config.get("live_orders_allowed") is not False:
        raise ValueError("next-session cycle permits live orders")
    if config.get("broker_order_calls_allowed") is not False:
        raise ValueError("next-session cycle permits broker order calls")
    context = pd.Timestamp(_session(config.get("context_session")))
    target = pd.Timestamp(_session(config.get("target_session")))
    if target <= context or target.weekday() >= 5:
        raise ValueError("next-session target must be a later weekday")
    run_at = pd.Timestamp(config.get("run_at_kst"))
    deadline = pd.Timestamp(config.get("state_ready_deadline_kst"))
    if run_at.tzinfo is None or deadline.tzinfo is None:
        raise ValueError("next-session cycle times must be timezone-aware")
    run_at = run_at.tz_convert(KST)
    deadline = deadline.tz_convert(KST)
    if str(run_at.date()) != str(target.date()) or not run_at < deadline:
        raise ValueError("next-session run window is invalid")
    if float(config.get("state_poll_seconds", 0.0)) < 5.0:
        raise ValueError("next-session state polling is too frequent")
    for name in (
        "python",
        "forward_state",
        "daily_config",
        "base_live_schedule",
        "base_rank_contract",
    ):
        if not _resolve(config.get(name)).exists():
            raise FileNotFoundError(f"next-session static input is missing: {name}")
    clocks = tuple(str(value) for value in config.get("capture_clocks_kst") or ())
    if list(clocks) != sorted(set(clocks)):
        raise ValueError("next-session capture clocks must be unique and sorted")
    base_rank = load_rank_contract(_resolve(config["base_rank_contract"]))
    first_session, rank_minutes = prospective_scope(base_rank)
    if first_session != str(target.date()):
        raise ValueError("rank-adapter first session differs from target")
    capture_minutes = {
        int(clock[:2]) * 60 + int(clock[3:]) for clock in clocks
    }
    if any(minute not in capture_minutes for minute in rank_minutes):
        raise ValueError("capture schedule omits a frozen rank-adapter clock")


def _verified_state_artifact(
    state: Mapping[str, Any], name: str
) -> tuple[Path, str]:
    record = state.get("artifacts", {}).get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"forward state lacks {name}")
    root = _resolve(record.get("root"))
    manifest = _resolve(record.get("path"))
    if not root.is_dir() or manifest != root / "manifest.json":
        raise ValueError(f"forward state {name} path changed")
    observed = file_sha256(manifest)
    if observed != record.get("sha256"):
        raise ValueError(f"forward state {name} hash changed")
    return root, observed


def load_ready_forward_state(
    path: Path, *, context_session: str
) -> dict[str, Any] | None:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("role") != "post_impact_forward_daily_state":
        raise ValueError("unexpected forward state role")
    if state.get("live_orders_allowed") is not False or state.get("orders_sent") != 0:
        raise ValueError("forward state is not zero-order")
    if str(state.get("latest_session")) != str(context_session):
        return None
    _verified_state_artifact(state, "day_release")
    _verified_state_artifact(state, "lifecycle")
    return state


def validate_prospective_cache(
    path: Path, *, context_session: str, target_session: str
) -> str:
    cache = StaleCache(path)
    manifest = cache.manifest
    prospective = manifest.get("prospective_target") or {}
    if not (
        cache.dates == (target_session,)
        and cache.context_dates == (context_session,)
        and prospective.get("enabled") is True
        and prospective.get("target_date") == target_session
        and prospective.get("context_date") == context_session
        and prospective.get("target_observations_injected") is False
        and manifest.get("live_orders_allowed") is False
        and manifest.get("promotion_eligible") is False
    ):
        raise ValueError("next-session prospective cache contract changed")
    return file_sha256(path / "manifest.json")


def build_prospective_cache_command(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    daily: Mapping[str, Any],
) -> list[str]:
    lifecycle, _manifest_sha = _verified_state_artifact(state, "lifecycle")
    target = str(config["target_session"])
    context = str(config["context_session"])
    command = [
        str(_resolve(config["python"])),
        str(ROOT / "scripts/build_stale_jepa_rollout_cache.py"),
        "--model-dir",
        str(_resolve(daily["jepa"]["model_dir"])),
        "--target-start",
        target,
        "--target-end",
        target,
        "--device",
        str(config.get("device", "mps")),
        "--batch-size",
        str(daily["jepa"]["batch_size"]),
        "--edge-cache-workers",
        str(daily["jepa"]["edge_cache_workers"]),
        "--cache-dir",
        str(lifecycle / "ohlcv"),
        "--prospective-target-date",
        target,
        "--prospective-context-date",
        context,
        "--output-dir",
        str(_resolve(config["prospective_stale_cache_dir"])),
    ]
    if any("order" in Path(value).name.lower() for value in command):
        raise ValueError("prospective cache command unexpectedly contains an order script")
    return command


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"immutable next-session output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def build_daily_rank_contract(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    prospective_cache_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    base_path = _resolve(config["base_rank_contract"])
    child = json.loads(base_path.read_text(encoding="utf-8"))
    target = str(config["target_session"])
    day_release, day_sha = _verified_state_artifact(state, "day_release")
    lifecycle, lifecycle_sha = _verified_state_artifact(state, "lifecycle")
    cache = _resolve(config["prospective_stale_cache_dir"])
    child["contract_id"] = f"post_impact_rank_adapter_live_shadow_{target.replace('-', '')}"
    child["created_at"] = created_at
    child["daily_session"] = target
    child["first_live_session"] = target
    child["first_primary_session"] = target
    child["parent_scope_contract"] = {
        "path": _portable(base_path),
        "sha256": file_sha256(base_path),
    }
    child["runtime_inputs"] = {
        "historical_day_release": {
            "path": _portable(day_release),
            "manifest_sha256": day_sha,
        },
        "prospective_stale_cache": {
            "path": _portable(cache),
            "manifest_sha256": prospective_cache_sha256,
        },
        "lifecycle_release": {
            "path": _portable(lifecycle),
            "manifest_sha256": lifecycle_sha,
        },
    }
    source_pins = dict(child["source_pins"])
    for relative in (
        "scripts/run_post_impact_next_session_shadow_cycle.py",
        "scripts/run_post_impact_rank_adapter_shadow_follower.py",
        "scripts/run_post_impact_live_prospective_schedule.py",
        "scripts/capture_kiwoom_live_minute_snapshot.py",
    ):
        source_pins[relative] = file_sha256(ROOT / relative)
    child["source_pins"] = source_pins
    child["promotion_eligible"] = False
    child["live_orders_allowed"] = False
    child["broker_order_calls_allowed"] = False
    return child


def build_daily_live_schedule(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    created_at: str,
) -> dict[str, Any]:
    base_path = _resolve(config["base_live_schedule"])
    schedule = json.loads(base_path.read_text(encoding="utf-8"))
    day_release, _day_sha = _verified_state_artifact(state, "day_release")
    lifecycle, _lifecycle_sha = _verified_state_artifact(state, "lifecycle")
    schedule["session"] = str(config["target_session"])
    schedule["clocks_kst"] = list(config["capture_clocks_kst"])
    schedule["historical_day_release_dir"] = _portable(day_release)
    schedule["prospective_stale_cache_dir"] = _portable(
        _resolve(config["prospective_stale_cache_dir"])
    )
    schedule["lifecycle_release_dir"] = _portable(lifecycle)
    schedule["created_at"] = created_at
    schedule["parent_schedule"] = {
        "path": _portable(base_path),
        "sha256": file_sha256(base_path),
    }
    schedule["live_orders_allowed"] = False
    schedule["broker_order_calls_allowed"] = False
    return schedule


def _run_command(command: Sequence[str], phase_log: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    with phase_log.open("a", encoding="utf-8") as handle:
        handle.write(completed.stdout)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "returncode": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - started),
        "output_tail": completed.stdout[-4000:],
    }


def _schedule_commands(
    config: Mapping[str, Any], daily_rank_contract: Path, daily_schedule: Path
) -> tuple[list[str], list[str]]:
    python = str(_resolve(config["python"]))
    target = str(config["target_session"]).replace("-", "")
    main_root = _resolve(config["main_schedule_ops_root"])
    rank_root = _resolve(config["rank_schedule_ops_root"])
    main = [
        python,
        str(ROOT / "scripts/run_post_impact_live_prospective_schedule.py"),
        "--config",
        str(daily_schedule),
        "--event-log",
        str(main_root / "events.jsonl"),
        "--lock-file",
        str(main_root / "schedule.lock"),
    ]
    rank = [
        python,
        str(ROOT / "scripts/run_post_impact_rank_adapter_shadow_follower.py"),
        "--capture-config",
        str(daily_schedule),
        "--rank-contract",
        str(daily_rank_contract),
        "--artifact-root",
        str(_resolve(config["rank_artifact_root"])),
        "--ledger",
        str(_resolve(config["rank_ledger"])),
        "--summary-root",
        str(_resolve(config["rank_summary_root"]) / target),
        "--event-log",
        str(rank_root / "events.jsonl"),
        "--lock-file",
        str(rank_root / "follower.lock"),
        "--poll-seconds",
        str(config.get("rank_poll_seconds", 5)),
        "--snapshot-wait-minutes",
        str(config.get("rank_snapshot_wait_minutes", 20)),
    ]
    if any("order" in Path(value).name.lower() for value in (*main, *rank)):
        raise ValueError("next-session schedule contains an order-capable script")
    return main, rank


def _wait_for_children(
    commands: tuple[list[str], list[str]], phase_log: Path
) -> tuple[int, int]:
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    with phase_log.open("a", encoding="utf-8") as handle:
        main = subprocess.Popen(
            commands[0], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
        rank = subprocess.Popen(
            commands[1], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
        rank_code = int(rank.wait())
        main_code = int(main.wait())
        handle.flush()
        os.fsync(handle.fileno())
    return main_code, rank_code


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_cycle_config(config)
    event_log = _resolve(args.event_log)
    phase_log = _resolve(args.phase_log)
    lock_path = _resolve(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("next-session shadow cycle is already running") from exc

        run_at = pd.Timestamp(config["run_at_kst"]).tz_convert(KST)
        now = pd.Timestamp.now(tz=KST)
        if now < run_at:
            time.sleep(float((run_at - now).total_seconds()))
        deadline = pd.Timestamp(config["state_ready_deadline_kst"]).tz_convert(KST)
        state_path = _resolve(config["forward_state"])
        state = load_ready_forward_state(
            state_path, context_session=str(config["context_session"])
        )
        while state is None and pd.Timestamp.now(tz=KST) <= deadline:
            time.sleep(float(config["state_poll_seconds"]))
            state = load_ready_forward_state(
                state_path, context_session=str(config["context_session"])
            )
        if state is None:
            raise RuntimeError("forward state did not mature before the frozen deadline")
        _append_event(
            event_log,
            {
                "event": "forward_state_ready",
                "context_session": config["context_session"],
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )

        daily = json.loads(_resolve(config["daily_config"]).read_text(encoding="utf-8"))
        if daily.get("safety", {}).get("live_orders_allowed") is not False:
            raise ValueError("daily config is not zero-order")
        expected_checkpoint = str(daily["jepa"]["checkpoint_sha256"])
        checkpoint = _resolve(daily["jepa"]["model_dir"]) / "graph_jepa_real.pt"
        if file_sha256(checkpoint) != expected_checkpoint:
            raise ValueError("daily JEPA checkpoint hash changed")

        cache_dir = _resolve(config["prospective_stale_cache_dir"])
        if cache_dir.exists():
            if not (cache_dir / "manifest.json").is_file():
                raise FileExistsError("incomplete prospective cache already exists")
        else:
            result = _run_command(
                build_prospective_cache_command(config, state, daily), phase_log
            )
            _append_event(
                event_log,
                {
                    "event": "prospective_cache_build_complete",
                    "result": result,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "live_orders_allowed": False,
                    "broker_order_calls_executed": 0,
                },
            )
            if result["returncode"] != 0:
                raise RuntimeError("prospective stale cache build failed")
        cache_sha = validate_prospective_cache(
            cache_dir,
            context_session=str(config["context_session"]),
            target_session=str(config["target_session"]),
        )

        created_at = datetime.now(timezone.utc).astimezone(KST).isoformat()
        rank_contract_path = _resolve(config["daily_rank_contract_output"])
        if rank_contract_path.exists():
            rank_contract = load_rank_contract(rank_contract_path)
            if rank_contract.get("daily_session") != config["target_session"]:
                raise ValueError("existing daily rank contract has a different session")
        else:
            rank_contract = build_daily_rank_contract(
                config,
                state,
                prospective_cache_sha256=cache_sha,
                created_at=created_at,
            )
            _write_immutable_json(rank_contract_path, rank_contract)
            load_rank_contract(rank_contract_path)

        live_schedule_path = _resolve(config["daily_live_schedule_output"])
        if live_schedule_path.exists():
            live_schedule = json.loads(
                live_schedule_path.read_text(encoding="utf-8")
            )
        else:
            live_schedule = build_daily_live_schedule(
                config, state, created_at=created_at
            )
            _write_immutable_json(live_schedule_path, live_schedule)
        validate_schedule_config(live_schedule)
        _append_event(
            event_log,
            {
                "event": "daily_contracts_frozen",
                "rank_contract_sha256": file_sha256(rank_contract_path),
                "live_schedule_sha256": file_sha256(live_schedule_path),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )

        codes = _wait_for_children(
            _schedule_commands(config, rank_contract_path, live_schedule_path),
            phase_log,
        )
        _append_event(
            event_log,
            {
                "event": "next_session_schedules_complete",
                "main_returncode": codes[0],
                "rank_returncode": codes[1],
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )
        return 0 if codes == (0, 0) else 1


if __name__ == "__main__":
    sys.exit(main())
