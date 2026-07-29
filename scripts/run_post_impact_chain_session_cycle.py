"""Continue the frozen post-impact prospective chain beyond its first session.

The pinned next-session runner (`run_post_impact_next_session_shadow_cycle.py`)
can only target the first primary session because it requires the base rank
contract's first primary session to equal the target. The prospective gate
auditor separately requires every daily rank contract's parent scope contract
to hash-match the frozen v1 scope contract, so per-day derived base contracts
would invalidate forward evidence.

This operational runner closes that gap without touching any pinned source or
frozen contract. It reuses the pinned building blocks unchanged and builds each
later session's daily rank contract directly from the frozen v1 scope contract,
keeping the frozen `first_primary_session` and advancing only `daily_session`.
The follower accepts any session on or after the frozen first primary session,
and the gate auditor sees the exact pinned parent SHA-256, unchanged clocks,
and unchanged checkpoints.

Subcommands:
  emit  Write the immutable chain-cycle config and the matching D+1 config for
        one later session, then validate both against the frozen validators.
  run   Wait until the frozen run time, then freeze the session's daily
        contracts and launch the pinned capture schedule and rank follower.

Zero-order guarantees are inherited: every generated config hard-codes
`live_orders_allowed=false` and `broker_order_calls_allowed=false`, and the
pinned child commands refuse order-capable scripts.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_post_impact_d_plus_one_cycle import (
    validate_config as validate_d_plus_one_config,
)
from scripts.run_post_impact_live_prospective_schedule import (
    _append_event,
    validate_schedule_config,
)
from scripts.run_post_impact_next_session_shadow_cycle import (
    _portable,
    _resolve,
    _run_command,
    _schedule_commands,
    _verified_state_artifact,
    _wait_for_children,
    _write_immutable_json,
    build_daily_live_schedule,
    build_prospective_cache_command,
    load_ready_forward_state,
    validate_prospective_cache,
)
from scripts.run_post_impact_rank_adapter_live_shadow import (
    load_rank_contract,
    prospective_scope,
)
from stock_v2.kiwoom_minute import KST
from stock_v2.prospective_ledger import file_sha256

ROLE = "post_impact_chain_session_zero_order_cycle"


def _session(value: object) -> str:
    return str(pd.Timestamp(str(value)).date())


def _compact(session: str) -> str:
    return session.replace("-", "")


def validate_chain_cycle_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("role") != ROLE:
        raise ValueError("invalid chain-session cycle config")
    if config.get("live_orders_allowed") is not False:
        raise ValueError("chain-session cycle permits live orders")
    if config.get("broker_order_calls_allowed") is not False:
        raise ValueError("chain-session cycle permits broker order calls")
    context = pd.Timestamp(_session(config.get("context_session")))
    target = pd.Timestamp(_session(config.get("target_session")))
    if target <= context or target.weekday() >= 5:
        raise ValueError("chain-session target must be a later weekday")
    run_at = pd.Timestamp(config.get("run_at_kst"))
    deadline = pd.Timestamp(config.get("state_ready_deadline_kst"))
    if run_at.tzinfo is None or deadline.tzinfo is None:
        raise ValueError("chain-session cycle times must be timezone-aware")
    run_at = run_at.tz_convert(KST)
    deadline = deadline.tz_convert(KST)
    if str(run_at.date()) != str(target.date()) or not run_at < deadline:
        raise ValueError("chain-session run window is invalid")
    if float(config.get("state_poll_seconds", 0.0)) < 5.0:
        raise ValueError("chain-session state polling is too frequent")
    for name in (
        "python",
        "forward_state",
        "daily_config",
        "base_live_schedule",
        "base_rank_contract",
    ):
        if not _resolve(config.get(name)).exists():
            raise FileNotFoundError(f"chain-session static input is missing: {name}")
    clocks = tuple(str(value) for value in config.get("capture_clocks_kst") or ())
    if list(clocks) != sorted(set(clocks)):
        raise ValueError("chain-session capture clocks must be unique and sorted")
    base_rank = load_rank_contract(_resolve(config["base_rank_contract"]))
    first_session, rank_minutes = prospective_scope(base_rank)
    if _session(target) <= first_session:
        raise ValueError(
            "chain-session target must be after the frozen first primary session; "
            "use the pinned next-session runner for the first session"
        )
    capture_minutes = {int(clock[:2]) * 60 + int(clock[3:]) for clock in clocks}
    if any(minute not in capture_minutes for minute in rank_minutes):
        raise ValueError("capture schedule omits a frozen rank-adapter clock")


def build_chain_daily_rank_contract(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    prospective_cache_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    base_path = _resolve(config["base_rank_contract"])
    child = json.loads(base_path.read_text(encoding="utf-8"))
    target = _session(config["target_session"])
    day_release, day_sha = _verified_state_artifact(state, "day_release")
    lifecycle, lifecycle_sha = _verified_state_artifact(state, "lifecycle")
    cache = _resolve(config["prospective_stale_cache_dir"])
    child["contract_id"] = (
        f"post_impact_rank_adapter_live_shadow_{_compact(target)}"
    )
    child["created_at"] = created_at
    child["daily_session"] = target
    # Unlike the pinned first-session builder, the frozen scope is preserved:
    # the gate auditor pins the parent scope SHA-256 and the follower accepts
    # any session on or after `first_primary_session`.
    child["parent_scope_contract"] = {
        "path": _portable(base_path),
        "sha256": file_sha256(base_path),
    }
    child["chain_extension"] = {
        "builder": "scripts/run_post_impact_chain_session_cycle.py",
        "reason": (
            "operational continuation of the frozen prospective scope to a "
            "later session; no model, threshold, clock, horizon, or session "
            "minimum changed"
        ),
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


def next_weekday(session: str) -> str:
    stamp = pd.Timestamp(session) + pd.offsets.Day(1)
    while stamp.weekday() >= 5:
        stamp += pd.offsets.Day(1)
    return str(stamp.date())


def build_chain_cycle_config(
    template: Mapping[str, Any], *, context_session: str, target_session: str
) -> dict[str, Any]:
    target = _session(target_session)
    compact = _compact(target)
    config = dict(template)
    config["role"] = ROLE
    config["context_session"] = _session(context_session)
    config["target_session"] = target
    config["run_at_kst"] = f"{target}T06:05:00+09:00"
    config["state_ready_deadline_kst"] = f"{target}T08:30:00+09:00"
    config["prospective_stale_cache_dir"] = (
        f"data/intraday_stale_jepa_cache/fold4_seed17_prospective_{compact}_v1"
    )
    config["daily_live_schedule_output"] = (
        f"configs/frozen/post_impact_live_prospective_{compact}.json"
    )
    config["daily_rank_contract_output"] = (
        f"configs/frozen/post_impact_rank_adapter_live_shadow_{compact}.json"
    )
    config["main_schedule_ops_root"] = f"ops/prospective_live/schedule_{compact}"
    config["rank_schedule_ops_root"] = (
        f"ops/prospective_live/rank_adapter_schedule_{compact}"
    )
    config["live_orders_allowed"] = False
    config["broker_order_calls_allowed"] = False
    return config


def build_chain_d_plus_one_config(
    previous: Mapping[str, Any], *, session: str, run_date: str
) -> dict[str, Any]:
    target = _session(session)
    compact = _compact(target)
    run_day = _session(run_date)
    if pd.Timestamp(run_day) <= pd.Timestamp(target):
        raise ValueError("D+1 run date must be after the session")
    config = json.loads(json.dumps(previous))
    previous_priors = [str(value) for value in previous.get(
        "prior_reconciliation_dirs"
    ) or []]
    previous_dir = str(previous["reconciliation_output_dir"])
    config["session"] = target
    config["run_at_kst"] = f"{run_day}T06:00:00+09:00"
    config["gate_output"] = (
        f"reports/post_impact_prospective_ledger_gate_v1/through_{compact}.json"
    )
    config["reconciliation_output_dir"] = (
        f"reports/post_impact_prospective_reconciliation/{compact}"
    )
    config["prior_reconciliation_dirs"] = previous_priors + [previous_dir]
    rank = dict(config["rank_adapter"])
    rank_previous_priors = [str(value) for value in rank.get(
        "prior_reconciliation_dirs"
    ) or []]
    rank_previous_dir = str(rank["reconciliation_output_dir"])
    rank["gate_output"] = (
        f"reports/post_impact_rank_adapter_prospective_gate_v1/through_{compact}.json"
    )
    rank["rank_contract"] = (
        f"configs/frozen/post_impact_rank_adapter_live_shadow_{compact}.json"
    )
    rank["reconciliation_output_dir"] = (
        f"reports/post_impact_rank_adapter_prospective_reconciliation/{compact}"
    )
    rank["prior_reconciliation_dirs"] = rank_previous_priors + [rank_previous_dir]
    rank["live_orders_allowed"] = False
    rank["broker_order_calls_allowed"] = False
    config["rank_adapter"] = rank
    config["live_orders_allowed"] = False
    config["broker_order_calls_allowed"] = False
    return config


def emit(args: argparse.Namespace) -> int:
    template = json.loads(_resolve(args.template).read_text(encoding="utf-8"))
    cycle = build_chain_cycle_config(
        template,
        context_session=args.context_session,
        target_session=args.target_session,
    )
    validate_chain_cycle_config(cycle)
    target = _session(args.target_session)
    compact = _compact(target)
    cycle_path = _resolve(f"configs/post_impact_chain_session_{compact}.json")
    _write_immutable_json(cycle_path, cycle)
    validate_chain_cycle_config(
        json.loads(cycle_path.read_text(encoding="utf-8"))
    )

    previous_d1 = json.loads(
        _resolve(args.previous_d_plus_one).read_text(encoding="utf-8")
    )
    run_date = args.d_plus_one_run or next_weekday(target)
    d1 = build_chain_d_plus_one_config(previous_d1, session=target, run_date=run_date)
    d1_path = _resolve(
        f"configs/post_impact_d_plus_one_{_compact(_session(run_date))}.json"
    )
    _write_immutable_json(d1_path, d1)
    validate_d_plus_one_config(json.loads(d1_path.read_text(encoding="utf-8")))

    print(json.dumps({
        "chain_cycle_config": _portable(cycle_path),
        "chain_cycle_config_sha256": file_sha256(cycle_path),
        "d_plus_one_config": _portable(d1_path),
        "d_plus_one_config_sha256": file_sha256(d1_path),
        "target_session": target,
        "d_plus_one_run_date": _session(run_date),
        "live_orders_allowed": False,
    }, indent=2, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_chain_cycle_config(config)
    event_log = _resolve(args.event_log)
    phase_log = _resolve(args.phase_log)
    lock_path = _resolve(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("chain-session cycle is already running") from exc

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
            rank_contract = build_chain_daily_rank_contract(
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
                "event": "chain_session_schedules_complete",
                "main_returncode": codes[0],
                "rank_returncode": codes[1],
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )
        return 0 if codes == (0, 0) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--target-session", required=True)
    emit_parser.add_argument("--context-session", required=True)
    emit_parser.add_argument(
        "--template",
        default="configs/post_impact_next_session_shadow_20260717.json",
    )
    emit_parser.add_argument(
        "--previous-d-plus-one",
        default="configs/post_impact_d_plus_one_20260720.json",
    )
    emit_parser.add_argument("--d-plus-one-run", default=None)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--event-log", required=True)
    run_parser.add_argument("--phase-log", required=True)
    run_parser.add_argument("--lock-file", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "emit":
        return emit(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
