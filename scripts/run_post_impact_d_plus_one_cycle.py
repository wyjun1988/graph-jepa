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

from stock_v2.kiwoom_minute import KST
from stock_v2.prospective_ledger import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for D+1, run the resumable zero-order forward release, reconcile "
            "precommitted predictions, and audit the frozen prospective gate."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--phase-log", required=True)
    parser.add_argument("--lock-file", required=True)
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("role") != (
        "post_impact_d_plus_one_zero_order_cycle"
    ):
        raise ValueError("invalid D+1 zero-order config")
    if config.get("live_orders_allowed") is not False:
        raise ValueError("D+1 cycle permits live orders")
    if config.get("broker_order_calls_allowed") is not False:
        raise ValueError("D+1 cycle permits broker order calls")
    session = pd.Timestamp(str(config["session"])).normalize()
    run_at = pd.Timestamp(str(config["run_at_kst"]))
    if run_at.tzinfo is None or run_at.tz_convert(KST).normalize().tz_localize(None) <= session:
        raise ValueError("D+1 cycle run time is not after the target session")
    for name in (
        "python",
        "daily_config",
        "daily_state",
        "ledger",
        "artifact_root",
        "gate_contract",
    ):
        if not _resolve(config[name]).exists():
            raise FileNotFoundError(f"D+1 cycle input is missing: {name}")
    if file_sha256(_resolve(config["gate_contract"])) != str(
        config.get("gate_contract_sha256")
    ):
        raise ValueError("D+1 prospective gate contract changed")
    prior = tuple(config.get("prior_reconciliation_dirs") or ())
    if len(set(map(str, prior))) != len(prior):
        raise ValueError("D+1 prior reconciliation directories are duplicated")
    if str(config["reconciliation_output_dir"]) in set(map(str, prior)):
        raise ValueError("D+1 current reconciliation is also listed as prior")

    rank = config.get("rank_adapter")
    if rank is None:
        return
    if not isinstance(rank, Mapping) or rank.get("enabled") is not True:
        raise ValueError("D+1 rank-adapter section is invalid")
    if rank.get("live_orders_allowed") is not False:
        raise ValueError("D+1 rank-adapter section permits live orders")
    if rank.get("broker_order_calls_allowed") is not False:
        raise ValueError("D+1 rank-adapter section permits broker order calls")
    for name in (
        "rank_contract",
        "ledger",
        "artifact_root",
        "gate_contract",
        "reconciliation_output_dir",
        "gate_output",
    ):
        if not str(rank.get(name) or ""):
            raise ValueError(f"D+1 rank-adapter field is missing: {name}")
    rank_gate = _resolve(rank["gate_contract"])
    if not rank_gate.is_file():
        raise FileNotFoundError("D+1 rank-adapter gate contract is missing")
    if file_sha256(rank_gate) != str(rank.get("gate_contract_sha256")):
        raise ValueError("D+1 rank-adapter gate contract changed")
    rank_prior = tuple(rank.get("prior_reconciliation_dirs") or ())
    if len(set(map(str, rank_prior))) != len(rank_prior):
        raise ValueError("D+1 rank prior reconciliation directories are duplicated")
    if str(rank["reconciliation_output_dir"]) in set(map(str, rank_prior)):
        raise ValueError("D+1 current rank reconciliation is also listed as prior")


def _gate_command(
    *,
    python: str,
    auditor: Path,
    contract: Path,
    prior_reconciliations: Sequence[object],
    current_reconciliation: Path,
    output: Path,
) -> list[str]:
    command = [python, str(auditor), "--contract", str(contract)]
    for value in (*prior_reconciliations, current_reconciliation):
        path = _resolve(value)
        if path != current_reconciliation and not (path / "summary.json").is_file():
            raise FileNotFoundError(f"prior D+1 reconciliation is missing: {path}")
        command.extend(["--reconciliation-dir", str(path)])
    command.extend(["--output", str(output)])
    return command


def build_commands(
    config: Mapping[str, Any],
    *,
    day_release_dir: Path | None,
) -> dict[str, list[str]]:
    python = str(_resolve(config["python"]))
    commands = {
        "daily_forward": [
            python,
            str(ROOT / "scripts/run_post_impact_forward_daily.py"),
            "--config",
            str(_resolve(config["daily_config"])),
            "--state",
            str(_resolve(config["daily_state"])),
            "--session",
            str(config["session"]),
        ]
    }
    if day_release_dir is not None:
        reconciliation = _resolve(config["reconciliation_output_dir"])
        commands["reconcile"] = [
            python,
            str(ROOT / "scripts/reconcile_post_impact_prospective_ledger.py"),
            "--ledger",
            str(_resolve(config["ledger"])),
            "--artifact-root",
            str(_resolve(config["artifact_root"])),
            "--day-release-dir",
            str(day_release_dir),
            "--session",
            str(config["session"]),
            "--output-dir",
            str(reconciliation),
        ]
        commands["gate"] = _gate_command(
            python=python,
            auditor=ROOT / "scripts/audit_post_impact_prospective_ledger_gate.py",
            contract=_resolve(config["gate_contract"]),
            prior_reconciliations=tuple(
                config.get("prior_reconciliation_dirs") or ()
            ),
            current_reconciliation=reconciliation,
            output=_resolve(config["gate_output"]),
        )
        rank = config.get("rank_adapter")
        if isinstance(rank, Mapping) and rank.get("enabled") is True:
            rank_contract = _resolve(rank["rank_contract"])
            rank_ledger = _resolve(rank["ledger"])
            rank_artifacts = _resolve(rank["artifact_root"])
            for name, path in (
                ("rank contract", rank_contract),
                ("rank ledger", rank_ledger),
                ("rank artifact root", rank_artifacts),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"D+1 {name} is not ready: {path}")
            rank_reconciliation = _resolve(rank["reconciliation_output_dir"])
            commands["rank_reconcile"] = [
                python,
                str(ROOT / "scripts/reconcile_post_impact_rank_adapter_ledger.py"),
                "--rank-contract",
                str(rank_contract),
                "--ledger",
                str(rank_ledger),
                "--artifact-root",
                str(rank_artifacts),
                "--day-release-dir",
                str(day_release_dir),
                "--session",
                str(config["session"]),
                "--output-dir",
                str(rank_reconciliation),
            ]
            commands["rank_gate"] = _gate_command(
                python=python,
                auditor=ROOT
                / "scripts/audit_post_impact_rank_adapter_prospective_gate.py",
                contract=_resolve(rank["gate_contract"]),
                prior_reconciliations=tuple(
                    rank.get("prior_reconciliation_dirs") or ()
                ),
                current_reconciliation=rank_reconciliation,
                output=_resolve(rank["gate_output"]),
            )
    for command in commands.values():
        if "order" in Path(command[1]).name.lower():
            raise ValueError("D+1 cycle includes an order-capable script")
    return commands


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("D+1 event append made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run(command: Sequence[str], phase_log: Path) -> dict[str, Any]:
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with phase_log.open("ab", buffering=0) as output:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "returncode": int(completed.returncode),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _advanced_day_release(config: Mapping[str, Any]) -> Path | None:
    state = json.loads(_resolve(config["daily_state"]).read_text(encoding="utf-8"))
    if str(state.get("latest_session")) != str(config["session"]):
        return None
    path = _resolve(state["artifacts"]["day_release"]["root"])
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError("advanced D+1 day release is missing")
    return path


def main() -> int:
    args = parse_args()
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    event_log = _resolve(args.event_log)
    phase_log = _resolve(args.phase_log)
    lock_path = _resolve(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("D+1 zero-order cycle is already running") from exc
        run_at = pd.Timestamp(str(config["run_at_kst"])).tz_convert(KST)
        now = pd.Timestamp.now(tz=KST)
        if now < run_at:
            if args.no_wait:
                raise RuntimeError("D+1 cycle is not mature yet")
            time.sleep(float((run_at - now).total_seconds()))
        _append_event(
            event_log,
            {
                "event": "cycle_started",
                "session": config["session"],
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )

        day_release = _advanced_day_release(config)
        if day_release is None:
            command = build_commands(config, day_release_dir=None)["daily_forward"]
            result = _run(command, phase_log)
            _append_event(
                event_log,
                {
                    "event": "daily_forward_complete",
                    "result": result,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "live_orders_allowed": False,
                    "broker_order_calls_executed": 0,
                },
            )
            if result["returncode"] != 0:
                raise RuntimeError("D+1 daily forward pipeline failed")
            day_release = _advanced_day_release(config)
            if day_release is None:
                raise RuntimeError("D+1 daily state did not advance")

        commands = build_commands(config, day_release_dir=day_release)
        phase_order = ("reconcile", "rank_reconcile", "gate", "rank_gate")
        for name in (value for value in phase_order if value in commands):
            result = _run(commands[name], phase_log)
            _append_event(
                event_log,
                {
                    "event": f"{name}_complete",
                    "result": result,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "live_orders_allowed": False,
                    "broker_order_calls_executed": 0,
                },
            )
            if result["returncode"] != 0:
                raise RuntimeError(f"D+1 {name} failed")
        _append_event(
            event_log,
            {
                "event": "cycle_complete",
                "session": config["session"],
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
