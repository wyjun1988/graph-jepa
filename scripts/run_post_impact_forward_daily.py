from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
KST = ZoneInfo("Asia/Seoul")
ALLOWED_SCRIPTS = {
    "assemble_forward_intraday_inputs.py",
    "assemble_intraday_trajectory_days.py",
    "audit_intraday_day_release.py",
    "audit_intraday_trajectory_release.py",
    "audit_kiwoom_minute_collection.py",
    "audit_kiwoom_ohlcv_collection.py",
    "audit_post_impact_clock_gated_forward.py",
    "audit_stale_jepa_cache.py",
    "backfill_kiwoom_minute.py",
    "backfill_kiwoom_ohlcv.py",
    "build_intraday_trajectory_release.py",
    "build_stale_jepa_rollout_cache.py",
    "evaluate_post_impact_adaptive_events.py",
    "extend_lifecycle_release.py",
    "finalize_intraday_trajectory_release.py",
    "finalize_kiwoom_minute_collection.py",
    "finalize_kiwoom_ohlcv_collection.py",
    "merge_intraday_day_releases.py",
    "merge_stale_jepa_rollout_caches.py",
}
MODEL_NAMES = ("direct", "state", "latent", "latent_only_placebo")
ALLOWED_FORWARD_DECISIONS = {
    "insufficient_forward_evidence_accumulating",
    "eligible_for_longer_read_only_shadow_only",
    "clock_gated_latent_not_confirmed",
}


@dataclass(frozen=True)
class Phase:
    name: str
    command: tuple[str, ...]
    outputs: tuple[str, ...]
    allow_existing_outputs: bool = False


@dataclass(frozen=True)
class DailyPaths:
    data_root: str
    report_root: str
    control_root: str
    minute_release: str
    daily_raw_release: str
    daily_adjusted_release: str
    lifecycle_root: str
    forward_inputs: str
    trajectory_release: str
    incremental_day_release: str
    merged_day_release: str
    incremental_stale_cache: str
    merged_stale_cache: str
    model_report_dir: str
    forward_audit_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the append-only, zero-order post-impact forward evaluation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_safety_claims(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"{label} does not explicitly prohibit live orders")
    if payload.get("broker_order_calls_allowed", False) is not False:
        raise ValueError(f"{label} permits broker order calls")
    if int(payload.get("orders_sent", 0)) != 0:
        raise ValueError(f"{label} records non-zero orders")


def validate_session_contract(
    latest_session: str,
    requested_session: str,
    *,
    today: str,
) -> None:
    latest = datetime.fromisoformat(latest_session).date()
    requested = datetime.fromisoformat(requested_session).date()
    current = datetime.fromisoformat(today).date()
    if requested <= latest:
        raise ValueError("forward session must strictly follow the current state")
    if requested >= current:
        raise ValueError("forward session must be completed before today in KST")


def _verify_pin(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = resolve_path(str(record.get("path") or ""))
    expected = str(record.get("sha256") or "")
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"pinned artifact changed: {label}")
    return load_json(path)


def validate_config_and_state(
    config_path: Path,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    if config.get("schema_version") != 1 or config.get("role") != (
        "post_impact_forward_daily_config"
    ):
        raise ValueError("invalid forward daily config")
    if state.get("schema_version") != 1 or state.get("role") != (
        "post_impact_forward_daily_state"
    ):
        raise ValueError("invalid forward daily state")
    validate_safety_claims(config.get("safety", {}), "config safety")
    validate_safety_claims(state, "forward state")
    if state.get("config_sha256") != file_sha256(config_path):
        raise ValueError("forward state config hash changed")
    python = resolve_path(str(config["python"]))
    env_file = resolve_path(str(config["env_file"]))
    if not python.is_file() or not env_file.is_file():
        raise ValueError("forward runtime or Kiwoom environment file is missing")

    universe = _verify_pin(config["universe"], "universe")
    if len(universe.get("universe", [])) != 500:
        raise ValueError("forward universe is not the frozen 500-stock axis")
    _verify_pin(config["timestamp_semantics_evidence"], "timestamp semantics")
    contract = _verify_pin(config["policy_contract"], "policy contract")
    validate_safety_claims(contract, "policy contract")
    if contract.get("role") != "post_impact_clock_gated_forward_shadow_contract":
        raise ValueError("unexpected forward policy contract")
    for name in MODEL_NAMES:
        configured = config["models"][name]
        expected = contract["models"][name]
        if configured != {
            "checkpoint": expected["checkpoint"],
            "checkpoint_sha256": expected["checkpoint_sha256"],
        }:
            raise ValueError(f"configured model differs from policy contract: {name}")
        checkpoint = resolve_path(configured["checkpoint"])
        if file_sha256(checkpoint) != configured["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint changed: {name}")
    jepa_checkpoint = resolve_path(
        Path(config["jepa"]["model_dir"]) / "graph_jepa_real.pt"
    )
    if file_sha256(jepa_checkpoint) != config["jepa"]["checkpoint_sha256"]:
        raise ValueError("frozen JEPA checkpoint changed")

    latest = str(state["latest_session"])
    artifacts = state["artifacts"]
    lifecycle = _verify_pin(artifacts["lifecycle"], "lifecycle manifest")
    forward_inputs = _verify_pin(
        artifacts["forward_inputs"], "forward input manifest"
    )
    day_release = _verify_pin(artifacts["day_release"], "day release manifest")
    stale_cache = _verify_pin(artifacts["stale_cache"], "stale cache manifest")
    if lifecycle.get("end") != latest:
        raise ValueError("lifecycle state date differs from latest session")
    if forward_inputs.get("incremental_date") != latest:
        raise ValueError("forward input state date differs from latest session")
    if day_release.get("last_date") != latest:
        raise ValueError("day release state date differs from latest session")
    if stale_cache.get("target_end") != latest:
        raise ValueError("stale cache state date differs from latest session")
    for label, payload in (
        ("forward inputs", forward_inputs),
        ("day release", day_release),
        ("stale cache", stale_cache),
    ):
        validate_safety_claims(payload, label)
    if stale_cache.get("strict_out_of_sample") is not True:
        raise ValueError("stale cache is not strict out-of-sample")

    report_hashes: dict[str, str] = {}
    for name in MODEL_NAMES:
        report = _verify_pin(state["reports"][name], f"previous report {name}")
        validate_safety_claims(report, f"previous report {name}")
        report_hashes[name] = state["reports"][name]["sha256"]
    audit = _verify_pin(state["forward_audit"], "previous forward audit")
    validate_safety_claims(audit, "previous forward audit")
    if audit.get("broker_order_calls_executed") != 0:
        raise ValueError("previous forward audit records broker calls")
    if audit.get("inputs", {}).get("reports") != report_hashes:
        raise ValueError("previous audit report hashes differ from state")


def _phase(
    python: str,
    name: str,
    script: str,
    arguments: Sequence[str],
    outputs: Sequence[str],
    *,
    allow_existing_outputs: bool = False,
) -> Phase:
    if script not in ALLOWED_SCRIPTS or "order" in script.lower():
        raise ValueError(f"script is not permitted in zero-order daily runs: {script}")
    command = (python, f"scripts/{script}", *tuple(str(value) for value in arguments))
    if any("live-order" in value.lower() for value in command):
        raise ValueError("daily phase contains an order-enabling argument")
    return Phase(
        name=name,
        command=command,
        outputs=tuple(outputs),
        allow_existing_outputs=allow_existing_outputs,
    )


def daily_paths(config: Mapping[str, Any], session: str) -> DailyPaths:
    compact = session.replace("-", "")
    data = Path(config["data_root"]) / compact
    report = Path(config["report_root"]) / compact
    control = Path(config["control_root"]) / compact
    return DailyPaths(
        data_root=data.as_posix(),
        report_root=report.as_posix(),
        control_root=control.as_posix(),
        minute_release=(data / "minute/release").as_posix(),
        daily_raw_release=(data / "daily/raw_release").as_posix(),
        daily_adjusted_release=(data / "daily/adjusted_release").as_posix(),
        lifecycle_root=(data / "lifecycle_release").as_posix(),
        forward_inputs=(data / "forward_inputs").as_posix(),
        trajectory_release=(data / "trajectory_increment").as_posix(),
        incremental_day_release=(data / "day_increment").as_posix(),
        merged_day_release=(data / "day_merged").as_posix(),
        incremental_stale_cache=(data / "stale_increment").as_posix(),
        merged_stale_cache=(data / "stale_merged").as_posix(),
        model_report_dir=(report / "model_reports").as_posix(),
        forward_audit_dir=(report / "forward_audit").as_posix(),
    )


def build_plan(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    session: str,
) -> tuple[DailyPaths, list[Phase]]:
    paths = daily_paths(config, session)
    python = str(config["python"])
    universe = str(config["universe"]["path"])
    semantics = str(config["timestamp_semantics_evidence"]["path"])
    contract = str(config["policy_contract"]["path"])
    previous = str(state["latest_session"])
    run_id = f"post-impact-forward-{session.replace('-', '')}"
    data = Path(paths.data_root)
    reports = Path(paths.report_root)
    minute_source = data / "minute/source"
    minute_raw_pages = data / "minute/raw_pages"
    minute_coverage = data / "minute/coverage.jsonl"
    daily_source = data / "daily/source"
    daily_raw_pages = data / "daily/raw_pages"
    daily_coverage = data / "daily/coverage.jsonl"
    minute_outputs = Path(paths.minute_release) / "outputs"
    raw_outputs = Path(paths.daily_raw_release) / "outputs"
    adjusted_outputs = Path(paths.daily_adjusted_release) / "outputs"
    lifecycle_manifest = Path(paths.lifecycle_root) / "lifecycle/manifest.json"
    base_forward_root = Path(state["artifacts"]["forward_inputs"]["root"])
    forward_manifest = Path(paths.forward_inputs) / "manifest.json"
    trajectory_manifest = Path(paths.trajectory_release) / "manifest.json"
    increment_day_manifest = Path(paths.incremental_day_release) / "manifest.json"
    merged_day_manifest = Path(paths.merged_day_release) / "manifest.json"
    increment_stale_manifest = Path(paths.incremental_stale_cache) / "manifest.json"
    merged_stale_manifest = Path(paths.merged_stale_cache) / "manifest.json"

    phases: list[Phase] = []

    def add(
        name: str,
        script: str,
        arguments: Sequence[str],
        outputs: Sequence[Path],
        *,
        allow_existing_outputs: bool = False,
    ) -> None:
        phases.append(
            _phase(
                python,
                name,
                script,
                arguments,
                [path.as_posix() for path in outputs],
                allow_existing_outputs=allow_existing_outputs,
            )
        )

    add(
        "minute_collect",
        "backfill_kiwoom_minute.py",
        (
            "--universe-manifest", universe,
            "--start", session,
            "--end", session,
            "--interval-minutes", "5",
            "--basis", "raw",
            "--output-format", "parquet",
            "--cache-dir", minute_source.as_posix(),
            "--raw-cache-dir", minute_raw_pages.as_posix(),
            "--coverage-output", minute_coverage.as_posix(),
            "--run-id", run_id,
            "--env-file", str(config["env_file"]),
            "--server", "real",
            "--sleep-sec", str(config["collection_sleep_seconds"]),
            "--resume",
        ),
        (minute_coverage,),
        allow_existing_outputs=True,
    )
    add(
        "minute_audit",
        "audit_kiwoom_minute_collection.py",
        (
            "--coverage", minute_coverage.as_posix(),
            "--universe-manifest", universe,
            "--raw-cache-dir", minute_raw_pages.as_posix(),
            "--run-id", run_id,
            "--start", session,
            "--end", session,
            "--interval-minutes", "5",
            "--basis", "raw",
            "--output", (reports / "minute_collection_audit.json").as_posix(),
        ),
        (reports / "minute_collection_audit.json",),
    )
    add(
        "minute_finalize",
        "finalize_kiwoom_minute_collection.py",
        (
            "--coverage", minute_coverage.as_posix(),
            "--universe-manifest", universe,
            "--run-id", run_id,
            "--start", session,
            "--end", session,
            "--interval-minutes", "5",
            "--basis", "raw",
            "--output-dir", paths.minute_release,
        ),
        (Path(paths.minute_release) / "manifest.json",),
    )
    add(
        "daily_collect",
        "backfill_kiwoom_ohlcv.py",
        (
            "--universe-manifest", universe,
            "--start", previous,
            "--end", session,
            "--basis", "both",
            "--cache-dir", daily_source.as_posix(),
            "--raw-cache-dir", daily_raw_pages.as_posix(),
            "--coverage-output", daily_coverage.as_posix(),
            "--run-id", run_id,
            "--env-file", str(config["env_file"]),
            "--server", "real",
            "--sleep-sec", str(config["collection_sleep_seconds"]),
            "--resume",
        ),
        (daily_coverage,),
        allow_existing_outputs=True,
    )
    for basis in ("raw", "adjusted"):
        audit_path = reports / f"daily_{basis}_collection_audit.json"
        release = (
            paths.daily_raw_release
            if basis == "raw"
            else paths.daily_adjusted_release
        )
        add(
            f"daily_{basis}_audit",
            "audit_kiwoom_ohlcv_collection.py",
            (
                "--coverage", daily_coverage.as_posix(),
                "--universe-manifest", universe,
                "--raw-cache-dir", daily_raw_pages.as_posix(),
                "--run-id", run_id,
                "--start", previous,
                "--end", session,
                "--basis", basis,
                "--output", audit_path.as_posix(),
            ),
            (audit_path,),
        )
        add(
            f"daily_{basis}_finalize",
            "finalize_kiwoom_ohlcv_collection.py",
            (
                "--coverage", daily_coverage.as_posix(),
                "--universe-manifest", universe,
                "--run-id", run_id,
                "--start", previous,
                "--end", session,
                "--basis", basis,
                "--output-dir", release,
            ),
            (Path(release) / "manifest.json",),
        )
    add(
        "lifecycle_extend",
        "extend_lifecycle_release.py",
        (
            "--base-manifest", str(state["artifacts"]["lifecycle"]["path"]),
            "--universe-manifest", universe,
            "--incremental-raw-dir", raw_outputs.as_posix(),
            "--incremental-adjusted-dir", adjusted_outputs.as_posix(),
            "--incremental-start", previous,
            "--end", session,
            "--output-root", paths.lifecycle_root,
        ),
        (lifecycle_manifest, Path(paths.lifecycle_root) / "lifecycle_audit.json"),
    )
    add(
        "forward_inputs_assemble",
        "assemble_forward_intraday_inputs.py",
        (
            "--universe-manifest", universe,
            "--base-minute-dir", (base_forward_root / "minute/5min/raw").as_posix(),
            "--incremental-minute-dir", minute_outputs.as_posix(),
            "--base-daily-dir", (base_forward_root / "daily").as_posix(),
            "--incremental-daily-raw-dir", raw_outputs.as_posix(),
            "--incremental-daily-start", previous,
            "--base-start", str(config["base_start"]),
            "--base-end", previous,
            "--incremental-date", session,
            "--run-id", run_id,
            "--output-dir", paths.forward_inputs,
        ),
        (forward_manifest,),
    )
    add(
        "trajectory_build_increment",
        "build_intraday_trajectory_release.py",
        (
            "--coverage", (Path(paths.forward_inputs) / "coverage.jsonl").as_posix(),
            "--run-id", run_id,
            "--universe-manifest", universe,
            "--daily-ohlcv-dir", (Path(paths.forward_inputs) / "daily").as_posix(),
            "--start", str(config["base_start"]),
            "--end", session,
            "--output-start", session,
            "--output-end", session,
            "--context-sessions", str(config["trajectory"]["context_sessions"]),
            "--interval-minutes", "5",
            "--timestamp-semantics", "start",
            "--timestamp-semantics-evidence", semantics,
            "--decision-start", "09:15",
            "--decision-end", "15:15",
            "--horizons-minutes", "5,15,30,60",
            "--rolling-window", "20",
            "--min-history", "10",
            "--minimum-ticker-files", "400",
            "--minimum-snapshots-per-ticker", "60",
            "--output-dir", paths.trajectory_release,
        ),
        (Path(paths.trajectory_release) / "timestamp_index.npz",),
    )
    add(
        "trajectory_finalize",
        "finalize_intraday_trajectory_release.py",
        ("--release-dir", paths.trajectory_release),
        (trajectory_manifest,),
        allow_existing_outputs=True,
    )
    trajectory_audit = reports / "trajectory_increment_audit.json"
    add(
        "trajectory_audit",
        "audit_intraday_trajectory_release.py",
        (
            "--release-dir", paths.trajectory_release,
            "--minimum-shards", "400",
            "--require-input-files",
            "--output", trajectory_audit.as_posix(),
        ),
        (trajectory_audit,),
    )
    add(
        "day_assemble_increment",
        "assemble_intraday_trajectory_days.py",
        (
            "--release-dir", paths.trajectory_release,
            "--minimum-nodes-per-timestamp", "400",
            "--minimum-timestamps-per-day", "60",
            "--minimum-days", "1",
            "--systemic-min-nodes", "100",
            "--output-dir", paths.incremental_day_release,
        ),
        (increment_day_manifest,),
    )
    increment_day_audit = reports / "day_increment_audit.json"
    add(
        "day_audit_increment",
        "audit_intraday_day_release.py",
        (
            "--release-dir", paths.incremental_day_release,
            "--source-release-dir", paths.trajectory_release,
            "--minimum-days", "1",
            "--output", increment_day_audit.as_posix(),
        ),
        (increment_day_audit,),
    )
    add(
        "day_merge",
        "merge_intraday_day_releases.py",
        (
            "--base-dir", str(state["artifacts"]["day_release"]["root"]),
            "--incremental-dir", paths.incremental_day_release,
            "--output-dir", paths.merged_day_release,
        ),
        (merged_day_manifest,),
    )
    merged_day_audit = reports / "day_merged_audit.json"
    add(
        "day_audit_merged",
        "audit_intraday_day_release.py",
        (
            "--release-dir", paths.merged_day_release,
            "--minimum-days", "1",
            "--output", merged_day_audit.as_posix(),
        ),
        (merged_day_audit,),
    )
    add(
        "stale_build_increment",
        "build_stale_jepa_rollout_cache.py",
        (
            "--model-dir", str(config["jepa"]["model_dir"]),
            "--target-start", session,
            "--target-end", session,
            "--device", "mps",
            "--batch-size", str(config["jepa"]["batch_size"]),
            "--edge-cache-workers", str(config["jepa"]["edge_cache_workers"]),
            "--cache-dir", (Path(paths.lifecycle_root) / "lifecycle/ohlcv").as_posix(),
            "--output-dir", paths.incremental_stale_cache,
        ),
        (increment_stale_manifest,),
    )
    increment_stale_audit = reports / "stale_increment_audit.json"
    add(
        "stale_audit_increment",
        "audit_stale_jepa_cache.py",
        (
            "--cache-dir", paths.incremental_stale_cache,
            "--output", increment_stale_audit.as_posix(),
            "--require-causal-stock-graph",
        ),
        (increment_stale_audit,),
    )
    add(
        "stale_merge",
        "merge_stale_jepa_rollout_caches.py",
        (
            "--base-dir", str(state["artifacts"]["stale_cache"]["root"]),
            "--incremental-dir", paths.incremental_stale_cache,
            "--output-dir", paths.merged_stale_cache,
        ),
        (merged_stale_manifest,),
    )
    merged_stale_audit = reports / "stale_merged_audit.json"
    add(
        "stale_audit_merged",
        "audit_stale_jepa_cache.py",
        (
            "--cache-dir", paths.merged_stale_cache,
            "--output", merged_stale_audit.as_posix(),
            "--require-causal-stock-graph",
        ),
        (merged_stale_audit,),
    )
    for name in MODEL_NAMES:
        output = Path(paths.model_report_dir) / f"{name}.json"
        add(
            f"evaluate_{name}",
            "evaluate_post_impact_adaptive_events.py",
            (
                "--day-release-dir", paths.merged_day_release,
                "--stale-cache-dir", paths.merged_stale_cache,
                "--checkpoint", str(config["models"][name]["checkpoint"]),
                "--reference-summary", str(state["reports"][name]["path"]),
                "--require-reference-parity",
                "--train-end", str(config["evaluation"]["train_end"]),
                "--validation-end", str(config["evaluation"]["validation_end"]),
                "--test-end", session,
                "--quantile", str(config["evaluation"]["quantile"]),
                "--window-sessions", str(config["evaluation"]["window_sessions"]),
                "--minimum-history", str(config["evaluation"]["minimum_history"]),
                "--batch-days", str(config["evaluation"]["batch_days"]),
                "--device", "mps",
                "--amp-dtype", "none",
                "--cache-day-shards",
                "--output", output.as_posix(),
            ),
            (output,),
        )
    forward_summary = Path(paths.forward_audit_dir) / "summary.json"
    add(
        "forward_audit",
        "audit_post_impact_clock_gated_forward.py",
        (
            "--contract", contract,
            "--report-dir", paths.model_report_dir,
            "--output-dir", paths.forward_audit_dir,
        ),
        (forward_summary,),
    )
    return paths, phases


def _phase_marker_payload(phase: Phase) -> dict[str, Any]:
    outputs = []
    for value in phase.outputs:
        path = resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"phase output is missing: {path}")
        outputs.append(
            {"path": value, "sha256": file_sha256(path), "bytes": path.stat().st_size}
        )
    return {
        "schema_version": 1,
        "phase": phase.name,
        "command": list(phase.command),
        "command_sha256": canonical_sha256(list(phase.command)),
        "outputs": outputs,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "broker_order_calls_executed": 0,
        "live_orders_allowed": False,
    }


def _validate_phase_marker(path: Path, phase: Phase) -> None:
    marker = load_json(path)
    validate_safety_claims(marker, f"phase marker {phase.name}")
    if marker.get("phase") != phase.name or marker.get("command_sha256") != (
        canonical_sha256(list(phase.command))
    ):
        raise ValueError(f"phase resume contract changed: {phase.name}")
    expected = {value for value in phase.outputs}
    records = marker.get("outputs") or []
    if {str(record.get("path")) for record in records} != expected:
        raise ValueError(f"phase output contract changed: {phase.name}")
    for record in records:
        output = resolve_path(record["path"])
        if (
            not output.is_file()
            or output.stat().st_size != int(record["bytes"])
            or file_sha256(output) != record["sha256"]
        ):
            raise ValueError(f"completed phase output changed: {phase.name}")


def run_phase(phase: Phase, control_root: Path) -> None:
    marker = control_root / "phases" / f"{phase.name}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        _validate_phase_marker(marker, phase)
        print(f"phase={phase.name} status=verified_resume", flush=True)
        return
    existing = [value for value in phase.outputs if resolve_path(value).exists()]
    if existing and not phase.allow_existing_outputs:
        raise FileExistsError(
            f"untracked outputs exist for phase {phase.name}: {existing}"
        )
    log_path = control_root / "logs" / f"{phase.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write("command=" + json.dumps(list(phase.command)) + "\n")
        log.flush()
        completed = subprocess.run(
            phase.command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"returncode={completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(
            f"phase failed: {phase.name}; inspect {log_path.relative_to(ROOT)}"
        )
    _atomic_write_json(marker, _phase_marker_payload(phase))
    print(f"phase={phase.name} status=complete", flush=True)


def _require_passed_audit(path: Path, label: str) -> dict[str, Any]:
    payload = load_json(path)
    validate_safety_claims(payload, label)
    passed = payload.get("passed")
    if passed is None:
        passed = payload.get("integrity_gate_passed")
    if passed is not True:
        raise ValueError(f"final audit did not pass: {label}")
    return payload


def build_advanced_state(
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    state_path: Path,
    paths: DailyPaths,
    session: str,
) -> dict[str, Any]:
    report_root = resolve_path(paths.report_root)
    lifecycle_manifest = resolve_path(
        Path(paths.lifecycle_root) / "lifecycle/manifest.json"
    )
    forward_manifest = resolve_path(Path(paths.forward_inputs) / "manifest.json")
    day_manifest = resolve_path(Path(paths.merged_day_release) / "manifest.json")
    stale_manifest = resolve_path(Path(paths.merged_stale_cache) / "manifest.json")
    trajectory_audit = _require_passed_audit(
        report_root / "trajectory_increment_audit.json", "trajectory audit"
    )
    increment_day_audit = _require_passed_audit(
        report_root / "day_increment_audit.json", "increment day audit"
    )
    merged_day_audit = _require_passed_audit(
        report_root / "day_merged_audit.json", "merged day audit"
    )
    increment_stale_audit = _require_passed_audit(
        report_root / "stale_increment_audit.json", "increment stale audit"
    )
    merged_stale_audit = _require_passed_audit(
        report_root / "stale_merged_audit.json", "merged stale audit"
    )
    for payload in (
        trajectory_audit,
        increment_day_audit,
        merged_day_audit,
        increment_stale_audit,
        merged_stale_audit,
    ):
        validate_safety_claims(payload, "final component audit")

    lifecycle = load_json(lifecycle_manifest)
    lifecycle_audit = load_json(
        resolve_path(Path(paths.lifecycle_root) / "lifecycle_audit.json")
    )
    forward = load_json(forward_manifest)
    day = load_json(day_manifest)
    stale = load_json(stale_manifest)
    if lifecycle_audit.get("status") != "pass":
        raise ValueError("new lifecycle audit did not pass")
    if lifecycle.get("end") != session:
        raise ValueError("new lifecycle does not end on the requested session")
    if forward.get("incremental_date") != session:
        raise ValueError("new forward inputs do not end on the requested session")
    if day.get("last_date") != session:
        raise ValueError("new day release does not end on the requested session")
    if stale.get("target_end") != session or stale.get("strict_out_of_sample") is not True:
        raise ValueError("new stale cache is not strict OOS through the session")
    for label, payload in (
        ("new forward inputs", forward),
        ("new day release", day),
        ("new stale cache", stale),
    ):
        validate_safety_claims(payload, label)

    reports: dict[str, dict[str, str]] = {}
    report_hashes: dict[str, str] = {}
    for name in MODEL_NAMES:
        relative = Path(paths.model_report_dir) / f"{name}.json"
        absolute = resolve_path(relative)
        payload = load_json(absolute)
        validate_safety_claims(payload, f"new model report {name}")
        parity = payload.get("reference_inference_parity")
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise ValueError(f"historical inference parity failed: {name}")
        digest = file_sha256(absolute)
        reports[name] = {"path": relative.as_posix(), "sha256": digest}
        report_hashes[name] = digest

    audit_relative = Path(paths.forward_audit_dir) / "summary.json"
    audit_absolute = resolve_path(audit_relative)
    audit = load_json(audit_absolute)
    validate_safety_claims(audit, "new forward audit")
    if (
        audit.get("status") != "complete"
        or audit.get("broker_order_calls_executed") != 0
        or audit.get("decision") not in ALLOWED_FORWARD_DECISIONS
        or audit.get("forward_period", {}).get("end") != session
    ):
        raise ValueError("new forward audit did not complete its frozen contract")
    if audit.get("inputs", {}).get("reports") != report_hashes:
        raise ValueError("new forward audit report hashes differ")
    if audit["inputs"].get("day_release_manifest_sha256") != file_sha256(day_manifest):
        raise ValueError("new forward audit day-release hash differs")
    if audit["inputs"].get("stale_cache_manifest_sha256") != file_sha256(stale_manifest):
        raise ValueError("new forward audit stale-cache hash differs")

    previous_state_sha = file_sha256(state_path)
    advanced = dict(state)
    advanced.update(
        {
            "generation": int(state.get("generation", 0)) + 1,
            "latest_session": session,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_state_sha256": previous_state_sha,
            "artifacts": {
                "lifecycle": {
                    "root": (Path(paths.lifecycle_root) / "lifecycle").as_posix(),
                    "path": (Path(paths.lifecycle_root) / "lifecycle/manifest.json").as_posix(),
                    "sha256": file_sha256(lifecycle_manifest),
                },
                "forward_inputs": {
                    "root": paths.forward_inputs,
                    "path": (Path(paths.forward_inputs) / "manifest.json").as_posix(),
                    "sha256": file_sha256(forward_manifest),
                },
                "day_release": {
                    "root": paths.merged_day_release,
                    "path": (Path(paths.merged_day_release) / "manifest.json").as_posix(),
                    "sha256": file_sha256(day_manifest),
                },
                "stale_cache": {
                    "root": paths.merged_stale_cache,
                    "path": (Path(paths.merged_stale_cache) / "manifest.json").as_posix(),
                    "sha256": file_sha256(stale_manifest),
                },
            },
            "reports": reports,
            "forward_audit": {
                "path": audit_relative.as_posix(),
                "sha256": file_sha256(audit_absolute),
            },
            "orders_sent": 0,
            "broker_order_calls_allowed": False,
            "live_orders_allowed": False,
        }
    )
    history = list(state.get("history") or [])
    history.append(
        {
            "generation": advanced["generation"],
            "session": session,
            "decision": audit["decision"],
            "forward_sessions": audit["forward_period"]["sessions"],
            "forward_audit_sha256": advanced["forward_audit"]["sha256"],
            "orders_sent": 0,
        }
    )
    advanced["history"] = history
    return advanced


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    state_path = resolve_path(args.state)
    config = load_json(config_path)
    state = load_json(state_path)
    validate_config_and_state(config_path, config, state)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "latest_session": state["latest_session"],
                    "orders_sent": 0,
                    "live_orders_allowed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    today = datetime.now(KST).date().isoformat()
    validate_session_contract(state["latest_session"], args.session, today=today)
    paths, phases = build_plan(config, state, args.session)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "session": args.session,
                    "phases": [
                        {
                            "name": phase.name,
                            "command": list(phase.command),
                            "outputs": list(phase.outputs),
                        }
                        for phase in phases
                    ],
                    "broker_order_calls_executed": 0,
                    "live_orders_allowed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    minimum_free = int(config["minimum_free_bytes"])
    if shutil.disk_usage(ROOT).free < minimum_free:
        raise RuntimeError("insufficient free disk for an atomic forward update")
    control_root = resolve_path(paths.control_root)
    control_root.mkdir(parents=True, exist_ok=True)
    resolve_path(paths.data_root).mkdir(parents=True, exist_ok=True)
    resolve_path(paths.report_root).mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(state_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another forward daily run holds the state lock") from exc
        state_before = control_root / "state_before.json"
        if not state_before.exists():
            state_before.write_bytes(state_path.read_bytes())
        elif file_sha256(state_before) != file_sha256(state_path):
            raise ValueError("resume state differs from the run's frozen input state")
        for phase in phases:
            run_phase(phase, control_root)
        advanced = build_advanced_state(
            config, state, state_path, paths, args.session
        )
        _atomic_write_json(control_root / "state_after.json", advanced)
        _atomic_write_json(state_path, advanced)
    print(
        json.dumps(
            {
                "status": "complete",
                "session": args.session,
                "generation": advanced["generation"],
                "forward_audit": advanced["forward_audit"],
                "orders_sent": 0,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
