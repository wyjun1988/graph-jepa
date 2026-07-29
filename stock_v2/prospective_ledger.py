from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence, Union
import zipfile

import numpy as np


LEDGER_ROLE = "post_impact_prospective_prediction_commit"
LEDGER_SCHEMA_VERSION = 1
GENESIS_SHA256 = "0" * 64
_CHAIN_FIELDS = {"sequence", "previous_record_sha256", "record_sha256"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARRAY_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class ProspectiveLedgerError(ValueError):
    pass


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProspectiveLedgerError(
            "prospective ledger payload is not canonical JSON"
        ) from exc
    return encoded.encode("utf-8")


def record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _safe_artifact_path(root: Path, relative: object) -> Path:
    value = str(relative or "")
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ProspectiveLedgerError(
            "prediction artifact path must be a safe relative path"
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ProspectiveLedgerError(
            "prediction artifact escapes its immutable root"
        ) from exc
    return resolved


def _validate_sha256(value: object, label: str) -> str:
    normalized = str(value or "")
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ProspectiveLedgerError(f"{label} must be a lowercase SHA-256")
    return normalized


def _validate_commit_payload(payload: Mapping[str, Any]) -> None:
    if any(field in payload for field in _CHAIN_FIELDS):
        raise ProspectiveLedgerError("caller must not supply ledger chain fields")
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ProspectiveLedgerError("prospective ledger schema mismatch")
    if payload.get("role") != LEDGER_ROLE:
        raise ProspectiveLedgerError("prospective ledger role mismatch")
    commit_id = str(payload.get("commit_id") or "")
    if not commit_id or len(commit_id) > 256:
        raise ProspectiveLedgerError("prospective commit_id is missing or too long")
    try:
        date.fromisoformat(str(payload.get("session")))
    except (TypeError, ValueError) as exc:
        raise ProspectiveLedgerError(
            "prospective session must be an ISO calendar date"
        ) from exc
    timestamp = payload.get("decision_timestamp_utc_ns")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise ProspectiveLedgerError(
            "decision timestamp must be a positive integer UTC nanosecond value"
        )
    if payload.get("source_mode") not in {
        "historical_causal_replay",
        "live_read_only",
    }:
        raise ProspectiveLedgerError("unsupported prospective source mode")
    if payload.get("live_orders_allowed") is not False:
        raise ProspectiveLedgerError("prospective commits must prohibit live orders")
    if payload.get("promotion_eligible") is not False:
        raise ProspectiveLedgerError(
            "prospective commits cannot be promotion eligible"
        )
    if payload.get("broker_order_calls_executed") != 0:
        raise ProspectiveLedgerError(
            "prospective commits require zero broker order calls"
        )
    causality = payload.get("causality")
    required_causality = {
        "completed_bars_only",
        "future_intraday_rows_absent_from_model_input",
        "labels_absent_from_model_input",
        "model_eval_mode",
    }
    if not isinstance(causality, Mapping) or any(
        causality.get(claim) is not True for claim in required_causality
    ):
        raise ProspectiveLedgerError(
            "prospective commit is missing causal inference claims"
        )
    input_pins = payload.get("input_pins")
    if not isinstance(input_pins, Mapping) or not input_pins:
        raise ProspectiveLedgerError("prospective commit has no input pins")
    for name, value in input_pins.items():
        _validate_sha256(value, f"input pin {name}")
    models = payload.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ProspectiveLedgerError("prospective commit has no model pins")
    for name, model in models.items():
        if not isinstance(model, Mapping):
            raise ProspectiveLedgerError(f"model pin {name} must be an object")
        _validate_sha256(model.get("checkpoint_sha256"), f"model pin {name}")
    artifact = payload.get("prediction_artifact")
    if not isinstance(artifact, Mapping):
        raise ProspectiveLedgerError("prospective commit has no prediction artifact")
    _validate_sha256(artifact.get("sha256"), "prediction artifact")
    size = artifact.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ProspectiveLedgerError("prediction artifact size must be positive")


def _validate_artifact(record: Mapping[str, Any], artifact_root: Path) -> None:
    artifact = record["prediction_artifact"]
    path = _safe_artifact_path(artifact_root, artifact.get("path"))
    if not path.is_file() or path.is_symlink():
        raise ProspectiveLedgerError(f"prediction artifact is missing: {path}")
    if path.stat().st_size != int(artifact["bytes"]):
        raise ProspectiveLedgerError(f"prediction artifact size changed: {path}")
    if file_sha256(path) != artifact["sha256"]:
        raise ProspectiveLedgerError(f"prediction artifact hash changed: {path}")


def read_prediction_ledger(
    path: Union[str, Path],
    *,
    artifact_root: Union[str, Path, None] = None,
) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    root = Path(artifact_root) if artifact_root is not None else ledger_path.parent
    records: list[dict[str, Any]] = []
    expected_previous = GENESIS_SHA256
    commit_ids: set[str] = set()
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ProspectiveLedgerError(
                    f"prospective ledger has a torn line: {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProspectiveLedgerError(
                    f"prospective ledger JSON is invalid: {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ProspectiveLedgerError(
                    f"prospective ledger line is not an object: {line_number}"
                )
            payload = {key: value for key, value in record.items() if key not in _CHAIN_FIELDS}
            _validate_commit_payload(payload)
            expected_sequence = len(records) + 1
            if record.get("sequence") != expected_sequence:
                raise ProspectiveLedgerError(
                    f"prospective ledger sequence mismatch: {line_number}"
                )
            if record.get("previous_record_sha256") != expected_previous:
                raise ProspectiveLedgerError(
                    f"prospective ledger chain mismatch: {line_number}"
                )
            actual_hash = record_sha256(record)
            if record.get("record_sha256") != actual_hash:
                raise ProspectiveLedgerError(
                    f"prospective ledger record hash mismatch: {line_number}"
                )
            commit_id = str(record["commit_id"])
            if commit_id in commit_ids:
                raise ProspectiveLedgerError(
                    f"prospective ledger commit_id is duplicated: {commit_id}"
                )
            _validate_artifact(record, root)
            commit_ids.add(commit_id)
            records.append(record)
            expected_previous = actual_hash
    return records


@contextmanager
def _locked_ledger(path: Path) -> Iterator[None]:
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_prediction_commit(
    path: Union[str, Path],
    payload: Mapping[str, Any],
    *,
    artifact_root: Union[str, Path, None] = None,
) -> tuple[dict[str, Any], bool]:
    ledger_path = Path(path)
    root = Path(artifact_root) if artifact_root is not None else ledger_path.parent
    normalized = dict(payload)
    _validate_commit_payload(normalized)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _locked_ledger(ledger_path):
        records = read_prediction_ledger(ledger_path, artifact_root=root)
        for existing in records:
            if existing["commit_id"] != normalized["commit_id"]:
                continue
            existing_payload = {
                key: value
                for key, value in existing.items()
                if key not in _CHAIN_FIELDS
            }
            if canonical_json_bytes(existing_payload) == canonical_json_bytes(normalized):
                return existing, False
            raise ProspectiveLedgerError(
                f"commit_id already exists with different content: {normalized['commit_id']}"
            )
        _validate_artifact(normalized, root)
        record = {
            **normalized,
            "sequence": len(records) + 1,
            "previous_record_sha256": (
                records[-1]["record_sha256"] if records else GENESIS_SHA256
            ),
        }
        record["record_sha256"] = record_sha256(record)
        encoded = canonical_json_bytes(record) + b"\n"
        descriptor = os.open(
            ledger_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            view = memoryview(encoded)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise OSError("prospective ledger append made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        verified = read_prediction_ledger(ledger_path, artifact_root=root)
        if verified[-1]["record_sha256"] != record["record_sha256"]:
            raise ProspectiveLedgerError("prospective ledger append did not verify")
        return record, True


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if not arrays:
        raise ProspectiveLedgerError("prediction artifact must contain arrays")
    names = sorted(str(name) for name in arrays)
    if len(names) != len(set(names)) or any(
        not _ARRAY_NAME_PATTERN.fullmatch(name) for name in names
    ):
        raise ProspectiveLedgerError("prediction artifact array names are invalid")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in names:
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ProspectiveLedgerError(
                    "prediction artifacts cannot contain object arrays"
                )
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def write_immutable_prediction_artifact(
    path: Union[str, Path],
    arrays: Mapping[str, np.ndarray],
    *,
    artifact_root: Union[str, Path],
) -> dict[str, Any]:
    root = Path(artifact_root)
    output = Path(path)
    if not output.is_absolute():
        output = _safe_artifact_path(root, output)
    else:
        try:
            output.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ProspectiveLedgerError(
                "prediction artifact must be inside its immutable root"
            ) from exc
    encoded = deterministic_npz_bytes(arrays)
    digest = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise ProspectiveLedgerError(
                "immutable prediction artifact path is not a regular file"
            )
        if output.read_bytes() != encoded:
            raise ProspectiveLedgerError(
                f"immutable prediction artifact changed: {output}"
            )
    else:
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    relative = output.resolve().relative_to(root.resolve()).as_posix()
    return {
        "path": relative,
        "sha256": digest,
        "bytes": len(encoded),
        "format": "deterministic_npz_v1",
    }


def prediction_array_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        if array.dtype.hasobject:
            raise ProspectiveLedgerError(
                "prediction fingerprints cannot contain object arrays"
            )
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def ledger_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "first_commit_id": records[0]["commit_id"] if records else None,
        "last_commit_id": records[-1]["commit_id"] if records else None,
        "head_sha256": records[-1]["record_sha256"] if records else GENESIS_SHA256,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
