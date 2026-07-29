from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
KST = ZoneInfo("Asia/Seoul")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an immutable daily Kiwoom causal release and run read-only shadow inference."
    )
    parser.add_argument("--config", default="configs/ops.latent-head-shadow.json")
    parser.add_argument("--universe-manifest", default="data/universes/krx500_pit_20191231.json")
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--release-root", default="data/ops/releases")
    parser.add_argument("--current-link", default="data/ops/current_ohlcv_release")
    parser.add_argument("--current-signals-link", default="ops/state/current_shadow_signals.json")
    parser.add_argument("--reports-root", default="ops/reports/latent_head_shadow/daily")
    parser.add_argument("--lock-file", default="ops/state/daily_causal_shadow.lock")
    parser.add_argument(
        "--lifecycle-reference-manifest",
        default="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/manifest.json",
    )
    parser.add_argument("--expected-proxy-tickers", type=int, default=47)
    parser.add_argument("--sleep-sec", type=float, default=0.22)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str], allowed_returncodes: tuple[int, ...] = (0,)) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(result.returncode, command)


def atomic_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(link.name + ".next")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.resolve(), target_is_directory=True)
    temporary.replace(link)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_reference_path(value: object, manifest_path: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, manifest_path.parent / path):
        if candidate.exists():
            return candidate
    return ROOT / path


def _validate_proxy_cache(
    output_dir: Path,
    *,
    reference_path: Path,
    reference_sha256: str,
    proxy_rows: list[dict],
    start: str,
    end: str,
    expected_proxy_tickers: int,
) -> Path:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("lifecycle proxy cache must be a real directory")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("lifecycle proxy cache exists without a manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("role") != "immutable_daily_lifecycle_proxy_cache"
    ):
        raise ValueError("lifecycle proxy cache schema mismatch")
    if payload.get("reference_manifest_sha256") != reference_sha256:
        raise ValueError("lifecycle proxy cache reference hash mismatch")
    if payload.get("start") != start or payload.get("end") != end:
        raise ValueError("lifecycle proxy cache date contract mismatch")
    outputs = list(payload.get("outputs") or [])
    if len(outputs) != expected_proxy_tickers:
        raise ValueError("lifecycle proxy cache ticker count mismatch")
    output_by_ticker = {
        str(row.get("ticker") or "").zfill(6): row
        for row in outputs
        if isinstance(row, dict)
    }
    expected_tickers = {str(row.get("ticker") or "").zfill(6) for row in proxy_rows}
    if len(output_by_ticker) != expected_proxy_tickers or set(output_by_ticker) != expected_tickers:
        raise ValueError("lifecycle proxy cache ticker identity mismatch")

    suffix = f"{start.replace('-', '')}_{end.replace('-', '')}.csv"
    expected_files = {f"{ticker}_{suffix}" for ticker in expected_tickers}
    actual_files = {path.name for path in output_dir.glob("*.csv")}
    if actual_files != expected_files:
        raise ValueError("lifecycle proxy cache file set mismatch")
    for reference_row in proxy_rows:
        ticker = str(reference_row.get("ticker") or "").zfill(6)
        row = output_by_ticker[ticker]
        expected_file = f"{ticker}_{suffix}"
        expected_sha = str(reference_row.get("source_sha256") or "")
        source = _resolve_reference_path(reference_row.get("source_path"), reference_path)
        path = output_dir / expected_file
        if (
            row.get("file") != expected_file
            or row.get("sha256") != expected_sha
            or row.get("source_sha256") != expected_sha
            or row.get("source") != str(source)
            or not source.is_file()
            or file_sha256(source) != expected_sha
            or not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != expected_sha
        ):
            raise ValueError(f"lifecycle proxy cache file mismatch: {path.name}")
    return manifest_path


def _reference_proxy_rows(reference: dict, expected_proxy_tickers: int) -> list[dict]:
    proxy_rows = [
        row
        for row in reference.get("outputs") or []
        if isinstance(row, dict) and row.get("execution_supported") is False
    ]
    tickers = [str(row.get("ticker") or "").zfill(6) for row in proxy_rows]
    if (
        len(proxy_rows) != expected_proxy_tickers
        or len(set(tickers)) != expected_proxy_tickers
    ):
        raise ValueError(
            f"expected {expected_proxy_tickers} unique frozen proxy tickers, found {len(set(tickers))}"
        )
    if any(len(str(row.get("source_sha256") or "")) != 64 for row in proxy_rows):
        raise ValueError("frozen lifecycle proxy source hash is missing or malformed")
    return sorted(proxy_rows, key=lambda value: str(value.get("ticker") or ""))


def validate_causal_parent_audit(
    payload: dict,
    manifest_path: Path,
    *,
    proxy_tickers: set[str],
    expected_tickers: int = 500,
) -> None:
    """Accept the causal parent only when its sole gap is the frozen proxy set."""

    missing = [str(ticker).zfill(6) for ticker in payload.get("missing_tickers") or []]
    expected_outputs = expected_tickers - len(proxy_tickers)
    valid = (
        payload.get("status") == "blocked"
        and payload.get("blockers") == ["frozen_universe_coverage_incomplete"]
        and payload.get("provider") == "kiwoom_rest_ka10081"
        and payload.get("release_manifest_sha256") == file_sha256(manifest_path)
        and int(payload.get("expected_tickers", 0)) == expected_tickers
        and int(payload.get("output_tickers", 0)) == expected_outputs
        and int(payload.get("verified_files", 0)) == expected_outputs
        and len(missing) == len(set(missing))
        and set(missing) == proxy_tickers
    )
    if not valid:
        raise RuntimeError(
            "daily causal OHLCV parent does not match the frozen lifecycle proxy partition"
        )


def validate_lifecycle_audit(
    payload: dict,
    manifest_path: Path,
    *,
    expected_proxy_tickers: int,
    expected_tickers: int = 500,
) -> None:
    provider_counts = payload.get("provider_counts") or {}
    valid = (
        payload.get("status") == "pass"
        and not (payload.get("blockers") or [])
        and payload.get("live_orders_allowed") is False
        and payload.get("release_manifest_sha256") == file_sha256(manifest_path)
        and int(payload.get("expected_tickers", 0)) == expected_tickers
        and int(payload.get("verified_files", 0)) == expected_tickers
        and int(payload.get("lifecycle_violations", -1)) == 0
        and sum(int(value) for value in provider_counts.values()) == expected_tickers
        and int(provider_counts.get("finance_data_reader_adjusted_return_index_proxy", 0))
        == expected_proxy_tickers
    )
    if not valid:
        raise RuntimeError("daily lifecycle OHLCV release audit did not pass its bound contract")


def validate_readonly_shadow_output(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = (
        payload.get("status") == "complete"
        and payload.get("approval_scope") == "read_only_shadow"
        and payload.get("live_orders_allowed") is False
        and int(payload.get("orders_submitted", -1)) == 0
        and isinstance(payload.get("signals"), list)
    )
    if not valid:
        raise RuntimeError("shadow inference output violates the read-only contract")
    return payload


def materialize_lifecycle_proxy_cache(
    reference_manifest: str | Path,
    output_dir: str | Path,
    *,
    start: str,
    end: str,
    expected_proxy_tickers: int = 47,
) -> Path:
    """Copy frozen delisted sources under filenames covering the daily end date."""

    reference_path = Path(reference_manifest)
    if not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    reference_sha = file_sha256(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    proxy_rows = _reference_proxy_rows(reference, expected_proxy_tickers)
    output_root = Path(output_dir)
    if output_root.exists():
        return _validate_proxy_cache(
            output_root,
            reference_path=reference_path,
            reference_sha256=reference_sha,
            proxy_rows=proxy_rows,
            start=start,
            end=end,
            expected_proxy_tickers=expected_proxy_tickers,
        )

    temporary = output_root.with_name(f".{output_root.name}.tmp-{uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    suffix = f"{start.replace('-', '')}_{end.replace('-', '')}.csv"
    outputs = []
    try:
        for row in proxy_rows:
            ticker = str(row.get("ticker") or "").zfill(6)
            source = _resolve_reference_path(row.get("source_path"), reference_path)
            expected_sha = str(row.get("source_sha256") or "")
            if not source.is_file() or file_sha256(source) != expected_sha:
                raise ValueError(f"frozen lifecycle proxy source mismatch: {ticker}")
            target = temporary / f"{ticker}_{suffix}"
            shutil.copy2(source, target)
            target_sha = file_sha256(target)
            if target_sha != expected_sha:
                raise ValueError(f"frozen lifecycle proxy copy mismatch: {ticker}")
            outputs.append(
                {
                    "ticker": ticker,
                    "file": target.name,
                    "sha256": target_sha,
                    "source": str(source),
                    "source_sha256": expected_sha,
                }
            )
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "role": "immutable_daily_lifecycle_proxy_cache",
                    "reference_manifest": str(reference_path),
                    "reference_manifest_sha256": reference_sha,
                    "start": start,
                    "end": end,
                    "outputs": outputs,
                    "live_orders_allowed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root / "manifest.json"


def main() -> None:
    args = parse_args()
    now = datetime.now(KST)
    if not args.force and (now.weekday() >= 5 or (now.hour, now.minute) < (15, 40)):
        print(json.dumps({"status": "skipped", "reason": "outside_after_close_window"}))
        return

    lock_path = ROOT / args.lock_file
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("daily causal shadow job is already running") from exc

        release_date = now.strftime("%Y-%m-%d")
        release_id = f"ohlcv_causal_{now.strftime('%Y%m%d')}"
        release = ROOT / args.release_root / release_id
        source = release / "source"
        raw_pages = release / "raw_pages"
        causal_ohlcv = release / "causal_ohlcv"
        causal_manifest = release / "causal_manifest.json"
        causal_audit = release / "causal_audit.json"
        proxy_cache = release / "lifecycle_proxy_cache"
        lifecycle_release = release / "lifecycle"
        lifecycle_manifest = lifecycle_release / "manifest.json"
        audit = release / "lifecycle_audit.json"
        output = ROOT / args.reports_root / f"{now.strftime('%Y%m%d')}.json"

        if not causal_manifest.exists():
            run(
                [
                    sys.executable,
                    "scripts/backfill_kiwoom_ohlcv.py",
                    "--universe-manifest",
                    args.universe_manifest,
                    "--start",
                    "2020-01-01",
                    "--end",
                    release_date,
                    "--basis",
                    "both",
                    "--cache-dir",
                    str(source),
                    "--raw-cache-dir",
                    str(raw_pages),
                    "--coverage-output",
                    str(source / "coverage.jsonl"),
                    "--run-id",
                    release_id,
                    "--env-file",
                    args.env_file,
                    "--sleep-sec",
                    str(args.sleep_sec),
                    "--max-pages",
                    str(args.max_pages),
                    "--resume",
                ]
            )
            run(
                [
                    sys.executable,
                    "scripts/build_causal_ohlcv.py",
                    "--universe-manifest",
                    args.universe_manifest,
                    "--raw-dir",
                    str(source / "raw"),
                    "--adjusted-dir",
                    str(source / "adjusted"),
                    "--output-dir",
                    str(causal_ohlcv),
                    "--events-output",
                    str(release / "corporate_actions.jsonl"),
                    "--manifest-output",
                    str(causal_manifest),
                    "--start",
                    "2020-01-01",
                    "--end",
                    release_date,
                ],
                allowed_returncodes=(0, 2),
            )
        causal_audit.unlink(missing_ok=True)
        run(
            [
                sys.executable,
                "scripts/audit_causal_ohlcv_release.py",
                "--manifest",
                str(causal_manifest),
                "--output",
                str(causal_audit),
            ],
            allowed_returncodes=(0, 1),
        )

        causal_audit_payload = json.loads(causal_audit.read_text(encoding="utf-8"))
        reference_path = Path(args.lifecycle_reference_manifest)
        if not reference_path.is_absolute():
            reference_path = ROOT / reference_path
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_proxy_rows = _reference_proxy_rows(
            reference,
            args.expected_proxy_tickers,
        )
        validate_causal_parent_audit(
            causal_audit_payload,
            causal_manifest,
            proxy_tickers={
                str(row.get("ticker") or "").zfill(6)
                for row in reference_proxy_rows
            },
        )

        materialize_lifecycle_proxy_cache(
            args.lifecycle_reference_manifest,
            proxy_cache,
            start="2020-01-01",
            end=release_date,
            expected_proxy_tickers=args.expected_proxy_tickers,
        )
        if not lifecycle_manifest.exists():
            run(
                [
                    sys.executable,
                    "scripts/build_lifecycle_hybrid_ohlcv.py",
                    "--universe-manifest",
                    args.universe_manifest,
                    "--causal-manifest",
                    str(causal_manifest),
                    "--proxy-cache-dir",
                    str(proxy_cache),
                    "--output-dir",
                    str(lifecycle_release),
                    "--start",
                    "2020-01-01",
                    "--end",
                    release_date,
                    "--expected-proxy-tickers",
                    str(args.expected_proxy_tickers),
                    "--skip-provider-overlap",
                ]
            )
        audit.unlink(missing_ok=True)
        run(
            [
                sys.executable,
                "scripts/audit_lifecycle_hybrid_ohlcv.py",
                "--manifest",
                str(lifecycle_manifest),
                "--output",
                str(audit),
                "--expected-proxy-tickers",
                str(args.expected_proxy_tickers),
            ]
        )

        audit_payload = json.loads(audit.read_text(encoding="utf-8"))
        validate_lifecycle_audit(
            audit_payload,
            lifecycle_manifest,
            expected_proxy_tickers=args.expected_proxy_tickers,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
        try:
            run(
                [
                    sys.executable,
                    "scripts/run_readonly_shadow.py",
                    "--config",
                    args.config,
                    "--output",
                    str(temporary_output),
                    "--cache-dir",
                    str(lifecycle_release / "ohlcv"),
                    "--data-end",
                    release_date,
                ]
            )
            if not temporary_output.is_file():
                raise RuntimeError("read-only shadow inference did not create its output")
            validate_readonly_shadow_output(temporary_output)
            temporary_output.replace(output)
        finally:
            temporary_output.unlink(missing_ok=True)
        atomic_symlink(lifecycle_release, ROOT / args.current_link)
        atomic_symlink(output, ROOT / args.current_signals_link)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "approval_scope": "read_only_shadow",
                    "release": str(release),
                    "lifecycle_release": str(lifecycle_release),
                    "audit": str(audit),
                    "signals": str(output),
                    "live_orders_allowed": False,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
