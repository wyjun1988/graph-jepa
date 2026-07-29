from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_minute import normalize_kiwoom_ticker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze checksum-verified minute outputs into a run-specific release."
    )
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval-minutes", type=int, required=True)
    parser.add_argument("--basis", choices=["raw", "adjusted"], required=True)
    parser.add_argument("--output-dir", required=True)
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


def _load_universe(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("universe")
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty universe list")
    tickers = [normalize_kiwoom_ticker(row.get("ticker", "")) for row in rows]
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe manifest contains duplicate tickers")
    return tickers


def _select_records(
    path: Path,
    *,
    run_id: str,
    start: str,
    end: str,
    interval_minutes: int,
    basis: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not (
            str(record.get("run_id")) == run_id
            and str(record.get("requested_start")) == start
            and str(record.get("requested_end")) == end
            and int(record.get("interval_minutes", -1)) == int(interval_minutes)
            and str(record.get("basis")) == basis
        ):
            continue
        ticker = normalize_kiwoom_ticker(record.get("ticker", ""))
        if ticker in selected:
            raise ValueError(f"duplicate matching coverage record: {ticker}")
        selected[ticker] = record
    return selected


def main() -> int:
    args = parse_args()
    repository_root = Path(args.repository_root).resolve()
    coverage_path = Path(args.coverage).resolve()
    universe_path = Path(args.universe_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    try:
        output_dir.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("output-dir must be inside repository-root") from exc
    if output_dir.exists():
        raise FileExistsError(f"immutable collection already exists: {output_dir}")
    temporary_dir = Path(str(output_dir) + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    tickers = _load_universe(universe_path)
    records = _select_records(
        coverage_path,
        run_id=args.run_id,
        start=args.start,
        end=args.end,
        interval_minutes=args.interval_minutes,
        basis=args.basis,
    )
    if set(records) != set(tickers):
        missing = sorted(set(tickers) - set(records))
        extra = sorted(set(records) - set(tickers))
        raise ValueError(f"coverage/universe mismatch missing={missing} extra={extra}")

    temporary_dir.mkdir(parents=True)
    frozen_at = datetime.now(tz=timezone.utc).isoformat()
    frozen_records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    try:
        for ticker in tickers:
            record = dict(records[ticker])
            status = str(record.get("status"))
            status_counts[status] += 1
            if status == "error":
                raise ValueError(f"cannot freeze failed collection record: {ticker}")
            if status in {"ok", "partial"}:
                source = Path(str(record.get("output") or ""))
                if not source.is_absolute():
                    source = repository_root / source
                if not source.is_file():
                    raise FileNotFoundError(f"missing collection output: {source}")
                source_sha = file_sha256(source)
                if source_sha != record.get("output_sha256"):
                    raise ValueError(f"collection output checksum mismatch: {ticker}")
                destination_relative = Path("outputs") / source.name
                temporary_destination = temporary_dir / destination_relative
                temporary_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, temporary_destination)
                if file_sha256(temporary_destination) != source_sha:
                    raise ValueError(f"frozen copy checksum mismatch: {ticker}")
                final_destination = output_dir / destination_relative
                record["frozen_from_output"] = str(record["output"])
                record["output"] = final_destination.relative_to(
                    repository_root
                ).as_posix()
                record["frozen_at_utc"] = frozen_at
                output_records.append(
                    {
                        "ticker": ticker,
                        "path": destination_relative.as_posix(),
                        "sha256": source_sha,
                        "bytes": temporary_destination.stat().st_size,
                    }
                )
            elif status not in {"empty", "outside_lifecycle"}:
                raise ValueError(f"unsupported collection status {status}: {ticker}")
            frozen_records.append(record)

        frozen_coverage = temporary_dir / "coverage.jsonl"
        frozen_coverage.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in frozen_records
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "collection_contract": "immutable_kiwoom_minute_collection_v1",
            "run_id": args.run_id,
            "requested_start": args.start,
            "requested_end": args.end,
            "interval_minutes": int(args.interval_minutes),
            "basis": args.basis,
            "universe_tickers": len(tickers),
            "status_counts": dict(sorted(status_counts.items())),
            "frozen_at_utc": frozen_at,
            "inputs": {
                "coverage": str(coverage_path),
                "coverage_sha256": file_sha256(coverage_path),
                "universe_manifest": str(universe_path),
                "universe_manifest_sha256": file_sha256(universe_path),
            },
            "outputs": {
                "coverage": "coverage.jsonl",
                "coverage_sha256": file_sha256(frozen_coverage),
                "files": output_records,
                "files_sha256": canonical_sha256(output_records),
            },
            "portable_output_paths": True,
            "transactional_publish": True,
            "promotion_eligible": False,
            "live_orders_allowed": False,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir.replace(output_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    print(
        json.dumps(
            {
                "status": "complete",
                "files": len(output_records),
                "coverage": str(output_dir / "coverage.jsonl"),
                "manifest": str(output_dir / "manifest.json"),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
