from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_daily_causal_shadow import materialize_lifecycle_proxy_cache
from stock_v2.incremental_ohlcv import extend_causal_history, read_ohlcv_csv
from stock_v2.lifecycle_ohlcv import (
    CAUSAL_PROVIDER,
    PROXY_PROVIDER,
    audit_lifecycle_hybrid_release,
    build_lifecycle_hybrid_release,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_declared_path(value: object, manifest_path: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, manifest_path.parent / path):
        if candidate.exists():
            return candidate
    return ROOT / path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend an audited lifecycle release from bounded Kiwoom raw/adjusted bars."
    )
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument(
        "--incremental-dir",
        help="Root containing raw/ and adjusted/ bridge files.",
    )
    parser.add_argument(
        "--incremental-raw-dir",
        help="Immutable raw bridge directory; requires --incremental-adjusted-dir.",
    )
    parser.add_argument(
        "--incremental-adjusted-dir",
        help="Immutable adjusted bridge directory; requires --incremental-raw-dir.",
    )
    parser.add_argument("--incremental-start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-tickers", type=int, default=500)
    parser.add_argument("--expected-proxy-tickers", type=int, default=47)
    parser.add_argument("--current-link")
    return parser.parse_args()


def incremental_basis_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    root = getattr(args, "incremental_dir", None)
    raw = getattr(args, "incremental_raw_dir", None)
    adjusted = getattr(args, "incremental_adjusted_dir", None)
    if root:
        if raw or adjusted:
            raise ValueError(
                "use either --incremental-dir or explicit raw/adjusted directories"
            )
        root_path = Path(root)
        return root_path / "raw", root_path / "adjusted"
    if not raw or not adjusted:
        raise ValueError(
            "both --incremental-raw-dir and --incremental-adjusted-dir are required"
        )
    return Path(raw), Path(adjusted)


def main() -> None:
    args = parse_args()
    base_manifest_path = Path(args.base_manifest)
    universe_path = Path(args.universe_manifest)
    incremental_raw_dir, incremental_adjusted_dir = incremental_basis_dirs(args)
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"immutable output already exists: {output_root}")
    base_audit = audit_lifecycle_hybrid_release(
        base_manifest_path,
        expected_tickers=args.expected_tickers,
        expected_proxy_tickers=args.expected_proxy_tickers,
    )
    if base_audit["status"] != "pass":
        raise RuntimeError(f"base lifecycle audit failed: {base_audit['blockers']}")

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_outputs = list(base_manifest.get("outputs") or [])
    causal_rows = [row for row in base_outputs if row.get("provider") == CAUSAL_PROVIDER]
    proxy_rows = [row for row in base_outputs if row.get("provider") == PROXY_PROVIDER]
    if len(causal_rows) != args.expected_tickers - args.expected_proxy_tickers:
        raise ValueError("base causal ticker count does not match the release contract")
    if len(proxy_rows) != args.expected_proxy_tickers:
        raise ValueError("base proxy ticker count does not match the release contract")

    causal_dir = output_root / "causal_ohlcv"
    causal_dir.mkdir(parents=True, exist_ok=False)
    suffix = (
        f"{str(base_manifest['start']).replace('-', '')}_"
        f"{str(args.end).replace('-', '')}.csv"
    )
    incremental_suffix = (
        f"{str(args.incremental_start).replace('-', '')}_"
        f"{str(args.end).replace('-', '')}.csv"
    )
    outputs: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for index, row in enumerate(causal_rows, start=1):
        ticker = str(row.get("ticker") or "").zfill(6)
        base_path = resolve_declared_path(row.get("path"), base_manifest_path)
        if file_sha256(base_path) != str(row.get("sha256") or ""):
            raise ValueError(f"base source hash mismatch: {ticker}")
        raw_path = incremental_raw_dir / f"{ticker}_{incremental_suffix}"
        adjusted_path = incremental_adjusted_dir / f"{ticker}_{incremental_suffix}"
        if not raw_path.exists() or not adjusted_path.exists():
            raise FileNotFoundError(f"incremental raw/adjusted pair missing: {ticker}")
        base = read_ohlcv_csv(base_path)
        raw = read_ohlcv_csv(raw_path)
        adjusted = read_ohlcv_csv(adjusted_path)
        merged, ticker_events = extend_causal_history(
            base,
            raw,
            adjusted,
            ticker=ticker,
        )
        output = causal_dir / f"{ticker}_{suffix}"
        temporary = output.with_suffix(output.suffix + ".tmp")
        merged.to_csv(temporary)
        temporary.replace(output)
        events.extend(ticker_events)
        outputs.append(
            {
                "ticker": ticker,
                "rows": int(len(merged)),
                "first_date": str(merged.index.min().date()),
                "last_date": str(merged.index.max().date()),
                "corporate_actions": int(
                    pd.to_numeric(
                        merged.get("CorporateActionFlag", pd.Series(False, index=merged.index)),
                        errors="coerce",
                    ).fillna(0).astype(bool).sum()
                ),
                "path": str(output),
                "sha256": file_sha256(output),
            }
        )
        if index % 50 == 0:
            print(f"extended={index}/{len(causal_rows)}", flush=True)

    events.sort(key=lambda row: (str(row.get("effective_date")), str(row.get("ticker"))))
    events_path = output_root / "corporate_actions_incremental.jsonl"
    write_jsonl(events_path, events)
    missing = sorted(str(row.get("ticker") or "").zfill(6) for row in proxy_rows)
    causal_manifest = {
        "schema_version": 3,
        "source": {
            "provider": CAUSAL_PROVIDER,
            "extension": "bounded_adjacent_return_bridge",
            "base_manifest": str(base_manifest_path),
            "base_manifest_sha256": file_sha256(base_manifest_path),
        },
        "universe_manifest": str(universe_path),
        "universe_sha256": file_sha256(universe_path),
        "start": str(base_manifest["start"]),
        "end": str(args.end),
        "expected_tickers": int(args.expected_tickers),
        "output_tickers": len(outputs),
        "missing_tickers": missing,
        "events_path": str(events_path),
        "events_sha256": file_sha256(events_path),
        "outputs": outputs,
    }
    causal_manifest_path = output_root / "causal_manifest.json"
    causal_manifest_path.write_text(
        json.dumps(causal_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    proxy_cache = output_root / "lifecycle_proxy_cache"
    materialize_lifecycle_proxy_cache(
        base_manifest_path,
        proxy_cache,
        start=str(base_manifest["start"]),
        end=str(args.end),
        expected_proxy_tickers=args.expected_proxy_tickers,
    )
    lifecycle = output_root / "lifecycle"
    lifecycle_manifest = build_lifecycle_hybrid_release(
        universe_manifest=universe_path,
        causal_manifest=causal_manifest_path,
        proxy_cache_dir=proxy_cache,
        output_dir=lifecycle,
        start=str(base_manifest["start"]),
        end=str(args.end),
        expected_tickers=args.expected_tickers,
        expected_proxy_tickers=args.expected_proxy_tickers,
        validate_provider_overlap=False,
    )
    audit = audit_lifecycle_hybrid_release(
        lifecycle_manifest,
        expected_tickers=args.expected_tickers,
        expected_proxy_tickers=args.expected_proxy_tickers,
    )
    audit_path = output_root / "lifecycle_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if audit["status"] != "pass":
        raise RuntimeError(f"extended lifecycle audit failed: {audit['blockers']}")

    if args.current_link:
        link = Path(args.current_link)
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary_link = link.with_name(link.name + ".next")
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(lifecycle.resolve(), target_is_directory=True)
        temporary_link.replace(link)
    print(
        json.dumps(
            {
                "status": "complete",
                "lifecycle_manifest": str(lifecycle_manifest),
                "lifecycle_manifest_sha256": file_sha256(lifecycle_manifest),
                "audit": str(audit_path),
                "last_observation_date": audit["last_observation_date"],
                "last_observation_nodes": audit["last_observation_nodes"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
