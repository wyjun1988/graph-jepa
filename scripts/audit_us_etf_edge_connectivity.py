from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_real_backtest


NODE_CLASSES = ("stock", "external", "etf")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_class_codes(
    node_tickers: Sequence[str], tradable_count: int
) -> np.ndarray:
    if tradable_count <= 0 or tradable_count > len(node_tickers):
        raise ValueError("invalid tradable node count")
    codes = np.full(len(node_tickers), 1, dtype=np.int8)
    codes[:tradable_count] = 0
    for index, node_id in enumerate(node_tickers[tradable_count:], tradable_count):
        if str(node_id).startswith("EXTETF:"):
            codes[index] = 2
    return codes


def summarize_edge_cache(
    *,
    node_tickers: Sequence[str],
    tradable_count: int,
    dates: Sequence[object],
    edge_cache: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    node_ids = tuple(str(value) for value in node_tickers)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("edge audit requires unique node IDs")
    class_codes = _node_class_codes(node_ids, int(tradable_count))
    class_counts = {
        name: int(np.count_nonzero(class_codes == index))
        for index, name in enumerate(NODE_CLASSES)
    }
    pair_labels = tuple(
        f"{source}_to_{destination}"
        for source in NODE_CLASSES
        for destination in NODE_CLASSES
    )
    aggregate_counts = np.zeros(len(pair_labels), dtype=np.int64)
    aggregate_abs_weight = np.zeros(len(pair_labels), dtype=np.float64)
    etf_to_stock_sources = np.zeros(len(node_ids), dtype=bool)
    etf_to_stock_destinations = np.zeros(len(node_ids), dtype=bool)
    stock_to_etf_sources = np.zeros(len(node_ids), dtype=bool)
    stock_to_etf_destinations = np.zeros(len(node_ids), dtype=bool)
    step_rows: list[dict[str, Any]] = []

    for step in sorted(int(value) for value in edge_cache):
        edge_index, edge_weight = edge_cache[step]
        index = np.asarray(edge_index.detach().cpu(), dtype=np.int64)
        weight = np.asarray(edge_weight.detach().cpu(), dtype=np.float64)
        if index.ndim != 2 or index.shape[0] != 2 or index.shape[1] != weight.size:
            raise ValueError(f"invalid edge geometry at step {step}")
        if index.size and (index.min() < 0 or index.max() >= len(node_ids)):
            raise ValueError(f"edge node index is out of range at step {step}")
        source = index[0]
        destination = index[1]
        pair_code = class_codes[source] * len(NODE_CLASSES) + class_codes[destination]
        counts = np.bincount(pair_code, minlength=len(pair_labels)).astype(np.int64)
        abs_weight = np.bincount(
            pair_code,
            weights=np.abs(weight),
            minlength=len(pair_labels),
        ).astype(np.float64)
        aggregate_counts += counts
        aggregate_abs_weight += abs_weight

        etf_to_stock = (class_codes[source] == 2) & (class_codes[destination] == 0)
        stock_to_etf = (class_codes[source] == 0) & (class_codes[destination] == 2)
        etf_to_stock_sources[source[etf_to_stock]] = True
        etf_to_stock_destinations[destination[etf_to_stock]] = True
        stock_to_etf_sources[source[stock_to_etf]] = True
        stock_to_etf_destinations[destination[stock_to_etf]] = True
        step_rows.append(
            {
                "step": step,
                "date": str(dates[step]),
                "edges": int(weight.size),
                "pair_counts": {
                    label: int(counts[index])
                    for index, label in enumerate(pair_labels)
                },
            }
        )

    etf_to_stock_code = pair_labels.index("etf_to_stock")
    stock_to_etf_code = pair_labels.index("stock_to_etf")
    return {
        "schema_version": 1,
        "role": "us_etf_training_edge_connectivity_audit",
        "node_counts": class_counts,
        "steps": len(step_rows),
        "aggregate_pair_counts": {
            label: int(aggregate_counts[index])
            for index, label in enumerate(pair_labels)
        },
        "aggregate_pair_abs_weight": {
            label: float(aggregate_abs_weight[index])
            for index, label in enumerate(pair_labels)
        },
        "etf_to_stock": {
            "edges": int(aggregate_counts[etf_to_stock_code]),
            "unique_etf_sources": int(np.count_nonzero(etf_to_stock_sources)),
            "unique_stock_destinations": int(
                np.count_nonzero(etf_to_stock_destinations)
            ),
            "steps_with_edges": int(
                sum(row["pair_counts"]["etf_to_stock"] > 0 for row in step_rows)
            ),
        },
        "stock_to_etf": {
            "edges": int(aggregate_counts[stock_to_etf_code]),
            "unique_stock_sources": int(np.count_nonzero(stock_to_etf_sources)),
            "unique_etf_destinations": int(
                np.count_nonzero(stock_to_etf_destinations)
            ),
            "steps_with_edges": int(
                sum(row["pair_counts"]["stock_to_etf"] > 0 for row in step_rows)
            ),
        },
        "step_rows": step_rows,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }


def _replace_option(arguments: list[str], option: str, value: str) -> None:
    positions = [index for index, token in enumerate(arguments) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"expected one {option} option in captured command")
    arguments[positions[0] + 1] = value


def _captured_arguments(log_path: Path, train_end: str) -> list[str]:
    matches: list[list[str]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("RUN "):
            continue
        tokens = shlex.split(line[4:])
        script_positions = [
            index
            for index, token in enumerate(tokens)
            if token.endswith("scripts/run_real_backtest.py")
        ]
        if len(script_positions) != 1:
            continue
        arguments = tokens[script_positions[0] + 1 :]
        try:
            position = arguments.index("--train-end")
        except ValueError:
            continue
        if position + 1 < len(arguments) and arguments[position + 1] == train_end:
            matches.append(arguments)
    if len(matches) != 1:
        raise ValueError(
            f"expected one captured run command for train_end={train_end}; "
            f"found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a frozen preflight and audit ETF/stock edge connectivity."
    )
    parser.add_argument("--source-log", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--require-cross-edges", action="store_true")
    args = parser.parse_args()

    source_log = Path(args.source_log).resolve()
    output = Path(args.output).resolve()
    scratch = Path(args.scratch_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace immutable audit: {output}")
    arguments = _captured_arguments(source_log, str(args.train_end))
    _replace_option(arguments, "--reports-dir", str(scratch / "reports"))
    _replace_option(arguments, "--models-dir", str(scratch / "models"))
    if "--edge-manifest-only" not in arguments:
        arguments.append("--edge-manifest-only")

    captured: dict[str, Any] = {}
    original = run_real_backtest.build_training_edge_manifest

    def audited_manifest(features: Any, edge_cache: Any) -> dict[str, object]:
        captured["audit"] = summarize_edge_cache(
            node_tickers=features.node_tickers,
            tradable_count=features.tradable_count,
            dates=features.dates,
            edge_cache=edge_cache,
        )
        return original(features, edge_cache)

    run_real_backtest.build_training_edge_manifest = audited_manifest
    previous_argv = sys.argv
    try:
        sys.argv = [str(ROOT / "scripts/run_real_backtest.py"), *arguments]
        run_real_backtest.main()
    finally:
        sys.argv = previous_argv
        run_real_backtest.build_training_edge_manifest = original
    audit = captured.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError("training edge audit callback was not invoked")
    if args.require_cross_edges and (
        int(audit["etf_to_stock"]["edges"]) <= 0
        or int(audit["stock_to_etf"]["edges"]) <= 0
    ):
        raise RuntimeError("ETF and stock nodes lack bidirectional message paths")
    audit["train_end"] = str(args.train_end)
    audit["captured_command"] = arguments
    audit["source_provenance"] = {
        "source_log": str(source_log),
        "source_log_sha256": file_sha256(source_log),
        "run_real_backtest_sha256": file_sha256(
            ROOT / "scripts/run_real_backtest.py"
        ),
        "real_features_sha256": file_sha256(ROOT / "stock_v2/real_features.py"),
        "audit_script_sha256": file_sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": "pass",
                "train_end": str(args.train_end),
                "etf_to_stock_edges": audit["etf_to_stock"]["edges"],
                "stock_to_etf_edges": audit["stock_to_etf"]["edges"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
