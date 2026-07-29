from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_FILES = (
    "context_latent_f16.npy",
    "predicted_delta_f16.npy",
    "predicted_state_f32.npy",
)
SHARED_MANIFEST_FIELDS = (
    "cache_contract",
    "checkpoint_sha256",
    "checkpoint_panel_end",
    "checkpoint_train_end",
    "latent_dim",
    "state_features",
    "stocks",
    "strict_out_of_sample",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cache_contract") != "strict_oos_stale_daily_jepa_h1_v2":
        raise ValueError(f"unsupported stale-cache contract: {root}")
    if payload.get("strict_out_of_sample") is not True:
        raise ValueError(f"stale cache is not strict out-of-sample: {root}")
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe stale cache: {root}")
    for record in payload["files"].values():
        file_path = root / record["path"]
        if file_sha256(file_path) != record["sha256"]:
            raise ValueError(f"stale-cache checksum mismatch: {file_path}")
    return payload


def _load_metadata(root: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    path = root / manifest["files"]["dates_and_tickers"]["path"]
    with np.load(path) as bundle:
        metadata = {name: np.asarray(bundle[name]) for name in bundle.files}
    required = {
        "target_dates",
        "context_dates",
        "context_steps",
        "tickers",
        "state_feature_names",
    }
    if set(metadata) != required:
        raise ValueError("stale-cache metadata fields do not match the v2 contract")
    target_dates = metadata["target_dates"].astype(str)
    context_dates = metadata["context_dates"].astype(str)
    context_steps = metadata["context_steps"].astype(np.int64)
    if not len(target_dates) or target_dates.shape != context_dates.shape:
        raise ValueError("stale-cache date axes are empty or misaligned")
    if context_steps.shape != target_dates.shape:
        raise ValueError("stale-cache context-step axis is misaligned")
    if len(set(target_dates.tolist())) != len(target_dates):
        raise ValueError("stale-cache target dates are duplicated")
    if not np.all(target_dates[:-1] < target_dates[1:]):
        raise ValueError("stale-cache target dates are not sorted")
    if not np.all(context_dates < target_dates):
        raise ValueError("stale-cache context dates are not strictly causal")
    if len(target_dates) > 1 and not np.array_equal(
        context_dates[1:], target_dates[:-1]
    ):
        raise ValueError("stale-cache sessions are not causally adjacent")
    if len(context_steps) > 1 and not np.all(np.diff(context_steps) == 1):
        raise ValueError("stale-cache context steps are not contiguous")
    return metadata


def _load_graph(
    root: Path,
    manifest: dict[str, Any],
    metadata: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    path = root / manifest["files"]["causal_stock_graph"]["path"]
    with np.load(path) as bundle:
        graph = {name: np.asarray(bundle[name]) for name in bundle.files}
    required = {
        "target_dates",
        "context_dates",
        "edge_offsets",
        "edge_index",
        "edge_weight",
    }
    if set(graph) != required:
        raise ValueError("stale-cache graph fields do not match the v2 contract")
    if not np.array_equal(graph["target_dates"], metadata["target_dates"]):
        raise ValueError("stale-cache graph target dates are misaligned")
    if not np.array_equal(graph["context_dates"], metadata["context_dates"]):
        raise ValueError("stale-cache graph context dates are misaligned")
    offsets = graph["edge_offsets"].astype(np.int64, copy=False)
    edge_index = graph["edge_index"].astype(np.int32, copy=False)
    edge_weight = graph["edge_weight"].astype(np.float32, copy=False)
    if offsets.shape != (len(metadata["target_dates"]) + 1,):
        raise ValueError("stale-cache graph offsets are misaligned")
    if offsets[0] != 0 or (np.diff(offsets) <= 0).any():
        raise ValueError("stale-cache graph contains an empty or invalid date")
    if edge_index.shape != (2, int(offsets[-1])):
        raise ValueError("stale-cache graph index is misaligned")
    if edge_weight.shape != (int(offsets[-1]),):
        raise ValueError("stale-cache graph weights are misaligned")
    return {
        **graph,
        "edge_offsets": offsets,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    }


def _copy_concatenated_array(
    base_path: Path,
    incremental_path: Path,
    output_path: Path,
    base_dates: int,
    incremental_dates: int,
) -> None:
    base = np.load(base_path, mmap_mode="r")
    incremental = np.load(incremental_path, mmap_mode="r")
    if base.shape[0] != base_dates or incremental.shape[0] != incremental_dates:
        raise ValueError(f"stale-cache array date axis mismatch: {base_path.name}")
    if base.shape[1:] != incremental.shape[1:] or base.dtype != incremental.dtype:
        raise ValueError(f"stale-cache array contract changed: {base_path.name}")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=base.dtype,
        shape=(base_dates + incremental_dates, *base.shape[1:]),
    )
    output[:base_dates] = base
    output[base_dates:] = incremental
    output.flush()


def merge_caches(
    base_root: Path,
    incremental_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"immutable output already exists: {output_root}")
    temporary = Path(str(output_root) + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output already exists: {temporary}")
    base_manifest = _load_manifest(base_root)
    incremental_manifest = _load_manifest(incremental_root)
    for field in SHARED_MANIFEST_FIELDS:
        if base_manifest.get(field) != incremental_manifest.get(field):
            raise ValueError(f"stale-cache manifest field changed: {field}")
    for claim, value in base_manifest["causality"].items():
        if value is True and incremental_manifest["causality"].get(claim) is not True:
            raise ValueError(f"incremental causality claim changed: {claim}")

    base_metadata = _load_metadata(base_root, base_manifest)
    incremental_metadata = _load_metadata(incremental_root, incremental_manifest)
    for field in ("tickers", "state_feature_names"):
        if not np.array_equal(base_metadata[field], incremental_metadata[field]):
            raise ValueError(f"stale-cache metadata field changed: {field}")
    if incremental_metadata["target_dates"][0] <= base_metadata["target_dates"][-1]:
        raise ValueError("incremental stale-cache dates overlap the immutable base")
    if incremental_metadata["context_dates"][0] != base_metadata["target_dates"][-1]:
        raise ValueError("incremental stale cache does not continue the base session")
    if int(incremental_metadata["context_steps"][0]) != int(
        base_metadata["context_steps"][-1]
    ) + 1:
        raise ValueError("incremental stale-cache context steps are not contiguous")

    base_graph = _load_graph(base_root, base_manifest, base_metadata)
    incremental_graph = _load_graph(
        incremental_root, incremental_manifest, incremental_metadata
    )
    temporary.mkdir(parents=True)
    base_dates = len(base_metadata["target_dates"])
    incremental_dates = len(incremental_metadata["target_dates"])
    combined_metadata = {
        "target_dates": np.concatenate(
            [base_metadata["target_dates"], incremental_metadata["target_dates"]]
        ),
        "context_dates": np.concatenate(
            [base_metadata["context_dates"], incremental_metadata["context_dates"]]
        ),
        "context_steps": np.concatenate(
            [base_metadata["context_steps"], incremental_metadata["context_steps"]]
        ),
        "tickers": base_metadata["tickers"],
        "state_feature_names": base_metadata["state_feature_names"],
    }
    metadata_path = temporary / "dates_and_tickers.npz"
    np.savez_compressed(metadata_path, **combined_metadata)

    for filename in ARRAY_FILES:
        _copy_concatenated_array(
            base_root / filename,
            incremental_root / filename,
            temporary / filename,
            base_dates,
            incremental_dates,
        )

    base_edges = int(base_graph["edge_offsets"][-1])
    edge_offsets = np.concatenate(
        [
            base_graph["edge_offsets"],
            incremental_graph["edge_offsets"][1:] + base_edges,
        ]
    )
    edge_index = np.concatenate(
        [base_graph["edge_index"], incremental_graph["edge_index"]], axis=1
    )
    edge_weight = np.concatenate(
        [base_graph["edge_weight"], incremental_graph["edge_weight"]]
    )
    graph_path = temporary / "causal_stock_graph.npz"
    np.savez_compressed(
        graph_path,
        target_dates=combined_metadata["target_dates"],
        context_dates=combined_metadata["context_dates"],
        edge_offsets=edge_offsets,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    files = {
        "dates_and_tickers": _file_record(metadata_path),
        "context_latent_f16": _file_record(
            temporary / "context_latent_f16.npy"
        ),
        "predicted_delta_f16": _file_record(
            temporary / "predicted_delta_f16.npy"
        ),
        "predicted_state_f32": _file_record(
            temporary / "predicted_state_f32.npy"
        ),
        "causal_stock_graph": _file_record(graph_path),
    }
    edges_per_date = np.diff(edge_offsets)
    manifest = {
        **base_manifest,
        "checkpoint": incremental_manifest["checkpoint"],
        "target_end": incremental_manifest["target_end"],
        "dates": int(base_dates + incremental_dates),
        "device": "immutable_base_plus_incremental_merge",
        "files": files,
        "stock_graph": {
            **base_manifest["stock_graph"],
            "minimum_edges_per_date": int(edges_per_date.min()),
            "median_edges_per_date": float(np.median(edges_per_date)),
            "maximum_edges_per_date": int(edges_per_date.max()),
            "total_edges": int(edge_offsets[-1]),
            "signed_weights": bool((edge_weight < 0.0).any()),
        },
        "causality": {
            **base_manifest["causality"],
            "base_payload_prefix_copied_without_recomputation": True,
        },
        "merge_provenance": {
            "contract": "immutable_stale_jepa_cache_append_v1",
            "base_manifest": str(base_root / "manifest.json"),
            "base_manifest_sha256": file_sha256(base_root / "manifest.json"),
            "incremental_manifest": str(incremental_root / "manifest.json"),
            "incremental_manifest_sha256": file_sha256(
                incremental_root / "manifest.json"
            ),
            "base_dates": int(base_dates),
            "incremental_dates": int(incremental_dates),
            "prefix_arrays_copied_without_recomputation": True,
        },
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append a strict-OOS stale JEPA cache without recomputing its base."
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--incremental-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    manifest = merge_caches(
        Path(args.base_dir), Path(args.incremental_dir), output_root
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_root),
                "manifest_sha256": file_sha256(output_root / "manifest.json"),
                "dates": manifest["dates"],
                "target_start": manifest["target_start"],
                "target_end": manifest["target_end"],
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
