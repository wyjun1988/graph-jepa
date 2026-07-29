from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd


MANIFEST_FLOAT_QUANTIZATION = 1e-4


def _update_quantized_array_digest(
    digest: Any,
    name: str,
    values: np.ndarray,
) -> None:
    """Hash arrays canonically across CPU architectures and numeric libraries."""

    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    if name == "available_mask":
        digest.update(b"bool-u8")
        digest.update(np.ascontiguousarray(array > 0.5, dtype=np.uint8).tobytes())
        return
    if np.issubdtype(array.dtype, np.floating):
        digest.update(f"quantized-{MANIFEST_FLOAT_QUANTIZATION:g}-i8".encode("ascii"))
        flat = array.reshape(-1)
        chunk_size = 1_000_000
        for start in range(0, flat.size, chunk_size):
            chunk = np.asarray(flat[start : start + chunk_size], dtype=np.float64)
            finite = np.isfinite(chunk)
            quantized = np.empty(chunk.shape, dtype="<i8")
            quantized[finite] = np.rint(
                chunk[finite] / MANIFEST_FLOAT_QUANTIZATION
            ).astype("<i8")
            quantized[np.isnan(chunk)] = np.iinfo(np.int64).min
            quantized[np.isposinf(chunk)] = np.iinfo(np.int64).min + 1
            quantized[np.isneginf(chunk)] = np.iinfo(np.int64).min + 2
            digest.update(quantized.tobytes())
        return
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    digest.update(str(canonical_dtype).encode("ascii"))
    digest.update(canonical.tobytes())


def quantized_array_sha256(name: str, values: np.ndarray) -> str:
    """Return the same architecture-tolerant array digest used by schema v3."""

    digest = hashlib.sha256()
    _update_quantized_array_digest(digest, name, np.asarray(values))
    return digest.hexdigest()


def build_training_data_diagnostics(features, train_end: str) -> dict[str, object]:
    """Expose date and feature digests for cross-host panel mismatch audits."""

    train_mask = np.asarray(features.dates <= pd.Timestamp(train_end), dtype=bool)
    if not train_mask.any():
        raise ValueError(f"no feature rows on or before train_end={train_end}")
    dates = pd.DatetimeIndex(features.dates[train_mask])
    values = np.asarray(features.features)[train_mask]
    available = np.asarray(features.available_mask)[train_mask]
    date_rows = []
    for index, date in enumerate(dates):
        date_rows.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "features_sha256": quantized_array_sha256("features", values[index]),
                "available_mask_sha256": quantized_array_sha256(
                    "available_mask", available[index]
                ),
            }
        )
    feature_rows = []
    for index, feature_name in enumerate(features.feature_names):
        feature_rows.append(
            {
                "feature": str(feature_name),
                "features_sha256": quantized_array_sha256(
                    "features", values[:, :, index]
                ),
                "available_mask_sha256": quantized_array_sha256(
                    "available_mask", available[:, :, index]
                ),
            }
        )
    return {
        "schema_version": 1,
        "float_quantization": MANIFEST_FLOAT_QUANTIZATION,
        "train_end": str(pd.Timestamp(train_end).date()),
        "dates": int(len(dates)),
        "nodes": int(values.shape[1]),
        "features": int(values.shape[2]),
        "component_sha256": {
            "features": quantized_array_sha256("features", values),
            "available_mask": quantized_array_sha256(
                "available_mask", available
            ),
        },
        "date_digests": date_rows,
        "feature_digests": feature_rows,
    }


def build_training_data_manifest(
    features,
    train_end: str,
    schema_version: int = 3,
) -> dict[str, object]:
    """Fingerprint the exact observed training panel used by a checkpoint."""

    if schema_version not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported training data manifest schema_version={schema_version}")
    train_mask = np.asarray(features.dates <= pd.Timestamp(train_end), dtype=bool)
    if not train_mask.any():
        raise ValueError(f"no feature rows on or before train_end={train_end}")
    labels: dict[str, object] = {
        "schema_version": int(schema_version),
        "train_end": str(pd.Timestamp(train_end).date()),
        "dates": [str(date.date()) for date in features.dates[train_mask]],
        "tickers": list(features.tickers),
        "node_tickers": list(features.node_tickers or features.tickers),
        "feature_names": list(features.feature_names),
        "stock_node_count": int(features.tradable_count),
    }
    if schema_version >= 2:
        labels["event_theme_names"] = list(getattr(features, "event_theme_names", []) or [])
        labels["has_static_edges"] = bool(
            getattr(features, "static_edge_index", None) is not None
            and getattr(features, "static_edge_weight", None) is not None
        )
    if schema_version >= 3:
        labels["float_quantization"] = MANIFEST_FLOAT_QUANTIZATION
    if schema_version >= 4:
        target_paths = getattr(features, "target_return_paths", None)
        if not target_paths:
            raise ValueError("schema v4 requires target_return_paths")
        labels["target_return_path_horizons"] = sorted(
            int(horizon) for horizon in target_paths
        )
    digest = hashlib.sha256(
        json.dumps(labels, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    arrays: list[tuple[str, np.ndarray]] = [
        ("features", features.features[train_mask]),
        ("available_mask", features.available_mask[train_mask]),
    ]
    if schema_version >= 2:
        theme_exposure = getattr(features, "event_theme_exposure", None)
        arrays.append(
            (
                "event_theme_exposure",
                np.zeros((0,), dtype=np.float32)
                if theme_exposure is None
                else np.asarray(theme_exposure)[train_mask],
            )
        )
        arrays.extend(
            [
                (
                    "static_edge_index",
                    np.zeros((2, 0), dtype=np.int64)
                    if getattr(features, "static_edge_index", None) is None
                    else np.asarray(features.static_edge_index),
                ),
                (
                    "static_edge_weight",
                    np.zeros((0,), dtype=np.float32)
                    if getattr(features, "static_edge_weight", None) is None
                    else np.asarray(features.static_edge_weight),
                ),
            ]
        )
    if schema_version >= 4:
        for horizon in labels["target_return_path_horizons"]:
            arrays.append(
                (
                    f"target_return_path_h{horizon}",
                    np.asarray(features.target_return_paths[int(horizon)])[train_mask],
                )
            )
    for name, values in arrays:
        if schema_version >= 3:
            _update_quantized_array_digest(digest, name, values)
            continue
        array = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return {**labels, "sha256": digest.hexdigest()}


def validate_checkpoint_panel(
    checkpoint: Mapping[str, Any],
    features,
    train_end: str,
    *,
    allow_unverified_legacy: bool = False,
) -> None:
    """Reject a panel that differs from the one used to train a checkpoint."""

    expected_tickers = list(checkpoint.get("tickers", []))
    expected_nodes = list(checkpoint.get("node_tickers", []))
    expected_features = list(checkpoint.get("feature_names", []))
    if expected_tickers and expected_tickers != list(features.tickers):
        raise ValueError("checkpoint ticker order does not match the reconstructed feature panel")
    if expected_nodes and expected_nodes != list(features.node_tickers or []):
        raise ValueError("checkpoint node order does not match the reconstructed feature panel")
    if expected_features and expected_features != list(features.feature_names):
        raise ValueError("checkpoint feature order does not match the reconstructed feature panel")

    expected_manifest = checkpoint.get("train_data_manifest")
    if not expected_manifest:
        if allow_unverified_legacy:
            return
        raise ValueError(
            "checkpoint has no training data manifest; only explicitly allowed exploratory runs may use it"
        )
    schema_version = int(expected_manifest.get("schema_version", 1))
    actual_manifest = build_training_data_manifest(
        features,
        train_end,
        schema_version=schema_version,
    )
    if expected_manifest.get("sha256") != actual_manifest.get("sha256"):
        raise ValueError(
            "reconstructed training data does not match the checkpoint manifest "
            f"(expected={expected_manifest.get('sha256')}, actual={actual_manifest.get('sha256')})"
        )
