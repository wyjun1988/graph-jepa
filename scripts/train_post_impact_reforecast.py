from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.intraday_trajectory import INTRADAY_TRAJECTORY_TARGET_NAMES
from stock_v2.post_impact_reforecast import (
    CausalPostImpactReforecast,
    GRAPH_MESSAGE_FUSIONS,
    RegressionMetricAccumulator,
    RobustArrayScaler,
    fit_robust_array_scaler,
    grouped_node_correlation_loss,
    impact_weighted_multitask_loss,
    normalize_with_mask,
)
from stock_v2.surprise_reforecast import (
    SURPRISE_STATISTIC_NAMES,
    ResidualSurpriseCalibration,
    fit_residual_surprise_calibration,
    summarize_residual_surprise,
)


TRAINING_CONTRACT = "strict_oos_causal_post_impact_reforecast_v2"
CONTEXT_PLACEBO_LOOKBACK_SESSIONS = 20
STALE_CACHE_CONTRACT_V1 = "strict_oos_stale_daily_jepa_h1_v1"
STALE_CACHE_CONTRACT_V2 = "strict_oos_stale_daily_jepa_h1_v2"
NODE_SURPRISE_FEATURE_NAMES = (
    "return_from_prev_close",
    "return_5m",
    "cumulative_volume_shock_20",
    "recent_volume_5m_shock_20",
    "realized_absolute_return_15m_shock_20",
)
SURPRISE_GRAPH_MESSAGE_MODES = (
    "surprise_disabled",
    "surprise_causal",
    "surprise_node_permuted",
    "surprise_own_permuted",
)
GRAPH_MESSAGE_MODES = (
    "none",
    "disabled",
    "causal",
    "node_permuted",
    *SURPRISE_GRAPH_MESSAGE_MODES,
)
DAILY_CONTEXT_PLACEBO_MODES = ("none", "all", "latent_only")
EVALUATION_SCOPES = ("full", "validation_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal post-impact reforecast head on strict-OOS JEPA states."
    )
    parser.add_argument("--day-release-dir", required=True)
    parser.add_argument("--stale-cache-dir", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument(
        "--evaluation-scope",
        choices=EVALUATION_SCOPES,
        default="full",
        help="Skip all test-label evaluation while selecting a validation candidate.",
    )
    parser.add_argument("--variant", choices=["direct", "state", "latent"], required=True)
    parser.add_argument(
        "--shuffle-daily-context",
        action="store_true",
        help="Legacy alias for --daily-context-placebo-mode=all.",
    )
    parser.add_argument(
        "--daily-context-placebo-mode",
        choices=DAILY_CONTEXT_PLACEBO_MODES,
        default="none",
        help=(
            "Causally replace all stale daily context, or only JEPA latent and "
            "rollout inputs, with a prior-session control."
        ),
    )
    parser.add_argument(
        "--disable-stale-graph",
        action="store_true",
        help="Ablate graph coherence while retaining the exact same v2 stale cache.",
    )
    parser.add_argument(
        "--permute-stale-graph-nodes",
        action="store_true",
        help="Use a deterministic node-label permutation as a graph-alignment placebo.",
    )
    parser.add_argument(
        "--graph-message-mode",
        choices=GRAPH_MESSAGE_MODES,
        default="none",
        help=(
            "Optionally aggregate current intraday features or node-level JEPA "
            "surprises over the stale directed graph while preserving identity."
        ),
    )
    parser.add_argument(
        "--graph-message-fusion",
        choices=GRAPH_MESSAGE_FUSIONS,
        default="shared",
    )
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--base-summary")
    parser.add_argument(
        "--freeze-base-for-message-adapter",
        action="store_true",
        help=(
            "Load and freeze a no-message base while training only a "
            "long-horizon message residual adapter."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--latent-projection-dim", type=int, default=192)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-days", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--systemic-loss-weight", type=float, default=0.50)
    parser.add_argument(
        "--endpoint-return-weight",
        type=float,
        default=2.0,
        help="Relative node-loss weight for endpoint_return across every horizon.",
    )
    parser.add_argument(
        "--close-horizon-weight",
        type=float,
        default=1.0,
        help="Relative node-loss weight for the close horizon.",
    )
    parser.add_argument("--surprise-quantile", type=float, default=0.80)
    parser.add_argument("--realized-impact-quantile", type=float, default=0.80)
    parser.add_argument("--surprise-loss-weight", type=float, default=1.0)
    parser.add_argument("--realized-impact-loss-weight", type=float, default=1.0)
    parser.add_argument("--maximum-event-weight", type=float, default=4.0)
    parser.add_argument("--minimum-surprise-nodes", type=int, default=100)
    parser.add_argument(
        "--post-shock-correlation-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Grouped cross-sectional endpoint-correlation loss applied only "
            "after point-in-time observed shocks."
        ),
    )
    parser.add_argument(
        "--post-shock-correlation-horizons",
        default="15m,30m,60m",
    )
    parser.add_argument("--post-shock-lookback-minutes", type=int, default=30)
    parser.add_argument("--post-shock-minimum-nodes", type=int, default=100)
    parser.add_argument("--scaler-samples-per-day", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-day-shards",
        action="store_true",
        help="Keep checksum-verified day shards in host RAM after their first load.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_evaluation_enabled(scope: str) -> bool:
    resolved = str(scope)
    if resolved not in EVALUATION_SCOPES:
        raise ValueError(f"evaluation scope must be one of {EVALUATION_SCOPES}")
    return resolved == "full"


def _context_date_splits(
    train_dates: list[str],
    validation_dates: list[str],
    test_dates: list[str],
    evaluation_scope: str,
) -> tuple[list[str], ...]:
    if _test_evaluation_enabled(evaluation_scope):
        return train_dates, validation_dates, test_dates
    return train_dates, validation_dates


def _device(value: str) -> torch.device:
    requested = str(value).lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def _amp_dtype(value: str) -> torch.dtype | None:
    return {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


class DayRelease:
    def __init__(self, root: Path, *, cache: bool = False) -> None:
        self.root = root
        self.cache = bool(cache)
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self.manifest_path = root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("assembly_contract") != (
            "time_major_intraday_post_impact_days_v1"
        ):
            raise ValueError("unsupported intraday day-release contract")
        if self.manifest.get("live_orders_allowed") is not False:
            raise ValueError("day release must explicitly prohibit live orders")
        if self.manifest.get("promotion_eligible") is not False:
            raise ValueError("day release must remain research-only")
        if self.manifest.get("transactional_publish") is not True:
            raise ValueError("day release was not transactionally published")
        if self.manifest.get("portable_payload_paths") is not True:
            raise ValueError("day release payload paths are not portable")
        if self.manifest.get("source_trajectory_contract") != (
            "kiwoom_raw_rolling_post_impact_trajectory_v2"
        ):
            raise ValueError("unsupported source trajectory contract")
        required_causality = {
            "close_auction_required_for_close_target",
            "decision_bar_excluded_from_start_labelled_inputs",
            "endpoint_return_requires_exact_horizon_price_only",
            "mfe_mae_use_only_post_decision_bars",
            "missing_bars_never_filled",
            "path_targets_require_contiguous_future_bars",
            "same_clock_baselines_shifted_one_session",
            "targets_begin_strictly_after_decision",
        }
        causality = self.manifest.get("causality")
        if not isinstance(causality, dict) or any(
            causality.get(claim) is not True for claim in required_causality
        ):
            raise ValueError("day release is missing the corrected target causality evidence")
        metadata_record = self.manifest["metadata"]
        metadata_path = root / metadata_record["path"]
        if file_sha256(metadata_path) != metadata_record["sha256"]:
            raise ValueError("day release metadata checksum mismatch")
        with np.load(metadata_path) as metadata:
            self.tickers = tuple(str(value) for value in metadata["tickers"].tolist())
            self.feature_names = tuple(
                str(value) for value in metadata["feature_names"].tolist()
            )
            self.horizon_labels = tuple(
                str(value) for value in metadata["horizon_labels"].tolist()
            )
            self.target_names = tuple(
                str(value) for value in metadata["target_names"].tolist()
            )
            self.systemic_target_names = tuple(
                str(value) for value in metadata["systemic_target_names"].tolist()
            )
        self.records = {
            str(record["date"]): record for record in self.manifest["day_shards"]
        }
        if len(self.records) != len(self.manifest["day_shards"]):
            raise ValueError("day release contains duplicate day-shard dates")
        if int(self.manifest.get("days", -1)) != len(self.records):
            raise ValueError("day release day count does not match its shard records")
        if int(self.manifest.get("stocks", -1)) != len(self.tickers):
            raise ValueError("day release stock count does not match its metadata")
        self._verified: set[str] = set()

    @property
    def dates(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))

    def load(self, date: str) -> dict[str, np.ndarray]:
        cached = self._cache.get(str(date))
        if cached is not None:
            return cached
        record = self.records[str(date)]
        path = self.root / record["path"]
        if date not in self._verified:
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"day shard checksum mismatch: {date}")
            self._verified.add(date)
        with np.load(path) as bundle:
            loaded = {name: bundle[name] for name in bundle.files}
        if self.cache:
            self._cache[str(date)] = loaded
        return loaded


class StaleCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.cache_contract = str(self.manifest.get("cache_contract", ""))
        if self.cache_contract not in {
            STALE_CACHE_CONTRACT_V1,
            STALE_CACHE_CONTRACT_V2,
        }:
            raise ValueError("unsupported stale JEPA cache contract")
        if self.manifest.get("strict_out_of_sample") is not True:
            raise ValueError("stale JEPA cache is not strict out-of-sample")
        if self.manifest.get("live_orders_allowed") is not False:
            raise ValueError("stale cache must explicitly prohibit live orders")
        required_causality = {
            "checkpoint_training_precedes_all_targets",
            "context_strictly_precedes_target_session",
            "extended_panel_prefix_matches_checkpoint_panel",
            "full_target_session_absent_from_input",
            "one_day_rollout_only",
        }
        if self.cache_contract == STALE_CACHE_CONTRACT_V2:
            required_causality.update(
                {
                    "stock_graph_aligned_to_context_rows",
                    "stock_graph_uses_context_session_or_earlier_only",
                }
            )
        causality = self.manifest.get("causality")
        if not isinstance(causality, dict) or any(
            causality.get(claim) is not True for claim in required_causality
        ):
            raise ValueError("stale cache is missing required causality evidence")
        files = self.manifest["files"]
        for record in files.values():
            path = root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"stale cache checksum mismatch: {path.name}")
        with np.load(root / files["dates_and_tickers"]["path"]) as metadata:
            self.dates = tuple(str(value) for value in metadata["target_dates"].tolist())
            self.context_dates = tuple(
                str(value) for value in metadata["context_dates"].tolist()
            )
            self.tickers = tuple(str(value) for value in metadata["tickers"].tolist())
            self.state_feature_names = tuple(
                str(value) for value in metadata["state_feature_names"].tolist()
            )
        if not self.dates or len(self.dates) != len(self.context_dates):
            raise ValueError("stale cache target and context date axes are not aligned")
        target_dates = pd.DatetimeIndex(pd.to_datetime(self.dates, errors="raise"))
        context_dates = pd.DatetimeIndex(
            pd.to_datetime(self.context_dates, errors="raise")
        )
        if target_dates.has_duplicates or not target_dates.is_monotonic_increasing:
            raise ValueError("stale cache target dates must be unique and sorted")
        if not np.asarray(context_dates < target_dates).all():
            raise ValueError("stale cache context must strictly precede every target")
        if not self.tickers or len(set(self.tickers)) != len(self.tickers):
            raise ValueError("stale cache tickers must be non-empty and unique")
        if not self.state_feature_names or len(set(self.state_feature_names)) != len(
            self.state_feature_names
        ):
            raise ValueError("stale cache state features must be non-empty and unique")
        self.date_to_row = {date: index for index, date in enumerate(self.dates)}
        self.context = np.load(
            root / files["context_latent_f16"]["path"], mmap_mode="r"
        )
        self.delta = np.load(
            root / files["predicted_delta_f16"]["path"], mmap_mode="r"
        )
        self.state = np.load(
            root / files["predicted_state_f32"]["path"], mmap_mode="r"
        )
        expected_prefix = (len(self.dates), len(self.tickers))
        if (
            self.context.ndim != 3
            or self.delta.ndim != 3
            or self.context.shape != self.delta.shape
            or self.context.shape[:2] != expected_prefix
        ):
            raise ValueError("stale cache latent arrays do not match date and ticker axes")
        if self.state.shape != expected_prefix + (len(self.state_feature_names),):
            raise ValueError("stale cache state array does not match its metadata axes")
        self.edge_offsets: np.ndarray | None = None
        self.edge_index: np.ndarray | None = None
        self.edge_weight: np.ndarray | None = None
        if self.cache_contract == STALE_CACHE_CONTRACT_V2:
            graph_record = files.get("causal_stock_graph")
            graph_manifest = self.manifest.get("stock_graph")
            if not isinstance(graph_record, dict) or not isinstance(
                graph_manifest, dict
            ):
                raise ValueError("v2 stale cache is missing its causal stock graph")
            required_graph_claims = {
                "date_aligned",
                "directed",
                "self_loops_excluded",
                "external_nodes_excluded",
            }
            if any(
                graph_manifest.get(claim) is not True
                for claim in required_graph_claims
            ):
                raise ValueError("v2 stale cache graph claims are incomplete")
            with np.load(root / graph_record["path"]) as graph:
                graph_target_dates = tuple(
                    str(value) for value in graph["target_dates"].tolist()
                )
                graph_context_dates = tuple(
                    str(value) for value in graph["context_dates"].tolist()
                )
                raw_offsets = np.asarray(graph["edge_offsets"])
                raw_index = np.asarray(graph["edge_index"])
                raw_weight = np.asarray(graph["edge_weight"])
            if graph_target_dates != self.dates or graph_context_dates != self.context_dates:
                raise ValueError("stale cache graph dates do not match cache metadata")
            if not np.issubdtype(raw_offsets.dtype, np.integer):
                raise ValueError("stale cache graph offsets must be integers")
            if not np.issubdtype(raw_index.dtype, np.integer):
                raise ValueError("stale cache graph indices must be integers")
            self.edge_offsets = raw_offsets.astype(np.int64, copy=False)
            self.edge_index = raw_index.astype(np.int64, copy=False)
            self.edge_weight = raw_weight.astype(np.float32, copy=False)
            if (
                self.edge_offsets.shape != (len(self.dates) + 1,)
                or self.edge_offsets[0] != 0
                or (np.diff(self.edge_offsets) <= 0).any()
            ):
                raise ValueError("stale cache graph offsets are invalid or contain empty dates")
            edge_count = int(self.edge_offsets[-1])
            if self.edge_index.shape != (2, edge_count) or self.edge_weight.shape != (
                edge_count,
            ):
                raise ValueError("stale cache graph arrays are not aligned")
            if (
                (self.edge_index < 0).any()
                or (self.edge_index >= len(self.tickers)).any()
                or (self.edge_index[0] == self.edge_index[1]).any()
                or not np.isfinite(self.edge_weight).all()
                or (self.edge_weight == 0.0).any()
            ):
                raise ValueError("stale cache graph contains invalid stock edges")
            if int(graph_manifest.get("total_edges", -1)) != edge_count:
                raise ValueError("stale cache graph edge count disagrees with its manifest")
            if bool(graph_manifest.get("signed_weights")) != bool(
                (self.edge_weight < 0.0).any()
            ):
                raise ValueError("stale cache signed-edge claim disagrees with its payload")
        self.node_order = np.arange(len(self.tickers), dtype=np.int64)
        self.edge_node_remap = np.arange(len(self.tickers), dtype=np.int64)

    def align_tickers(self, tickers: tuple[str, ...]) -> None:
        if len(tickers) != len(self.tickers) or set(tickers) != set(self.tickers):
            raise ValueError("intraday and stale JEPA ticker sets differ")
        positions = {ticker: index for index, ticker in enumerate(self.tickers)}
        self.node_order = np.asarray([positions[ticker] for ticker in tickers], dtype=np.int64)
        self.edge_node_remap = np.empty(len(self.tickers), dtype=np.int64)
        self.edge_node_remap[self.node_order] = np.arange(
            len(self.tickers), dtype=np.int64
        )

    def state_row(self, row: int) -> np.ndarray:
        return np.asarray(self.state[int(row)])[self.node_order]

    def context_row(self, row: int) -> np.ndarray:
        return np.asarray(self.context[int(row)])[self.node_order]

    def delta_row(self, row: int) -> np.ndarray:
        return np.asarray(self.delta[int(row)])[self.node_order]

    def edge_row(self, row: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.edge_offsets is None:
            return None, None
        start = int(self.edge_offsets[int(row)])
        stop = int(self.edge_offsets[int(row) + 1])
        edge_index = self.edge_node_remap[self.edge_index[:, start:stop]]
        return edge_index, self.edge_weight[start:stop]

    def graph_summary(self) -> dict[str, Any]:
        if self.edge_offsets is None:
            return {
                "cache_contract": self.cache_contract,
                "available": False,
                "total_edges": 0,
            }
        counts = np.diff(self.edge_offsets)
        return {
            "cache_contract": self.cache_contract,
            "available": True,
            "total_edges": int(self.edge_offsets[-1]),
            "minimum_edges_per_date": int(counts.min()),
            "median_edges_per_date": float(np.median(counts)),
            "maximum_edges_per_date": int(counts.max()),
            "negative_weight_fraction": float(np.mean(self.edge_weight < 0.0)),
        }

    def audit_context_map(
        self,
        sample_dates: Iterable[str],
        context_map: dict[str, str],
    ) -> dict[str, Any]:
        requested = tuple(str(value) for value in sample_dates)
        if set(context_map) != set(requested):
            raise ValueError("daily context map does not cover the requested dates exactly")
        lag_days: list[int] = []
        same_target_count = 0
        for sample_date in requested:
            source_target_date = str(context_map[sample_date])
            source_row = self.date_to_row.get(source_target_date)
            if source_row is None:
                raise ValueError("daily context map references an absent stale-cache date")
            sample_timestamp = pd.Timestamp(sample_date).normalize()
            source_context_timestamp = pd.Timestamp(
                self.context_dates[source_row]
            ).normalize()
            if source_context_timestamp >= sample_timestamp:
                raise ValueError("daily context map exposes a non-causal context date")
            lag_days.append(int((sample_timestamp - source_context_timestamp).days))
            same_target_count += int(source_target_date == sample_date)
        return {
            "dates": len(requested),
            "same_target_date_count": same_target_count,
            "minimum_context_lag_days": min(lag_days),
            "maximum_context_lag_days": max(lag_days),
            "median_context_lag_days": float(np.median(lag_days)),
            "future_context_violations": 0,
        }


def _split_dates(
    dates: Iterable[str], train_end: str, validation_end: str, test_end: str
) -> tuple[list[str], list[str], list[str]]:
    train_boundary = pd.Timestamp(train_end).normalize()
    validation_boundary = pd.Timestamp(validation_end).normalize()
    test_boundary = pd.Timestamp(test_end).normalize()
    if not train_boundary < validation_boundary < test_boundary:
        raise ValueError("split boundaries must be strictly increasing")
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for value in sorted(dates):
        date = pd.Timestamp(value).normalize()
        if date <= train_boundary:
            train.append(value)
        elif date <= validation_boundary:
            validation.append(value)
        elif date <= test_boundary:
            test.append(value)
    if min(len(train), len(validation), len(test)) < 10:
        raise ValueError(
            f"splits are too short: train={len(train)} validation={len(validation)} test={len(test)}"
        )
    return train, validation, test


def _sample_rows(
    values: np.ndarray,
    available: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_values = values.reshape(-1, values.shape[-1])
    flat_available = available.reshape(-1, available.shape[-1])
    if len(flat_values) <= int(maximum):
        return flat_values, flat_available
    rows = np.linspace(0, len(flat_values) - 1, int(maximum)).round().astype(np.int64)
    return flat_values[rows], flat_available[rows]


def fit_scalers(
    release: DayRelease,
    stale: StaleCache,
    train_dates: list[str],
    samples_per_day: int,
) -> tuple[RobustArrayScaler, RobustArrayScaler, RobustArrayScaler, RobustArrayScaler]:
    node_values: list[np.ndarray] = []
    node_available: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    target_available: list[np.ndarray] = []
    systemic_values: list[np.ndarray] = []
    systemic_available: list[np.ndarray] = []
    stale_values: list[np.ndarray] = []
    for date in train_dates:
        day = release.load(date)
        values, available = _sample_rows(
            day["node_values"], day["node_available"].astype(bool), samples_per_day
        )
        node_values.append(values)
        node_available.append(available)
        values, available = _sample_rows(
            day["targets"],
            day["target_available"].astype(bool),
            samples_per_day,
        )
        target_values.append(values)
        target_available.append(available)
        systemic_values.append(
            day["systemic_targets"].reshape(-1, len(release.systemic_target_names))
        )
        systemic_available.append(
            day["systemic_available"].astype(bool).reshape(
                -1, len(release.systemic_target_names)
            )
        )
        stale_values.append(stale.state_row(stale.date_to_row[date]))
    node_scaler = fit_robust_array_scaler(
        np.concatenate(node_values), np.concatenate(node_available), minimum_count=100
    )
    target_scaler = fit_robust_array_scaler(
        np.concatenate(target_values),
        np.concatenate(target_available),
        minimum_count=100,
        minimum_scale=1e-5,
    )
    systemic_scaler = fit_robust_array_scaler(
        np.concatenate(systemic_values),
        np.concatenate(systemic_available),
        minimum_count=20,
        minimum_scale=1e-5,
    )
    stale_matrix = np.concatenate(stale_values)
    stale_scaler = fit_robust_array_scaler(
        stale_matrix,
        np.isfinite(stale_matrix),
        minimum_count=100,
        minimum_scale=1e-5,
    )
    return node_scaler, target_scaler, systemic_scaler, stale_scaler


def _surprise_residuals(
    release: DayRelease,
    day: dict[str, np.ndarray],
    stale_state: np.ndarray,
    state_feature_names: tuple[str, ...],
    *,
    residual_conditioned: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = day["node_values"].astype(np.float64)
    available = day["node_available"].astype(bool)
    feature_index = {name: release.feature_names.index(name) for name in release.feature_names}
    selected_names = NODE_SURPRISE_FEATURE_NAMES
    result = np.stack([values[..., feature_index[name]] for name in selected_names], axis=-1)
    valid = np.stack(
        [available[..., feature_index[name]] for name in selected_names], axis=-1
    )
    if residual_conditioned:
        state_index = {name: state_feature_names.index(name) for name in state_feature_names}
        clock = values[..., feature_index["clock_fraction"]]
        clock_valid = available[..., feature_index["clock_fraction"]]
        predicted_return = stale_state[:, state_index["return_1d"]][None]
        result[..., 0] = result[..., 0] - clock * predicted_return
        valid[..., 0] &= clock_valid & np.isfinite(predicted_return)
        observed_open = values[..., feature_index["return_from_open"]]
        observed_open_valid = available[..., feature_index["return_from_open"]]
        predicted_intraday = stale_state[:, state_index["intraday_return"]][None]
        result[..., 1] = observed_open - clock * predicted_intraday
        valid[..., 1] = observed_open_valid & clock_valid & np.isfinite(predicted_intraday)
    return result, valid


def fit_surprise_calibration(
    release: DayRelease,
    stale: StaleCache,
    train_dates: list[str],
    *,
    residual_conditioned: bool,
    quantile: float,
    min_nodes: int,
    context_map: dict[str, str],
) -> ResidualSurpriseCalibration:
    residual_blocks: list[np.ndarray] = []
    valid_blocks: list[np.ndarray] = []
    for date in train_dates:
        context_date = context_map[date]
        state = stale.state_row(stale.date_to_row[context_date])
        residuals, valid = _surprise_residuals(
            release,
            release.load(date),
            state,
            stale.state_feature_names,
            residual_conditioned=residual_conditioned,
        )
        residual_blocks.append(residuals)
        valid_blocks.append(valid)
    residuals = np.concatenate(residual_blocks)
    valid = np.concatenate(valid_blocks)
    calibration, _design = fit_residual_surprise_calibration(
        residuals,
        valid,
        np.arange(len(residuals), dtype=np.int64),
        stock_count=len(release.tickers),
        threshold_quantile=float(quantile),
        min_nodes=int(min_nodes),
    )
    return calibration


def _surprise_values(
    release: DayRelease,
    day: dict[str, np.ndarray],
    stale_state: np.ndarray,
    state_feature_names: tuple[str, ...],
    calibration: ResidualSurpriseCalibration,
    *,
    residual_conditioned: bool,
    edge_index: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
) -> np.ndarray:
    residuals, valid = _surprise_residuals(
        release,
        day,
        stale_state,
        state_feature_names,
        residual_conditioned=residual_conditioned,
    )
    result = summarize_residual_surprise(
        residuals,
        valid,
        feature_center=calibration.feature_center,
        feature_scale=calibration.feature_scale,
        stock_count=len(release.tickers),
        min_nodes=calibration.min_nodes,
        node_z_threshold=calibration.node_z_threshold,
        clip=calibration.clip,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_realized_impact_thresholds(
    release: DayRelease,
    train_dates: list[str],
    quantile: float,
) -> dict[str, np.ndarray]:
    state_index = release.systemic_target_names.index("state_change_energy")
    volume_index = release.systemic_target_names.index("volume_expansion_breadth")
    state_blocks: list[np.ndarray] = []
    volume_blocks: list[np.ndarray] = []
    state_valid: list[np.ndarray] = []
    volume_valid: list[np.ndarray] = []
    for date in train_dates:
        day = release.load(date)
        state_blocks.append(day["systemic_targets"][..., state_index])
        volume_blocks.append(day["systemic_targets"][..., volume_index])
        state_valid.append(day["systemic_available"][..., state_index].astype(bool))
        volume_valid.append(day["systemic_available"][..., volume_index].astype(bool))

    def thresholds(blocks: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
        values = np.concatenate(blocks)
        valid = np.concatenate(masks)
        result = np.ones(values.shape[1], dtype=np.float64)
        for horizon in range(values.shape[1]):
            selected = values[:, horizon][valid[:, horizon] & np.isfinite(values[:, horizon])]
            if len(selected) < 20:
                raise ValueError("too few realized impact labels for calibration")
            result[horizon] = max(float(np.quantile(selected, quantile)), 1e-6)
        return result

    return {
        "state": thresholds(state_blocks, state_valid),
        "volume": thresholds(volume_blocks, volume_valid),
    }


def _context_maps(
    splits: tuple[list[str], ...],
    *,
    shuffle: bool,
    seed: int,
) -> dict[str, str]:
    result: dict[str, str] = {}
    history: list[str] = []
    for split_index, dates in enumerate(splits):
        ordered = [str(value) for value in dates]
        if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
            raise ValueError("context-map split dates must be unique and sorted")
        generator = np.random.default_rng(int(seed) + 1009 * split_index)
        for date in ordered:
            if date in result:
                raise ValueError("context-map splits must not overlap")
            if shuffle and history:
                causal_candidates = history[-CONTEXT_PLACEBO_LOOKBACK_SESSIONS:]
                source = causal_candidates[
                    int(generator.integers(0, len(causal_candidates)))
                ]
            else:
                source = date
            result[date] = source
            history.append(date)
    return result


def _resolved_daily_context_placebo_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "daily_context_placebo_mode", "none"))
    if mode not in DAILY_CONTEXT_PLACEBO_MODES:
        raise ValueError(
            "daily context placebo mode must be one of "
            f"{DAILY_CONTEXT_PLACEBO_MODES}"
        )
    if bool(getattr(args, "shuffle_daily_context", False)):
        if mode == "latent_only":
            raise ValueError(
                "legacy full-context shuffle cannot be combined with latent-only placebo"
            )
        mode = "all"
    if mode == "latent_only" and str(getattr(args, "variant", "")) != "latent":
        raise ValueError("latent-only placebo requires the latent model variant")
    return mode


def _daily_context_maps(
    splits: tuple[list[str], ...],
    *,
    mode: str,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    if mode not in DAILY_CONTEXT_PLACEBO_MODES:
        raise ValueError(
            "daily context placebo mode must be one of "
            f"{DAILY_CONTEXT_PLACEBO_MODES}"
        )
    identity = _context_maps(splits, shuffle=False, seed=seed)
    if mode == "none":
        return identity, identity.copy()
    placebo = _context_maps(splits, shuffle=True, seed=seed)
    if mode == "all":
        return placebo, placebo.copy()
    return identity, placebo


def _stale_graph_mode(stale: StaleCache, args: argparse.Namespace) -> str:
    disabled = bool(getattr(args, "disable_stale_graph", False))
    permuted = bool(getattr(args, "permute_stale_graph_nodes", False))
    if disabled and permuted:
        raise ValueError("stale graph cannot be both disabled and node-permuted")
    available = stale.edge_offsets is not None
    if (disabled or permuted) and not available:
        raise ValueError("stale graph ablations require a v2 cache with stock edges")
    if not available:
        return "unavailable"
    if disabled:
        return "disabled"
    if permuted:
        return "node_permuted_placebo"
    return "causal"


def _stale_graph_inputs(
    stale: StaleCache,
    row: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mode = _stale_graph_mode(stale, args)
    if mode in {"unavailable", "disabled"}:
        return None, None
    edge_index, edge_weight = stale.edge_row(int(row))
    if edge_index is None or edge_weight is None:
        raise ValueError("stale graph mode requires aligned edges")
    if mode == "causal":
        return edge_index, edge_weight
    return _permuted_edge_index(stale, row, edge_index, args), edge_weight


def _permuted_edge_index(
    stale: StaleCache,
    row: int,
    edge_index: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    permutation = _node_permutation(stale, row, args)
    return permutation[edge_index]


def _node_permutation(
    stale: StaleCache,
    row: int,
    args: argparse.Namespace,
) -> np.ndarray:
    generator = np.random.default_rng(
        int(getattr(args, "seed", 0)) + 104729 * (int(row) + 1)
    )
    permutation = generator.permutation(len(stale.tickers))
    if len(permutation) > 1 and np.array_equal(
        permutation, np.arange(len(permutation))
    ):
        permutation = np.roll(permutation, 1)
    return permutation


def _graph_message_mode(stale: StaleCache, args: argparse.Namespace) -> str:
    mode = str(getattr(args, "graph_message_mode", "none"))
    if mode not in GRAPH_MESSAGE_MODES:
        raise ValueError(f"graph message mode must be one of {GRAPH_MESSAGE_MODES}")
    if mode != "none" and stale.edge_offsets is None:
        raise ValueError("graph message controls require a v2 cache with stock edges")
    return mode


def _graph_message_feature_dim(
    feature_names: Iterable[str], mode: str
) -> int:
    return len(_graph_message_feature_names(feature_names, mode))


def _graph_message_feature_names(
    feature_names: Iterable[str], mode: str
) -> tuple[str, ...]:
    names = tuple(str(value) for value in feature_names)
    resolved = str(mode)
    if resolved not in GRAPH_MESSAGE_MODES:
        raise ValueError(f"graph message mode must be one of {GRAPH_MESSAGE_MODES}")
    if resolved == "none":
        return ()
    if resolved in SURPRISE_GRAPH_MESSAGE_MODES:
        missing = set(NODE_SURPRISE_FEATURE_NAMES).difference(names)
        if missing:
            raise ValueError(
                f"node surprise graph inputs are missing features: {sorted(missing)}"
            )
        return tuple(
            [f"own_surprise:{name}" for name in NODE_SURPRISE_FEATURE_NAMES]
            + [f"neighbor_surprise:{name}" for name in NODE_SURPRISE_FEATURE_NAMES]
        )
    return names


def _graph_message_edges_used(mode: str) -> bool:
    return str(mode) in {
        "causal",
        "node_permuted",
        "surprise_causal",
        "surprise_node_permuted",
    }


def _graph_message_inputs(
    stale: StaleCache,
    row: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mode = _graph_message_mode(stale, args)
    if mode in {
        "none",
        "disabled",
        "surprise_disabled",
        "surprise_own_permuted",
    }:
        return None, None
    edge_index, edge_weight = stale.edge_row(int(row))
    if edge_index is None or edge_weight is None:
        raise ValueError("graph message mode requires aligned edges")
    if mode in {"node_permuted", "surprise_node_permuted"}:
        edge_index = _permuted_edge_index(stale, row, edge_index, args)
    return edge_index, edge_weight


def _standardized_node_surprise(
    release: DayRelease,
    day: dict[str, np.ndarray],
    stale_state: np.ndarray,
    state_feature_names: tuple[str, ...],
    calibration: ResidualSurpriseCalibration,
    *,
    residual_conditioned: bool,
) -> tuple[np.ndarray, np.ndarray]:
    residuals, valid = _surprise_residuals(
        release,
        day,
        stale_state,
        state_feature_names,
        residual_conditioned=residual_conditioned,
    )
    center = np.asarray(calibration.feature_center, dtype=np.float64)
    scale = np.asarray(calibration.feature_scale, dtype=np.float64)
    expected = (len(NODE_SURPRISE_FEATURE_NAMES),)
    if center.shape != expected or scale.shape != expected:
        raise ValueError("node surprise calibration width does not match its contract")
    standardized = np.clip(
        (residuals - center[None, None, :]) / scale[None, None, :],
        -float(calibration.clip),
        float(calibration.clip),
    )
    available = valid & np.isfinite(standardized)
    values = np.where(available, standardized, 0.0).astype(np.float32)
    return values, available


def _signed_neighbor_aggregate(
    values: np.ndarray,
    available: np.ndarray,
    edge_index: np.ndarray | None,
    edge_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    if values.ndim != 3 or values.shape != available.shape:
        raise ValueError("graph message values and masks must be [time,node,feature]")
    output = np.zeros_like(values, dtype=np.float32)
    output_available = np.zeros_like(available, dtype=bool)
    if edge_index is None or edge_weight is None:
        return output, output_available
    edges = np.asarray(edge_index, dtype=np.int64)
    weights = np.asarray(edge_weight, dtype=np.float32)
    if edges.ndim != 2 or edges.shape[0] != 2 or weights.shape != (edges.shape[1],):
        raise ValueError("graph message edges and weights are misaligned")
    node_count = values.shape[1]
    selected = (
        (edges[0] >= 0)
        & (edges[1] >= 0)
        & (edges[0] < node_count)
        & (edges[1] < node_count)
        & np.isfinite(weights)
        & (np.abs(weights) > 0.0)
    )
    if not selected.any():
        return output, output_available
    source = edges[0, selected]
    target = edges[1, selected]
    selected_weight = weights[selected]
    source_values = values[:, source, :]
    source_available = (
        available[:, source, :] & np.isfinite(source_values)
    )
    weighted = np.where(
        source_available,
        source_values * selected_weight[None, :, None],
        0.0,
    )
    normalizer = np.where(
        source_available,
        np.abs(selected_weight)[None, :, None],
        0.0,
    )
    numerator = np.zeros(
        (node_count, values.shape[0], values.shape[2]), dtype=np.float32
    )
    denominator = np.zeros_like(numerator)
    np.add.at(numerator, target, np.transpose(weighted, (1, 0, 2)))
    np.add.at(denominator, target, np.transpose(normalizer, (1, 0, 2)))
    valid = denominator > 1e-8
    aggregated = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=valid,
    )
    return (
        np.transpose(aggregated, (1, 0, 2)),
        np.transpose(valid, (1, 0, 2)),
    )


def _causal_recent_event_mask(
    timestamps_utc_ns: np.ndarray,
    event: np.ndarray,
    lookback_minutes: int,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_utc_ns, dtype=np.int64)
    selected = np.asarray(event, dtype=bool)
    if timestamps.ndim != 1 or selected.shape != timestamps.shape:
        raise ValueError("recent event timestamps and mask must be aligned vectors")
    if int(lookback_minutes) < 0:
        raise ValueError("post-shock lookback must be non-negative")
    if len(timestamps) > 1 and bool((np.diff(timestamps) <= 0).any()):
        raise ValueError("recent event timestamps must be strictly increasing")
    missing = np.iinfo(np.int64).min
    last_event = np.maximum.accumulate(
        np.where(selected, timestamps, missing)
    )
    window_ns = int(lookback_minutes) * 60 * 1_000_000_000
    return (last_event != missing) & ((timestamps - last_event) <= window_ns)


def _post_shock_correlation_horizons(
    horizon_labels: Iterable[str], args: argparse.Namespace
) -> tuple[str, ...]:
    labels = tuple(str(value) for value in horizon_labels)
    configured = tuple(
        value.strip()
        for value in str(
            getattr(args, "post_shock_correlation_horizons", "15m,30m,60m")
        ).split(",")
        if value.strip()
    )
    if not configured or len(set(configured)) != len(configured):
        raise ValueError("post-shock correlation horizons must be unique and non-empty")
    missing = set(configured).difference(labels)
    if missing:
        raise ValueError(
            f"post-shock correlation horizons are absent: {sorted(missing)}"
        )
    if "5m" in configured:
        raise ValueError("post-shock correlation loss cannot modify protected 5m")
    return configured


def _batches(dates: list[str], size: int, rng: np.random.Generator | None) -> list[list[str]]:
    order = np.arange(len(dates))
    if rng is not None:
        rng.shuffle(order)
    return [
        [dates[index] for index in order[start : start + int(size)]]
        for start in range(0, len(order), int(size))
    ]


def _pad_batch(
    release: DayRelease,
    stale: StaleCache,
    dates: list[str],
    state_context_map: dict[str, str],
    latent_context_map: dict[str, str],
    observed_calibration: ResidualSurpriseCalibration,
    model_calibration: ResidualSurpriseCalibration,
    node_scaler: RobustArrayScaler,
    stale_scaler: RobustArrayScaler,
    impact_thresholds: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    days = [release.load(date) for date in dates]
    maximum_time = max(len(day["timestamps_utc_ns"]) for day in days)
    batch = len(days)
    nodes = len(release.tickers)
    features = len(release.feature_names)
    horizons = len(release.horizon_labels)
    targets = len(release.target_names)
    systemic_targets = len(release.systemic_target_names)
    graph_message_mode = _graph_message_mode(stale, args)
    graph_message_dim = _graph_message_feature_dim(
        release.feature_names, graph_message_mode
    )
    output = {
        "node_values": np.zeros((batch, maximum_time, nodes, features), np.float32),
        "node_available": np.zeros((batch, maximum_time, nodes, features), bool),
        "targets": np.zeros((batch, maximum_time, nodes, horizons, targets), np.float32),
        "target_available": np.zeros(
            (batch, maximum_time, nodes, horizons, targets), bool
        ),
        "systemic_targets": np.zeros(
            (batch, maximum_time, horizons, systemic_targets), np.float32
        ),
        "systemic_available": np.zeros(
            (batch, maximum_time, horizons, systemic_targets), bool
        ),
        "stale_state": np.zeros((batch, nodes, len(stale.state_feature_names)), np.float32),
        "context_latent": np.zeros((batch, nodes, stale.context.shape[-1]), np.float16),
        "predicted_delta": np.zeros((batch, nodes, stale.delta.shape[-1]), np.float16),
        "surprise": np.zeros(
            (batch, maximum_time, len(SURPRISE_STATISTIC_NAMES)), np.float32
        ),
        "event_weights": np.ones((batch, maximum_time, horizons), np.float32),
        "observed_surprise": np.zeros((batch, maximum_time), bool),
        "post_shock_recent": np.zeros((batch, maximum_time), bool),
        "realized_impact": np.zeros((batch, maximum_time, horizons), bool),
    }
    if graph_message_mode != "none":
        output["graph_neighbor_values"] = np.zeros(
            (batch, maximum_time, nodes, graph_message_dim), np.float32
        )
        output["graph_neighbor_available"] = np.zeros(
            (batch, maximum_time, nodes, graph_message_dim), bool
        )
    energy_index = SURPRISE_STATISTIC_NAMES.index("systemic_surprise_energy")
    state_index = release.systemic_target_names.index("state_change_energy")
    volume_index = release.systemic_target_names.index("volume_expansion_breadth")
    node_center = torch.as_tensor(node_scaler.center, dtype=torch.float32)
    node_scale = torch.as_tensor(node_scaler.scale, dtype=torch.float32)
    stale_center = torch.as_tensor(stale_scaler.center, dtype=torch.float32)
    stale_scale = torch.as_tensor(stale_scaler.scale, dtype=torch.float32)
    for batch_index, (date, day) in enumerate(zip(dates, days)):
        count = len(day["timestamps_utc_ns"])
        state_context_date = state_context_map[date]
        latent_context_date = latent_context_map[date]
        state_stale_row = stale.date_to_row[state_context_date]
        latent_stale_row = stale.date_to_row[latent_context_date]
        stale_state = np.asarray(
            stale.state_row(state_stale_row), dtype=np.float32
        )
        stale_edge_index, stale_edge_weight = _stale_graph_inputs(
            stale, state_stale_row, args
        )
        message_edge_index, message_edge_weight = _graph_message_inputs(
            stale, state_stale_row, args
        )
        observed = _surprise_values(
            release,
            day,
            stale_state,
            stale.state_feature_names,
            observed_calibration,
            residual_conditioned=False,
        )
        model_surprise = _surprise_values(
            release,
            day,
            stale_state,
            stale.state_feature_names,
            model_calibration,
            residual_conditioned=args.variant != "direct",
            edge_index=stale_edge_index,
            edge_weight=stale_edge_weight,
        )
        node_available = day["node_available"].astype(bool)
        normalized_node_values = normalize_with_mask(
            torch.as_tensor(day["node_values"]),
            torch.as_tensor(node_available),
            node_center,
            node_scale,
        ).numpy()
        output["node_values"][batch_index, :count] = normalized_node_values
        output["node_available"][batch_index, :count] = node_available
        if graph_message_mode != "none":
            if graph_message_mode in SURPRISE_GRAPH_MESSAGE_MODES:
                own_values, own_available = _standardized_node_surprise(
                    release,
                    day,
                    stale_state,
                    stale.state_feature_names,
                    model_calibration,
                    residual_conditioned=args.variant != "direct",
                )
                if graph_message_mode == "surprise_own_permuted":
                    permutation = _node_permutation(
                        stale, state_stale_row, args
                    )
                    own_values = own_values[:, permutation, :]
                    own_available = own_available[:, permutation, :]
                neighbor_values, neighbor_available = _signed_neighbor_aggregate(
                    own_values,
                    own_available,
                    message_edge_index,
                    message_edge_weight,
                )
                message_values = np.concatenate(
                    (own_values, neighbor_values), axis=-1
                )
                message_available = np.concatenate(
                    (own_available, neighbor_available), axis=-1
                )
            else:
                message_values, message_available = _signed_neighbor_aggregate(
                    normalized_node_values,
                    node_available,
                    message_edge_index,
                    message_edge_weight,
                )
            output["graph_neighbor_values"][batch_index, :count] = message_values
            output["graph_neighbor_available"][
                batch_index, :count
            ] = message_available
        output["targets"][batch_index, :count] = day["targets"]
        output["target_available"][batch_index, :count] = day[
            "target_available"
        ].astype(bool)
        output["systemic_targets"][batch_index, :count] = day["systemic_targets"]
        output["systemic_available"][batch_index, :count] = day[
            "systemic_available"
        ].astype(bool)
        output["stale_state"][batch_index] = (
            (stale_state - stale_scaler.center) / stale_scaler.scale
        ).clip(-10.0, 10.0)
        output["context_latent"][batch_index] = stale.context_row(
            latent_stale_row
        )
        output["predicted_delta"][batch_index] = stale.delta_row(
            latent_stale_row
        )
        output["surprise"][batch_index, :count] = model_surprise
        observed_energy = observed[:, energy_index]
        surprise_ratio = observed_energy / max(
            observed_calibration.energy_threshold, 1e-6
        )
        observed_event = observed_energy >= observed_calibration.energy_threshold
        output["observed_surprise"][batch_index, :count] = observed_event
        output["post_shock_recent"][batch_index, :count] = (
            _causal_recent_event_mask(
                day["timestamps_utc_ns"],
                observed_event,
                int(getattr(args, "post_shock_lookback_minutes", 30)),
            )
        )
        surprise_weight = 1.0 + float(args.surprise_loss_weight) * np.maximum(
            surprise_ratio - 1.0, 0.0
        )
        systemic = day["systemic_targets"]
        systemic_valid = day["systemic_available"].astype(bool)
        state_ratio = np.where(
            systemic_valid[..., state_index],
            systemic[..., state_index] / impact_thresholds["state"][None],
            0.0,
        )
        volume_ratio = np.where(
            systemic_valid[..., volume_index],
            systemic[..., volume_index] / impact_thresholds["volume"][None],
            0.0,
        )
        state_ratio = np.nan_to_num(state_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        volume_ratio = np.nan_to_num(volume_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        realized_ratio = np.maximum(state_ratio, volume_ratio)
        realized_event = realized_ratio >= 1.0
        output["realized_impact"][batch_index, :count] = realized_event
        realized_weight = 1.0 + float(args.realized_impact_loss_weight) * np.maximum(
            realized_ratio - 1.0, 0.0
        )
        output["event_weights"][batch_index, :count] = np.minimum(
            surprise_weight[:, None] * realized_weight,
            float(args.maximum_event_weight),
        )
    return output


def _systemic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    scaler: RobustArrayScaler,
    event_weights: torch.Tensor,
) -> torch.Tensor:
    center = torch.as_tensor(scaler.center, dtype=target.dtype, device=target.device)
    scale = torch.as_tensor(scaler.scale, dtype=target.dtype, device=target.device)
    normalized_target = (target - center) / scale.clamp_min(1e-8)
    valid = available.bool() & torch.isfinite(normalized_target) & torch.isfinite(prediction)
    residual = torch.where(
        valid, prediction - normalized_target, torch.zeros_like(prediction)
    )
    absolute = residual.abs()
    loss = torch.where(absolute <= 1.0, 0.5 * residual.square(), absolute - 0.5)
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    weights = event_weights[..., None].expand_as(loss)
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _objective_weights(
    release: DayRelease,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    horizon_defaults = {
        "5m": 2.0,
        "15m": 1.5,
        "30m": 1.25,
        "60m": 1.0,
        "close": float(args.close_horizon_weight),
    }
    horizon_weights = torch.as_tensor(
        [horizon_defaults.get(label, 1.0) for label in release.horizon_labels],
        dtype=torch.float32,
        device=device,
    )
    target_weights = {
        "endpoint_return": float(args.endpoint_return_weight),
        "mfe": 1.5,
        "mae": 1.5,
        "realized_absolute_return": 1.0,
        "future_range": 1.0,
        "time_to_peak_fraction": 0.5,
        "time_to_trough_fraction": 0.5,
        "future_volume_shock_20": 1.5,
    }
    selected = [target_weights.get(name, 1.0) for name in release.target_names]
    if (
        not torch.isfinite(horizon_weights).all()
        or bool((horizon_weights <= 0.0).any())
        or not np.isfinite(selected).all()
        or any(value <= 0.0 for value in selected)
    ):
        raise ValueError("objective weights must be finite and positive")
    return horizon_weights, target_weights


def _tensor_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, device=device)
        for name, value in batch.items()
        if name not in {"observed_surprise", "realized_impact"}
    }


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    accumulator = RegressionMetricAccumulator()
    accumulator.update(prediction, target)
    return accumulator.metrics()


def _daily_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def finite_values(name: str) -> np.ndarray:
        return np.asarray(
            [row[name] for row in rows if math.isfinite(float(row[name]))],
            dtype=np.float64,
        )

    skill = finite_values("skill_vs_zero_mse")
    pearson = finite_values("pearson")
    mae = finite_values("mae")
    direction = finite_values("direction_accuracy")
    return {
        "days": len(rows),
        "total_count": int(sum(int(row["count"]) for row in rows)),
        "median_mae": float(np.median(mae)) if len(mae) else float("nan"),
        "median_skill_vs_zero_mse": (
            float(np.median(skill)) if len(skill) else float("nan")
        ),
        "positive_skill_day_fraction": (
            float(np.mean(skill > 0.0)) if len(skill) else float("nan")
        ),
        "median_pearson": (
            float(np.median(pearson)) if len(pearson) else float("nan")
        ),
        "median_direction_accuracy": (
            float(np.median(direction)) if len(direction) else float("nan")
        ),
    }


def _prevalence_record(positive: int, valid: int) -> dict[str, int | float]:
    return {
        "positive_timestamps": int(positive),
        "valid_timestamps": int(valid),
        "fraction": float(positive / valid) if valid else float("nan"),
    }


def _strict_json_value(value: Any) -> Any:
    """Return a JSON-safe tree without non-standard NaN/Infinity tokens."""
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _set_model_mode(
    model: CausalPostImpactReforecast,
    *,
    training: bool,
    frozen_message_adapter: bool,
) -> None:
    model.train(training)
    if training and frozen_message_adapter:
        model.eval()
        for name, module in model.named_modules():
            if name.startswith("message_"):
                module.train(True)


def _load_frozen_message_base(
    model: CausalPostImpactReforecast,
    args: argparse.Namespace,
    release: DayRelease,
    stale: StaleCache,
) -> dict[str, Any] | None:
    enabled = bool(getattr(args, "freeze_base_for_message_adapter", False))
    base_checkpoint_value = getattr(args, "base_checkpoint", None)
    base_summary_value = getattr(args, "base_summary", None)
    if not enabled:
        if base_checkpoint_value or base_summary_value:
            raise ValueError(
                "base checkpoint inputs require --freeze-base-for-message-adapter"
            )
        return None
    if args.graph_message_fusion != "long_horizon_residual":
        raise ValueError(
            "frozen message adapters require long-horizon residual fusion"
        )
    if args.graph_message_mode not in {
        "surprise_disabled",
        "surprise_own_permuted",
    }:
        raise ValueError(
            "frozen message adapters require aligned or node-permuted own surprise"
        )
    if not base_checkpoint_value or not base_summary_value:
        raise ValueError("frozen message adapters require base checkpoint and summary")

    base_checkpoint_path = Path(base_checkpoint_value)
    base_summary_path = Path(base_summary_value)
    base_checkpoint = torch.load(
        base_checkpoint_path, map_location="cpu", weights_only=False
    )
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    if (
        base_summary.get("evaluation_scope") != "validation_only"
        or base_summary.get("test_evaluated") is not False
        or base_summary.get("test") is not None
        or base_summary.get("test_loss") is not None
    ):
        raise ValueError("frozen message base is not validation-only")
    if base_summary.get("live_orders_allowed") is not False or base_summary.get(
        "promotion_eligible"
    ) is not False:
        raise ValueError("frozen message base is not research-only")
    checkpoint_sha256 = file_sha256(base_checkpoint_path)
    if base_summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("frozen message base checkpoint hash mismatch")
    base_args = argparse.Namespace(**base_checkpoint["args"])
    expected_args = {
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "test_end": args.test_end,
        "variant": args.variant,
        "hidden_dim": args.hidden_dim,
        "latent_projection_dim": args.latent_projection_dim,
        "temporal_layers": args.temporal_layers,
        "dropout": args.dropout,
        "seed": args.seed,
        "daily_context_placebo_mode": args.daily_context_placebo_mode,
        "disable_stale_graph": args.disable_stale_graph,
        "permute_stale_graph_nodes": args.permute_stale_graph_nodes,
        "surprise_quantile": args.surprise_quantile,
        "realized_impact_quantile": args.realized_impact_quantile,
        "minimum_surprise_nodes": args.minimum_surprise_nodes,
        "scaler_samples_per_day": args.scaler_samples_per_day,
        "systemic_loss_weight": args.systemic_loss_weight,
        "endpoint_return_weight": args.endpoint_return_weight,
        "close_horizon_weight": args.close_horizon_weight,
    }
    for name, expected in expected_args.items():
        if getattr(base_args, name) != expected:
            raise ValueError(f"frozen message base argument mismatch: {name}")
    if getattr(base_args, "graph_message_mode", "none") != "none":
        raise ValueError("frozen message base must not use graph messages")
    if getattr(base_args, "graph_message_fusion", "shared") != "shared":
        raise ValueError("frozen message base must use the shared no-message architecture")
    if getattr(base_args, "evaluation_scope", "full") != "validation_only":
        raise ValueError("frozen message base checkpoint evaluated test labels")
    if tuple(base_checkpoint["feature_names"]) != release.feature_names:
        raise ValueError("frozen message base feature contract mismatch")
    if tuple(base_checkpoint["state_feature_names"]) != stale.state_feature_names:
        raise ValueError("frozen message base state contract mismatch")
    if tuple(base_checkpoint["horizon_labels"]) != release.horizon_labels:
        raise ValueError("frozen message base horizon contract mismatch")
    if tuple(base_checkpoint["target_names"]) != release.target_names:
        raise ValueError("frozen message base target contract mismatch")
    if tuple(base_checkpoint["systemic_target_names"]) != release.systemic_target_names:
        raise ValueError("frozen message base systemic contract mismatch")
    base_inputs = base_summary.get("inputs")
    if not isinstance(base_inputs, dict):
        raise ValueError("frozen message base summary is missing input lineage")
    expected_inputs = {
        "day_release_manifest_sha256": file_sha256(release.manifest_path),
        "stale_cache_manifest_sha256": file_sha256(stale.manifest_path),
    }
    for name, expected in expected_inputs.items():
        if base_inputs.get(name) != expected:
            raise ValueError(f"frozen message base input mismatch: {name}")

    candidate_state = model.state_dict()
    adapter_keys = {name for name in candidate_state if name.startswith("message_")}
    base_state = base_checkpoint["model_state"]
    if set(base_state) != set(candidate_state).difference(adapter_keys):
        raise ValueError("frozen message base parameter contract mismatch")
    incompatible = model.load_state_dict(base_state, strict=False)
    if set(incompatible.missing_keys) != adapter_keys or incompatible.unexpected_keys:
        raise ValueError("frozen message base did not load exactly")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("message_"))
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    if trainable <= 0 or frozen <= 0:
        raise ValueError("frozen message adapter parameter partition is empty")
    return {
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_sha256": checkpoint_sha256,
        "base_summary": str(base_summary_path),
        "base_summary_sha256": file_sha256(base_summary_path),
        "base_graph_message_mode": "none",
        "base_evaluation_scope": "validation_only",
        "protected_horizons": ["5m"],
        "trainable_adapter_parameters": int(trainable),
        "frozen_base_parameters": int(frozen),
    }


def run_epoch(
    model: CausalPostImpactReforecast,
    release: DayRelease,
    stale: StaleCache,
    dates: list[str],
    state_context_map: dict[str, str],
    latent_context_map: dict[str, str],
    observed_calibration: ResidualSurpriseCalibration,
    model_calibration: ResidualSurpriseCalibration,
    node_scaler: RobustArrayScaler,
    target_scaler: RobustArrayScaler,
    systemic_scaler: RobustArrayScaler,
    stale_scaler: RobustArrayScaler,
    impact_thresholds: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    _set_model_mode(
        model,
        training=training,
        frozen_message_adapter=bool(
            getattr(args, "freeze_base_for_message_adapter", False)
        ),
    )
    generator = np.random.default_rng(args.seed + epoch) if training else None
    batches = _batches(dates, args.batch_days, generator)
    total_losses: list[float] = []
    multitask_losses: list[float] = []
    node_losses: list[float] = []
    systemic_losses: list[float] = []
    correlation_losses: list[float] = []
    subset_names = (
        "all",
        "observed_surprise",
        "realized_impact",
        "surprise_and_impact",
    )
    node_statistics = {
        target: {
            label: {
                subset: RegressionMetricAccumulator() for subset in subset_names
            }
            for label in release.horizon_labels
        }
        for target in release.target_names
    }
    systemic_statistics = {
        target: {
            label: {
                subset: RegressionMetricAccumulator() for subset in subset_names
            }
            for label in release.horizon_labels
        }
        for target in release.systemic_target_names
    }
    daily_endpoint_rows: dict[str, list[dict[str, Any]]] = {
        label: [] for label in release.horizon_labels
    }
    endpoint_index = release.target_names.index("endpoint_return")
    state_change_index = release.systemic_target_names.index("state_change_energy")
    volume_expansion_index = release.systemic_target_names.index(
        "volume_expansion_breadth"
    )
    observed_event_count = 0
    observed_valid_count = 0
    realized_event_counts = {
        label: 0 for label in release.horizon_labels
    }
    realized_valid_counts = {
        label: 0 for label in release.horizon_labels
    }
    joint_event_counts = {label: 0 for label in release.horizon_labels}
    joint_valid_counts = {label: 0 for label in release.horizon_labels}
    amp_dtype = _amp_dtype(args.amp_dtype)
    target_scale = torch.as_tensor(
        target_scaler.scale, dtype=torch.float32, device=device
    )
    horizon_weights, target_weights = _objective_weights(release, args, device)
    correlation_weight = float(
        getattr(args, "post_shock_correlation_loss_weight", 0.0)
    )
    correlation_horizons = _post_shock_correlation_horizons(
        release.horizon_labels, args
    )
    correlation_horizon_mask = torch.as_tensor(
        [label in correlation_horizons for label in release.horizon_labels],
        dtype=torch.bool,
        device=device,
    )
    for date_batch in batches:
        numpy_batch = _pad_batch(
            release,
            stale,
            date_batch,
            state_context_map,
            latent_context_map,
            observed_calibration,
            model_calibration,
            node_scaler,
            stale_scaler,
            impact_thresholds,
            args,
        )
        batch = _tensor_batch(numpy_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_enabled = amp_dtype is not None and device.type in {"cuda", "mps"}
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=autocast_enabled,
        ):
            prediction = model(
                batch["node_values"],
                batch["node_available"],
                stale_state=batch["stale_state"] if args.variant != "direct" else None,
                context_latent=batch["context_latent"].float() if args.variant == "latent" else None,
                predicted_delta=batch["predicted_delta"].float() if args.variant == "latent" else None,
                surprise_values=batch["surprise"],
                graph_neighbor_values=(
                    batch["graph_neighbor_values"].float()
                    if "graph_neighbor_values" in batch
                    else None
                ),
                graph_neighbor_available=batch.get("graph_neighbor_available"),
            )
            node_loss, _components = impact_weighted_multitask_loss(
                prediction.node.float(),
                batch["targets"].float(),
                batch["target_available"],
                target_scale=target_scale,
                horizon_weights=horizon_weights,
                event_weights=batch["event_weights"].float(),
                target_weights=target_weights,
                target_names=release.target_names,
            )
            systemic_loss = _systemic_loss(
                prediction.systemic.float(),
                batch["systemic_targets"].float(),
                batch["systemic_available"],
                systemic_scaler,
                batch["event_weights"].float(),
            )
            if correlation_weight > 0.0:
                correlation_loss = grouped_node_correlation_loss(
                    prediction.node[..., endpoint_index].float(),
                    batch["targets"][..., endpoint_index].float(),
                    batch["target_available"][..., endpoint_index],
                    batch["post_shock_recent"],
                    correlation_horizon_mask,
                    minimum_nodes=int(
                        getattr(args, "post_shock_minimum_nodes", 100)
                    ),
                )
            else:
                correlation_loss = node_loss.new_tensor(0.0)
            multitask_loss = (
                node_loss + float(args.systemic_loss_weight) * systemic_loss
            )
            loss = multitask_loss + correlation_weight * correlation_loss
        node_requested = batch["target_available"].bool()
        systemic_requested = batch["systemic_available"].bool()
        if node_requested.any() and not torch.isfinite(
            prediction.node[node_requested]
        ).all():
            raise FloatingPointError("node prediction is non-finite on supervised cells")
        if systemic_requested.any() and not torch.isfinite(
            prediction.systemic[systemic_requested]
        ).all():
            raise FloatingPointError("systemic prediction is non-finite on supervised cells")
        if not torch.isfinite(loss):
            diagnostics = {
                "node_loss": float(node_loss.detach().cpu()),
                "systemic_loss": float(systemic_loss.detach().cpu()),
                "post_shock_correlation_loss": float(
                    correlation_loss.detach().cpu()
                ),
                "node_prediction_finite": float(
                    torch.isfinite(prediction.node).float().mean().detach().cpu()
                ),
                "systemic_prediction_finite": float(
                    torch.isfinite(prediction.systemic).float().mean().detach().cpu()
                ),
                "event_weight_finite": float(
                    torch.isfinite(batch["event_weights"]).float().mean().detach().cpu()
                ),
                "target_scale_finite": float(
                    torch.isfinite(target_scale).float().mean().detach().cpu()
                ),
            }
            raise FloatingPointError(
                "non-finite post-impact loss: " + json.dumps(diagnostics, sort_keys=True)
            )
        if training:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("post-impact gradient norm is non-finite")
            optimizer.step()
        total_losses.append(float(loss.detach().cpu()))
        multitask_losses.append(float(multitask_loss.detach().cpu()))
        node_losses.append(float(node_loss.detach().cpu()))
        systemic_losses.append(float(systemic_loss.detach().cpu()))
        correlation_losses.append(float(correlation_loss.detach().cpu()))
        if not training:
            node_prediction = prediction.node.detach().float().cpu().numpy()
            target = numpy_batch["targets"]
            target_valid = numpy_batch["target_available"]
            time_valid = numpy_batch["node_available"].any(axis=(2, 3))
            observed_event = numpy_batch["observed_surprise"] & time_valid
            observed_event_count += int(observed_event.sum())
            observed_valid_count += int(time_valid.sum())
            for horizon, label in enumerate(release.horizon_labels):
                realized_valid = (
                    numpy_batch["systemic_available"][
                        ..., horizon, state_change_index
                    ]
                    | numpy_batch["systemic_available"][
                        ..., horizon, volume_expansion_index
                    ]
                ) & time_valid
                realized_event = (
                    numpy_batch["realized_impact"][..., horizon]
                    & realized_valid
                )
                joint_valid = realized_valid & time_valid
                joint_event = realized_event & observed_event
                realized_event_counts[label] += int(realized_event.sum())
                realized_valid_counts[label] += int(realized_valid.sum())
                joint_event_counts[label] += int(joint_event.sum())
                joint_valid_counts[label] += int(joint_valid.sum())
                observed = np.broadcast_to(
                    numpy_batch["observed_surprise"][:, :, None],
                    target_valid[..., horizon, endpoint_index].shape,
                )
                impact = np.broadcast_to(
                    numpy_batch["realized_impact"][:, :, None, horizon],
                    target_valid[..., horizon, endpoint_index].shape,
                )
                subset_masks = {
                    "all": np.ones_like(observed, dtype=bool),
                    "observed_surprise": observed,
                    "realized_impact": impact,
                    "surprise_and_impact": observed & impact,
                }
                for target_position, target_name in enumerate(release.target_names):
                    valid = target_valid[..., horizon, target_position]
                    for subset, subset_mask in subset_masks.items():
                        node_statistics[target_name][label][subset].update(
                            node_prediction[..., horizon, target_position],
                            target[..., horizon, target_position],
                            valid & subset_mask,
                        )
                for batch_index, date in enumerate(date_batch):
                    valid = target_valid[
                        batch_index, ..., horizon, endpoint_index
                    ]
                    daily = _regression_metrics(
                        np.where(
                            valid,
                            node_prediction[
                                batch_index, ..., horizon, endpoint_index
                            ],
                            np.nan,
                        ),
                        np.where(
                            valid,
                            target[batch_index, ..., horizon, endpoint_index],
                            np.nan,
                        ),
                    )
                    daily_endpoint_rows[label].append({"date": date, **daily})

            systemic_prediction = (
                prediction.systemic.detach().float().cpu().numpy()
                * systemic_scaler.scale[None, None, None, :]
                + systemic_scaler.center[None, None, None, :]
            )
            systemic_target = numpy_batch["systemic_targets"]
            systemic_valid = numpy_batch["systemic_available"]
            for horizon, label in enumerate(release.horizon_labels):
                observed = numpy_batch["observed_surprise"]
                impact = numpy_batch["realized_impact"][..., horizon]
                subset_masks = {
                    "all": np.ones_like(observed, dtype=bool),
                    "observed_surprise": observed,
                    "realized_impact": impact,
                    "surprise_and_impact": observed & impact,
                }
                for target_position, target_name in enumerate(
                    release.systemic_target_names
                ):
                    valid = systemic_valid[..., horizon, target_position]
                    for subset, subset_mask in subset_masks.items():
                        systemic_statistics[target_name][label][subset].update(
                            systemic_prediction[..., horizon, target_position],
                            systemic_target[..., horizon, target_position],
                            valid & subset_mask,
                        )
    metrics: dict[str, Any] = {}
    if not training:
        node_metrics = {
            target: {
                label: {
                    subset: accumulator.metrics()
                    for subset, accumulator in subsets.items()
                }
                for label, subsets in horizons.items()
            }
            for target, horizons in node_statistics.items()
        }
        systemic_metrics = {
            target: {
                label: {
                    subset: accumulator.metrics()
                    for subset, accumulator in subsets.items()
                }
                for label, subsets in horizons.items()
            }
            for target, horizons in systemic_statistics.items()
        }
        metrics = {
            "objective_components": {
                "total": float(np.mean(total_losses)),
                "multitask": float(np.mean(multitask_losses)),
                "node": float(np.mean(node_losses)),
                "systemic": float(np.mean(systemic_losses)),
                "post_shock_correlation": float(
                    np.mean(correlation_losses)
                ),
                "post_shock_correlation_weight": correlation_weight,
                "post_shock_correlation_horizons": list(
                    correlation_horizons
                ),
            },
            "node_endpoint": node_metrics["endpoint_return"],
            "node_targets": node_metrics,
            "systemic_state_change_energy": {
                label: systemic_metrics["state_change_energy"][label]["all"]
                for label in release.horizon_labels
            },
            "systemic_targets": systemic_metrics,
            "daily_node_endpoint": {
                label: _daily_metric_summary(rows)
                for label, rows in daily_endpoint_rows.items()
            },
            "daily_node_endpoint_rows": daily_endpoint_rows,
            "event_prevalence": {
                "observed_surprise": _prevalence_record(
                    observed_event_count, observed_valid_count
                ),
                "realized_impact": {
                    label: _prevalence_record(
                        realized_event_counts[label], realized_valid_counts[label]
                    )
                    for label in release.horizon_labels
                },
                "surprise_and_impact": {
                    label: _prevalence_record(
                        joint_event_counts[label], joint_valid_counts[label]
                    )
                    for label in release.horizon_labels
                },
            },
        }
    return float(np.mean(total_losses)), metrics


def _scaler_json(scaler: RobustArrayScaler) -> dict[str, list[float]]:
    return {
        "center": np.asarray(scaler.center, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale, dtype=float).tolist(),
    }


def main() -> int:
    run_started = time.perf_counter()
    args = parse_args()
    daily_context_placebo_mode = _resolved_daily_context_placebo_mode(args)
    args.daily_context_placebo_mode = daily_context_placebo_mode
    args.shuffle_daily_context = daily_context_placebo_mode == "all"
    if min(args.epochs, args.patience, args.batch_days, args.scaler_samples_per_day) <= 0:
        raise ValueError("epochs, patience, batch days, and scaler samples must be positive")
    if not 0.5 < args.surprise_quantile < 1.0 or not 0.5 < args.realized_impact_quantile < 1.0:
        raise ValueError("impact quantiles must be in (0.5, 1)")
    if (
        not math.isfinite(float(args.endpoint_return_weight))
        or float(args.endpoint_return_weight) <= 0.0
        or not math.isfinite(float(args.close_horizon_weight))
        or float(args.close_horizon_weight) <= 0.0
    ):
        raise ValueError("objective weights must be finite and positive")
    if (
        args.graph_message_fusion == "long_horizon_residual"
        and not args.freeze_base_for_message_adapter
    ):
        raise ValueError(
            "long-horizon residual fusion requires --freeze-base-for-message-adapter"
        )
    if (
        not math.isfinite(float(args.post_shock_correlation_loss_weight))
        or float(args.post_shock_correlation_loss_weight) < 0.0
    ):
        raise ValueError("post-shock correlation weight must be finite and non-negative")
    if args.post_shock_lookback_minutes < 0 or args.post_shock_minimum_nodes < 3:
        raise ValueError("post-shock correlation window or minimum nodes is invalid")
    if args.post_shock_correlation_loss_weight > 0.0 and (
        not args.freeze_base_for_message_adapter
        or args.graph_message_fusion != "long_horizon_residual"
    ):
        raise ValueError(
            "post-shock correlation loss requires the frozen long-horizon adapter"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    release = DayRelease(
        Path(args.day_release_dir), cache=bool(args.cache_day_shards)
    )
    post_shock_correlation_horizons = _post_shock_correlation_horizons(
        release.horizon_labels, args
    )
    stale = StaleCache(Path(args.stale_cache_dir))
    stale.align_tickers(release.tickers)
    stale_graph_mode = _stale_graph_mode(stale, args)
    graph_message_mode = _graph_message_mode(stale, args)
    common_dates = sorted(set(release.dates) & set(stale.dates))
    train_dates, validation_dates, test_dates = _split_dates(
        common_dates, args.train_end, args.validation_end, args.test_end
    )
    test_evaluated = _test_evaluation_enabled(args.evaluation_scope)
    context_date_splits = _context_date_splits(
        train_dates,
        validation_dates,
        test_dates,
        args.evaluation_scope,
    )
    state_context_map, latent_context_map = _daily_context_maps(
        context_date_splits,
        mode=daily_context_placebo_mode,
        seed=args.seed,
    )
    selected_dates = [date for split in context_date_splits for date in split]
    context_map_audit = stale.audit_context_map(
        selected_dates, state_context_map
    )
    context_map_audit["contract"] = (
        f"causal_historical_placebo_last_"
        f"{CONTEXT_PLACEBO_LOOKBACK_SESSIONS}_sessions_v1"
        if daily_context_placebo_mode == "all"
        else "identity_strict_oos_stale_h1_v1"
    )
    latent_context_map_audit = stale.audit_context_map(
        selected_dates, latent_context_map
    )
    latent_context_map_audit["contract"] = (
        f"causal_historical_latent_placebo_last_"
        f"{CONTEXT_PLACEBO_LOOKBACK_SESSIONS}_sessions_v1"
        if daily_context_placebo_mode in {"all", "latent_only"}
        else "identity_strict_oos_stale_h1_v1"
    )
    node_scaler, target_scaler, systemic_scaler, stale_scaler = fit_scalers(
        release, stale, train_dates, args.scaler_samples_per_day
    )
    observed_calibration = fit_surprise_calibration(
        release,
        stale,
        train_dates,
        residual_conditioned=False,
        quantile=args.surprise_quantile,
        min_nodes=args.minimum_surprise_nodes,
        context_map={date: date for date in common_dates},
    )
    model_calibration = (
        observed_calibration
        if args.variant == "direct"
        else fit_surprise_calibration(
            release,
            stale,
            train_dates,
            residual_conditioned=True,
            quantile=args.surprise_quantile,
            min_nodes=args.minimum_surprise_nodes,
            context_map=state_context_map,
        )
    )
    impact_thresholds = fit_realized_impact_thresholds(
        release, train_dates, args.realized_impact_quantile
    )
    model = CausalPostImpactReforecast(
        node_feature_dim=len(release.feature_names),
        stale_state_dim=len(stale.state_feature_names),
        latent_dim=int(stale.context.shape[-1]),
        horizons=release.horizon_labels,
        systemic_target_dim=len(release.systemic_target_names),
        variant=args.variant,
        hidden_dim=args.hidden_dim,
        latent_projection_dim=args.latent_projection_dim,
        temporal_layers=args.temporal_layers,
        dropout=args.dropout,
        surprise_dim=len(SURPRISE_STATISTIC_NAMES),
        graph_message_dim=_graph_message_feature_dim(
            release.feature_names, graph_message_mode
        ),
        graph_message_fusion=args.graph_message_fusion,
        target_names=release.target_names,
        output_scales=target_scaler.scale,
    ).to(device)
    frozen_message_base = _load_frozen_message_base(model, args, release, stale)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("post-impact model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup_seconds = time.perf_counter() - run_started
    training_started = time.perf_counter()
    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    initial_validation_loss: float | None = None
    initial_validation_objective_components: dict[str, Any] | None = None
    if frozen_message_base is not None:
        initial_validation_loss, initial_validation_metrics = run_epoch(
            model,
            release,
            stale,
            validation_dates,
            state_context_map,
            latent_context_map,
            observed_calibration,
            model_calibration,
            node_scaler,
            target_scaler,
            systemic_scaler,
            stale_scaler,
            impact_thresholds,
            args,
            device,
            None,
            0,
        )
        initial_validation_objective_components = initial_validation_metrics.get(
            "objective_components"
        )
        best_score = initial_validation_loss
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        train_loss, _train_metrics = run_epoch(
            model, release, stale, train_dates,
            state_context_map, latent_context_map,
            observed_calibration, model_calibration, node_scaler, target_scaler,
            systemic_scaler, stale_scaler, impact_thresholds, args, device,
            optimizer, epoch,
        )
        validation_loss, validation_metrics = run_epoch(
            model, release, stale, validation_dates,
            state_context_map, latent_context_map,
            observed_calibration, model_calibration, node_scaler, target_scaler,
            systemic_scaler, stale_scaler, impact_thresholds, args, device,
            None, epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "duration_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"epoch={epoch:02d} train={train_loss:.6f} validation={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_score - 1e-5:
            best_score = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation candidate")
    model.load_state_dict(best_state)
    final_evaluation_started = time.perf_counter()
    validation_loss, validation_metrics = run_epoch(
        model, release, stale, validation_dates,
        state_context_map, latent_context_map,
        observed_calibration, model_calibration, node_scaler, target_scaler,
        systemic_scaler, stale_scaler, impact_thresholds, args, device, None, best_epoch,
    )
    if test_evaluated:
        test_loss, test_metrics = run_epoch(
            model, release, stale, test_dates,
            state_context_map, latent_context_map,
            observed_calibration, model_calibration, node_scaler, target_scaler,
            systemic_scaler, stale_scaler, impact_thresholds, args, device, None, best_epoch,
        )
    else:
        test_loss, test_metrics = None, None
    final_evaluation_seconds = time.perf_counter() - final_evaluation_started
    checkpoint_path = output_dir / "post_impact_reforecast.pt"
    torch.save(
        {
            "model_state": best_state,
            "args": vars(args),
            "feature_names": release.feature_names,
            "graph_message_feature_names": _graph_message_feature_names(
                release.feature_names, graph_message_mode
            ),
            "graph_message_fusion": args.graph_message_fusion,
            "frozen_message_base": frozen_message_base,
            "state_feature_names": stale.state_feature_names,
            "horizon_labels": release.horizon_labels,
            "target_names": release.target_names,
            "systemic_target_names": release.systemic_target_names,
            "node_scaler": _scaler_json(node_scaler),
            "target_scaler": _scaler_json(target_scaler),
            "systemic_scaler": _scaler_json(systemic_scaler),
            "stale_scaler": _scaler_json(stale_scaler),
            "observed_surprise_calibration": asdict(observed_calibration),
            "model_surprise_calibration": asdict(model_calibration),
            "impact_thresholds": {
                name: values.tolist() for name, values in impact_thresholds.items()
            },
            "daily_context_placebo_mode": daily_context_placebo_mode,
            "context_map_audit": context_map_audit,
            "latent_context_map_audit": latent_context_map_audit,
        },
        checkpoint_path,
    )
    summary = {
        "schema_version": 1,
        "training_contract": TRAINING_CONTRACT,
        "variant": args.variant,
        "evaluation_scope": args.evaluation_scope,
        "test_evaluated": test_evaluated,
        "shuffle_daily_context": bool(args.shuffle_daily_context),
        "daily_context_placebo_mode": daily_context_placebo_mode,
        "objective_weights": {
            "endpoint_return": float(args.endpoint_return_weight),
            "close_horizon": float(args.close_horizon_weight),
            "post_shock_correlation": float(
                args.post_shock_correlation_loss_weight
            ),
        },
        "post_shock_correlation_contract": {
            "enabled": bool(args.post_shock_correlation_loss_weight > 0.0),
            "horizons": list(post_shock_correlation_horizons),
            "lookback_minutes": int(args.post_shock_lookback_minutes),
            "minimum_nodes": int(args.post_shock_minimum_nodes),
            "point_in_time_observed_shock_only": True,
            "future_labels_used_for_event_selection": False,
        },
        "stale_stock_graph_used": bool(
            stale_graph_mode in {"causal", "node_permuted_placebo"}
        ),
        "stale_stock_graph_mode": stale_graph_mode,
        "graph_message_input_enabled": graph_message_mode != "none",
        "graph_message_edges_used": _graph_message_edges_used(graph_message_mode),
        "graph_message_mode": graph_message_mode,
        "graph_message_fusion": args.graph_message_fusion,
        "freeze_base_for_message_adapter": bool(
            args.freeze_base_for_message_adapter
        ),
        "frozen_message_base": frozen_message_base,
        "graph_message_feature_names": list(
            _graph_message_feature_names(release.feature_names, graph_message_mode)
        ),
        "context_map_audit": context_map_audit,
        "latent_context_map_audit": latent_context_map_audit,
        "strict_out_of_sample_stale_jepa": True,
        "splits": {
            "train": {"start": train_dates[0], "end": train_dates[-1], "days": len(train_dates)},
            "validation": {"start": validation_dates[0], "end": validation_dates[-1], "days": len(validation_dates)},
            "test": {"start": test_dates[0], "end": test_dates[-1], "days": len(test_dates)},
        },
        "best_epoch": best_epoch,
        "initial_validation_loss": initial_validation_loss,
        "initial_validation_objective_components": (
            initial_validation_objective_components
        ),
        "best_validation_loss": best_score,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "runtime": {
            "cache_day_shards": bool(args.cache_day_shards),
            "cached_day_count": len(release._cache),
            "device": str(device),
            "amp_dtype": args.amp_dtype,
            "model_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "frozen_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if not parameter.requires_grad
            ),
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
            "setup_seconds": setup_seconds,
            "training_seconds": training_seconds,
            "final_evaluation_seconds": final_evaluation_seconds,
            "total_seconds": time.perf_counter() - run_started,
        },
        "inputs": {
            "day_release_manifest_sha256": file_sha256(release.manifest_path),
            "stale_cache_manifest_sha256": file_sha256(stale.manifest_path),
            "stale_cache_contract": stale.cache_contract,
            "stale_stock_graph": stale.graph_summary(),
            "stale_stock_graph_used": bool(
                stale_graph_mode in {"causal", "node_permuted_placebo"}
            ),
            "stale_stock_graph_mode": stale_graph_mode,
            "graph_message_input_enabled": graph_message_mode != "none",
            "graph_message_edges_used": _graph_message_edges_used(
                graph_message_mode
            ),
            "graph_message_mode": graph_message_mode,
            "graph_message_fusion": args.graph_message_fusion,
            "frozen_message_base": frozen_message_base,
            "graph_message_feature_names": list(
                _graph_message_feature_names(
                    release.feature_names, graph_message_mode
                )
            ),
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    strict_summary = _strict_json_value(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            strict_summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    console_summary = {
        "status": "complete",
        "variant": args.variant,
        "evaluation_scope": args.evaluation_scope,
        "test_evaluated": test_evaluated,
        "shuffle_daily_context": bool(args.shuffle_daily_context),
        "daily_context_placebo_mode": daily_context_placebo_mode,
        "stale_stock_graph_used": bool(
            stale_graph_mode in {"causal", "node_permuted_placebo"}
        ),
        "stale_stock_graph_mode": stale_graph_mode,
        "graph_message_input_enabled": graph_message_mode != "none",
        "graph_message_edges_used": _graph_message_edges_used(graph_message_mode),
        "graph_message_mode": graph_message_mode,
        "graph_message_fusion": args.graph_message_fusion,
        "freeze_base_for_message_adapter": bool(
            args.freeze_base_for_message_adapter
        ),
        "post_shock_correlation_loss_weight": float(
            args.post_shock_correlation_loss_weight
        ),
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "checkpoint": str(checkpoint_path),
        "summary": str(output_dir / "summary.json"),
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    print(
        json.dumps(
            _strict_json_value(console_summary),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
