"""Recompute per-node prospective predictions and labels from immutable inputs.

The D+1 reconciliation records only scalar aggregates: `masked_node_metrics`
collapses all 500 nodes into count/mse/pearson/sufficient-statistics per
(model, horizon, target). That is enough for the frozen endpoint gates, which
pool every available node, but it cannot answer a conditional question such as
"among the nodes that had already been shocked at the decision timestamp, does
the latent tell us which shocks develop into large moves".

The raw material for that question is nevertheless preserved: the prediction
artifact stores `node_prediction` with shape (models, stocks, horizons,
targets), and the day release stores the matching per-node targets and
availability masks. Both are immutable and hash-pinned. This module joins them
back into a per-node frame so that a new read-only auditor can apply any
conditioning or weighting without touching a frozen contract, a pinned source,
or the ledger.

Read-only by construction: nothing here writes to the ledger, the artifacts, or
any contract. It imports the pinned reconciliation module for artifact loading
and schema assertions so that a schema drift fails here exactly as it would in
D+1, and it must never edit that module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from scripts.reconcile_post_impact_prospective_ledger import (  # noqa: E402
    _assert_artifact_schema,
    _load_artifact,
)
from scripts.train_post_impact_reforecast import DayRelease  # noqa: E402

KST_OFFSET_NS = 9 * 3600 * 10**9
SHOCK_FEATURE = "realized_absolute_return_15m_shock_20"


def decision_clock_minute(decision_timestamp_utc_ns: int) -> int:
    """KST minute-of-day for a decision timestamp, matching the ledger field."""
    minutes = (int(decision_timestamp_utc_ns) + KST_OFFSET_NS) // (60 * 10**9)
    return int(minutes % (24 * 60))


def _decision_index(day: Mapping[str, np.ndarray], decision_timestamp_utc_ns: int) -> int:
    timestamps = np.asarray(day["timestamps_utc_ns"], dtype=np.int64)
    locations = np.flatnonzero(timestamps == int(decision_timestamp_utc_ns))
    if len(locations) != 1:
        raise ValueError(
            "prospective decision timestamp does not map to exactly one label row"
        )
    return int(locations[0])


def _feature_index(release: DayRelease, name: str) -> int:
    metadata = np.load(release.root / "metadata.npz", allow_pickle=True)
    names = [str(value) for value in metadata["feature_names"]]
    if name not in names:
        raise ValueError(f"day release lacks the {name} feature channel")
    return names.index(name)


def load_session_node_frame(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
    release: DayRelease,
    targets: tuple[str, ...] | None = None,
    horizons: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Join one ledger record's predictions to its labels, one row per node cell.

    Columns: session, clock_minute, model, ticker, horizon, target, prediction,
    actual, available, decision_price, shock_magnitude, shock_available.

    `available` is the day release's target mask combined with finiteness, i.e.
    exactly the mask D+1 applies before aggregating. Rows are kept even when
    unavailable so a caller can audit coverage; filter on `available` before
    scoring.
    """
    if record.get("source_mode") != "live_read_only":
        raise ValueError("only live read-only records count as prospective evidence")
    session = str(record["session"])
    if session not in release.records:
        raise ValueError("prospective session is absent from the label release")

    arrays = _load_artifact(record, artifact_root)
    model_names, horizon_labels, target_names, _systemic = _assert_artifact_schema(
        arrays, release
    )
    tickers = [str(value) for value in arrays["tickers"].tolist()]

    decision_ns = int(record["decision_timestamp_utc_ns"])
    day = release.load(session)
    index = _decision_index(day, decision_ns)

    node_prediction = np.asarray(arrays["node_prediction"], dtype=np.float64)
    actual = np.asarray(day["targets"][index], dtype=np.float64)
    actual_available = np.asarray(day["target_available"][index], dtype=bool)
    decision_price = np.asarray(arrays["latest_decision_price"], dtype=np.float64)

    shock_index = _feature_index(release, SHOCK_FEATURE)
    shock_magnitude = np.asarray(
        day["node_values"][index][:, shock_index], dtype=np.float64
    )
    shock_available = np.asarray(
        day["node_available"][index][:, shock_index], dtype=bool
    )

    wanted_targets = tuple(target_names) if targets is None else tuple(targets)
    wanted_horizons = tuple(horizon_labels) if horizons is None else tuple(horizons)
    for name in wanted_targets:
        if name not in target_names:
            raise ValueError(f"artifact lacks target {name}")
    for name in wanted_horizons:
        if name not in horizon_labels:
            raise ValueError(f"artifact lacks horizon {name}")

    clock = decision_clock_minute(decision_ns)
    if int(record.get("decision_clock_minute_kst", clock)) != clock:
        raise ValueError("ledger decision clock disagrees with its timestamp")

    frames: list[pd.DataFrame] = []
    for model_index, model in enumerate(model_names):
        for horizon in wanted_horizons:
            horizon_index = horizon_labels.index(horizon)
            for target in wanted_targets:
                target_index = target_names.index(target)
                prediction = node_prediction[model_index, :, horizon_index, target_index]
                observed = actual[:, horizon_index, target_index]
                mask = (
                    actual_available[:, horizon_index, target_index]
                    & np.isfinite(prediction)
                    & np.isfinite(observed)
                )
                frames.append(
                    pd.DataFrame(
                        {
                            "session": session,
                            "clock_minute": clock,
                            "model": model,
                            "ticker": tickers,
                            "horizon": horizon,
                            "target": target,
                            "prediction": prediction,
                            "actual": observed,
                            "available": mask,
                            "decision_price": decision_price,
                            "shock_magnitude": shock_magnitude,
                            "shock_available": shock_available,
                        }
                    )
                )
    frame = pd.concat(frames, ignore_index=True)
    return frame


def aggregate_like_reconciliation(frame: pd.DataFrame) -> dict[str, Any]:
    """Reproduce `masked_node_metrics` from a recomputed frame.

    This exists to prove the recomputation is faithful: aggregating a frame for
    one (model, horizon, target) must reproduce the numbers D+1 already wrote.
    Any mismatch means the join is wrong, not that the reconciliation is wrong.
    """
    # Cast the mask: an empty or object-dtype column would otherwise be read as
    # a column selection rather than a boolean mask.
    usable = frame[frame["available"].astype(bool)]
    prediction = usable["prediction"].to_numpy(dtype=np.float64)
    observed = usable["actual"].to_numpy(dtype=np.float64)
    count = int(prediction.size)
    if count == 0:
        return {"count": 0, "mse": None, "mae": None, "pearson": None}
    error = prediction - observed
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    zero_mse = float(np.mean(observed**2))
    skill = None if zero_mse <= 0 else float(1.0 - mse / zero_mse)
    if count < 2:
        pearson = None
    else:
        centred_prediction = prediction - prediction.mean()
        centred_observed = observed - observed.mean()
        denominator = float(
            np.sqrt(float(centred_prediction @ centred_prediction))
            * np.sqrt(float(centred_observed @ centred_observed))
        )
        pearson = (
            None
            if denominator <= 0
            else float(float(centred_prediction @ centred_observed) / denominator)
        )
    return {
        "count": count,
        "mse": mse,
        "mae": mae,
        "zero_mse": zero_mse,
        "skill_vs_zero_mse": skill,
        "pearson": pearson,
    }


def development_auc(
    prediction: np.ndarray, actual: np.ndarray, *, big_threshold: float
) -> float | None:
    """ROC AUC for "did this observed shock develop into a large move".

    Positive class is `actual >= big_threshold`. Returns None when either class
    is absent, because AUC is undefined there and a silent 0.5 would read as
    chance-level skill rather than as missing evidence. Ties in the prediction
    receive mid-ranks, which is the standard AUC convention.
    """
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(predicted) & np.isfinite(observed)
    predicted = predicted[mask]
    observed = observed[mask]
    positive = observed >= float(big_threshold)
    positives = int(np.count_nonzero(positive))
    negatives = int(positive.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(predicted, kind="mergesort")
    ranks = np.empty(predicted.size, dtype=np.float64)
    ranks[order] = np.arange(1, predicted.size + 1, dtype=np.float64)
    # Mid-rank ties so that a model predicting one constant scores exactly 0.5.
    sorted_values = predicted[order]
    start = 0
    for stop in range(1, sorted_values.size + 1):
        if stop == sorted_values.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    positive_rank_sum = float(ranks[positive].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def magnitude_weighted_pearson(
    prediction: np.ndarray, actual: np.ndarray
) -> float | None:
    """Pearson weighted by the realized magnitude, so large moves carry weight.

    Unweighted Pearson is a rank-free but magnitude-blind statistic: a node that
    moved 0.1 percent counts as much as one that moved 10 percent. Weighting by
    |actual| is the continuous counterpart of the AUC threshold view.
    """
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(predicted) & np.isfinite(observed)
    predicted = predicted[mask]
    observed = observed[mask]
    if predicted.size < 2:
        return None
    weight = np.abs(observed)
    total = float(weight.sum())
    if total <= 0:
        return None
    weight = weight / total
    predicted_mean = float(weight @ predicted)
    observed_mean = float(weight @ observed)
    centred_prediction = predicted - predicted_mean
    centred_observed = observed - observed_mean
    covariance = float(weight @ (centred_prediction * centred_observed))
    prediction_variance = float(weight @ (centred_prediction**2))
    observed_variance = float(weight @ (centred_observed**2))
    if prediction_variance <= 0 or observed_variance <= 0:
        return None
    return float(covariance / np.sqrt(prediction_variance * observed_variance))


def load_ledger(path: Path) -> list[dict[str, Any]]:
    """Read the immutable prospective ledger as a list of records."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
