from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class CausalGroupedQuantileResult:
    thresholds: np.ndarray
    events: np.ndarray
    available: np.ndarray
    history_count: np.ndarray


def causal_grouped_upper_tail(
    values: Sequence[float] | np.ndarray,
    valid: Sequence[bool] | np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    *,
    quantile: float = 0.80,
    window: int = 60,
    minimum_history: int = 20,
) -> CausalGroupedQuantileResult:
    """Flag upper-tail observations using only earlier values in each group."""

    values_array = np.asarray(values, dtype=np.float64)
    valid_array = np.asarray(valid, dtype=bool)
    groups_array = np.asarray(groups, dtype=object)
    if (
        values_array.ndim != 1
        or valid_array.shape != values_array.shape
        or groups_array.shape != values_array.shape
    ):
        raise ValueError("causal grouped quantiles require aligned one-dimensional inputs")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("causal grouped quantile must be in (0, 1)")
    if not 1 <= int(minimum_history) <= int(window):
        raise ValueError("causal grouped quantiles require 1 <= minimum_history <= window")

    thresholds = np.full(len(values_array), np.nan, dtype=np.float64)
    events = np.zeros(len(values_array), dtype=bool)
    available = np.zeros(len(values_array), dtype=bool)
    history_count = np.zeros(len(values_array), dtype=np.int32)
    histories: dict[Hashable, deque[float]] = defaultdict(
        lambda: deque(maxlen=int(window))
    )
    for index, (value, is_valid, raw_group) in enumerate(
        zip(values_array, valid_array, groups_array)
    ):
        group = raw_group.item() if isinstance(raw_group, np.generic) else raw_group
        try:
            history = histories[group]
        except TypeError as exc:
            raise ValueError("causal quantile groups must be hashable") from exc
        history_count[index] = len(history)
        current_valid = bool(is_valid) and np.isfinite(value)
        if current_valid and len(history) >= int(minimum_history):
            threshold = float(
                np.quantile(
                    np.asarray(history, dtype=np.float64),
                    float(quantile),
                    method="higher",
                )
            )
            thresholds[index] = threshold
            available[index] = True
            events[index] = bool(value >= threshold)
        if current_valid:
            history.append(float(value))
    return CausalGroupedQuantileResult(
        thresholds=thresholds,
        events=events,
        available=available,
        history_count=history_count,
    )
