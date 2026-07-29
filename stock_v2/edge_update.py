from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class EdgeDelta:
    src: str
    dst: str
    edge_type: str
    delta_weight: float
    confidence: float = 1.0
    half_life_days: float = 5.0


class RollingCorrelationEdgeUpdater:
    """Build directed stock-stock edges from recent return correlation."""

    def __init__(self, window: int = 20, top_k: int = 5, min_abs_corr: float = 0.25):
        self.window = window
        self.top_k = top_k
        self.min_abs_corr = min_abs_corr

    def build_edges(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return edge_index [2, E] and edge_weight [E].

        returns shape is [time, num_nodes]. The newest `window` rows are used.
        Edges are directed both ways so the message passing layer can aggregate
        neighbors with plain index_add.
        """

        if returns.ndim != 2:
            raise ValueError("returns must have shape [time, num_nodes]")
        if returns.shape[0] < 3:
            raise ValueError("at least 3 time rows are required")

        recent = returns[-self.window :]
        corr = np.nan_to_num(np.corrcoef(recent, rowvar=False), nan=0.0)
        np.fill_diagonal(corr, 0.0)

        srcs: List[int] = []
        dsts: List[int] = []
        weights: List[float] = []

        num_nodes = corr.shape[0]
        for src in range(num_nodes):
            candidates = np.argsort(-np.abs(corr[src]))[: self.top_k]
            for dst in candidates:
                weight = float(corr[src, dst])
                if abs(weight) < self.min_abs_corr:
                    continue
                srcs.append(src)
                dsts.append(int(dst))
                weights.append(weight)

        if not weights:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        return (
            np.asarray([srcs, dsts], dtype=np.int64),
            np.asarray(weights, dtype=np.float32),
        )


class EdgeStateStore:
    """Simple typed edge store with exponential decay and event deltas."""

    def __init__(self, daily_decay: float = 0.97):
        self.daily_decay = daily_decay
        self._weights: Dict[Tuple[str, str, str], float] = {}

    def decay(self, days: float = 1.0) -> None:
        factor = self.daily_decay ** days
        for key in list(self._weights):
            value = self._weights[key] * factor
            if abs(value) < 1e-4:
                del self._weights[key]
            else:
                self._weights[key] = value

    def apply_deltas(self, deltas: Iterable[EdgeDelta]) -> None:
        for delta in deltas:
            key = (delta.src, delta.dst, delta.edge_type)
            current = self._weights.get(key, 0.0)
            self._weights[key] = current + delta.delta_weight * delta.confidence

    def items(self) -> List[Tuple[str, str, str, float]]:
        return [
            (src, dst, edge_type, weight)
            for (src, dst, edge_type), weight in sorted(self._weights.items())
        ]
