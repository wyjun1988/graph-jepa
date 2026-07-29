from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InnovationFilterConfig:
    name: str
    alpha: float
    node_mix: float
    clip: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("innovation filter name must not be empty")
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("innovation alpha must be in (0, 1]")
        if not np.isfinite(self.node_mix) or not 0.0 <= self.node_mix <= 1.0:
            raise ValueError("innovation node_mix must be between 0 and 1")
        if not np.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("innovation clip must be positive")


class CausalInnovationFilter:
    """Bounded EWMA memory for already-matured forecast residuals."""

    def __init__(
        self,
        node_count: int,
        feature_count: int,
        config: InnovationFilterConfig,
    ) -> None:
        if node_count < 1 or feature_count < 1:
            raise ValueError("innovation filter dimensions must be positive")
        self.config = config
        self.common_memory = np.zeros(feature_count, dtype=np.float32)
        self.node_memory = np.zeros(
            (node_count, feature_count),
            dtype=np.float32,
        )
        self.update_calls = 0
        self.updated_cells = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.node_memory.shape

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
    ) -> int:
        prediction = np.asarray(prediction, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if prediction.shape != self.shape or target.shape != self.shape:
            raise ValueError("innovation prediction and target shapes must match")
        if valid.shape != self.shape:
            raise ValueError("innovation valid mask shape must match")
        valid = valid & np.isfinite(prediction) & np.isfinite(target)
        count = int(valid.sum())
        if count == 0:
            return 0

        residual = np.clip(
            target - prediction,
            -self.config.clip,
            self.config.clip,
        ).astype(np.float32, copy=False)
        alpha = np.float32(self.config.alpha)
        if self.config.node_mix < 1.0:
            valid_features = np.flatnonzero(valid.any(axis=0))
            for feature_index in valid_features:
                feature_valid = valid[:, feature_index]
                innovation = np.float32(
                    np.median(residual[feature_valid, feature_index])
                )
                self.common_memory[feature_index] += alpha * (
                    innovation - self.common_memory[feature_index]
                )
        if self.config.node_mix > 0.0:
            self.node_memory[valid] += alpha * (
                residual[valid] - self.node_memory[valid]
            )

        self.update_calls += 1
        self.updated_cells += count
        return count

    def correction(self) -> np.ndarray:
        common_weight = np.float32(1.0 - self.config.node_mix)
        node_weight = np.float32(self.config.node_mix)
        correction = (
            common_weight * self.common_memory[None, :]
            + node_weight * self.node_memory
        )
        return np.clip(
            correction,
            -self.config.clip,
            self.config.clip,
        ).astype(np.float32, copy=False)

    def apply(
        self,
        prediction: np.ndarray,
        eligible: np.ndarray,
    ) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=np.float32)
        eligible = np.asarray(eligible, dtype=bool)
        if prediction.shape != self.shape or eligible.shape != self.shape:
            raise ValueError("innovation apply inputs must match filter shape")
        result = prediction.copy()
        valid = eligible & np.isfinite(result)
        correction = self.correction()
        result[valid] += correction[valid]
        return result

    def diagnostics(self) -> dict[str, float | int]:
        correction = self.correction().astype(np.float64)
        return {
            "update_calls": int(self.update_calls),
            "updated_cells": int(self.updated_cells),
            "correction_rms": float(np.sqrt(np.mean(correction**2))),
            "correction_max_abs": float(np.max(np.abs(correction))),
            "common_memory_rms": float(
                np.sqrt(np.mean(self.common_memory.astype(np.float64) ** 2))
            ),
            "node_memory_rms": float(
                np.sqrt(np.mean(self.node_memory.astype(np.float64) ** 2))
            ),
        }


@dataclass
class PendingForecast:
    prediction: np.ndarray
    source_available: np.ndarray


class CausalHorizonInnovationState:
    """Enforce mature-before-forecast ordering for one forecast horizon."""

    def __init__(
        self,
        node_count: int,
        feature_count: int,
        eligible_features: np.ndarray,
        config: InnovationFilterConfig,
    ) -> None:
        eligible_features = np.asarray(eligible_features, dtype=bool)
        if eligible_features.shape != (feature_count,):
            raise ValueError("eligible_features must contain one value per feature")
        self.filter = CausalInnovationFilter(
            node_count,
            feature_count,
            config,
        )
        self.eligible_features = eligible_features
        self.pending: dict[int, PendingForecast] = {}
        self.current_step: int | None = None
        self.last_forecast_step: int | None = None

    def mature(
        self,
        step: int,
        target: np.ndarray,
        target_available: np.ndarray,
    ) -> int:
        step = int(step)
        if self.current_step is not None and step <= self.current_step:
            raise ValueError("innovation maturity steps must be strictly increasing")
        self.current_step = step
        pending = self.pending.pop(step, None)
        if pending is None:
            return 0
        target_available = np.asarray(target_available, dtype=bool)
        eligible = np.broadcast_to(
            self.eligible_features[None, :],
            pending.prediction.shape,
        )
        valid = pending.source_available & target_available & eligible
        return self.filter.update(pending.prediction, target, valid)

    def correct_and_enqueue(
        self,
        context_step: int,
        horizon: int,
        prediction: np.ndarray,
        source_available: np.ndarray,
    ) -> np.ndarray:
        context_step = int(context_step)
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("innovation horizon must be positive")
        if self.current_step != context_step:
            raise ValueError(
                "mature must be called for the context step before forecasting"
            )
        if self.last_forecast_step is not None and context_step <= self.last_forecast_step:
            raise ValueError("innovation forecast steps must be strictly increasing")
        self.last_forecast_step = context_step
        source_available = np.asarray(source_available, dtype=bool)
        eligible = np.broadcast_to(
            self.eligible_features[None, :],
            self.filter.shape,
        )
        corrected = self.filter.apply(
            prediction,
            source_available & eligible,
        )
        target_step = context_step + horizon
        if target_step in self.pending:
            raise ValueError("duplicate pending innovation forecast target")
        self.pending[target_step] = PendingForecast(
            prediction=corrected.copy(),
            source_available=source_available.copy(),
        )
        return corrected
