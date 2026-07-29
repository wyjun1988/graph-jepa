from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class OnlineResidualConfig:
    alpha: float = 0.05
    ridge: float = 0.01
    min_updates: int = 10
    gain_clip: float = 1.0
    bias_clip: float = 0.5
    correction_clip: float = 1.0

    def validate(self) -> None:
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("online residual alpha must be in (0, 1]")
        if not np.isfinite(self.ridge) or float(self.ridge) < 0.0:
            raise ValueError("online residual ridge must be finite and non-negative")
        if int(self.min_updates) < 1:
            raise ValueError("online residual min_updates must be positive")
        if min(
            float(self.gain_clip),
            float(self.bias_clip),
            float(self.correction_clip),
        ) <= 0.0:
            raise ValueError("online residual clips must be positive")


class OnlineResidualAdapter:
    """Causal feature-wise AR correction updated only from matured errors."""

    STATE_KEYS = (
        "bias_mean",
        "input_mean",
        "input_square_mean",
        "input_target_mean",
        "bias_updates",
        "gain_updates",
    )

    def __init__(
        self,
        horizon: int,
        feature_count: int,
        config: OnlineResidualConfig,
        state: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        config.validate()
        if int(horizon) < 1 or int(feature_count) < 1:
            raise ValueError("online residual horizon and feature count must be positive")
        self.horizon = int(horizon)
        self.feature_count = int(feature_count)
        self.config = config
        self.bias_mean = np.zeros(self.feature_count, dtype=np.float64)
        self.input_mean = np.zeros(self.feature_count, dtype=np.float64)
        self.input_square_mean = np.zeros(self.feature_count, dtype=np.float64)
        self.input_target_mean = np.zeros(self.feature_count, dtype=np.float64)
        self.bias_updates = np.zeros(self.feature_count, dtype=np.int64)
        self.gain_updates = np.zeros(self.feature_count, dtype=np.int64)
        if state is not None:
            self.load_state(state)

    @staticmethod
    def _daily_mean(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = valid.sum(axis=0)
        total = np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)
        mean = np.divide(
            total,
            counts,
            out=np.zeros_like(total, dtype=np.float64),
            where=counts > 0,
        )
        return mean, counts > 0

    def _ewm_update(
        self,
        destination: np.ndarray,
        counts: np.ndarray,
        observation: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        first = valid & (counts == 0)
        continuing = valid & ~first
        destination[first] = observation[first]
        alpha = float(self.config.alpha)
        destination[continuing] += alpha * (
            observation[continuing] - destination[continuing]
        )
        counts[valid] += 1

    def observe_matured_error(
        self,
        target_error: np.ndarray,
        previous_error: np.ndarray | None,
    ) -> None:
        target = np.asarray(target_error, dtype=np.float64)
        if target.ndim != 2 or target.shape[1] != self.feature_count:
            raise ValueError("matured target error dimensions do not match the adapter")
        target_valid = np.isfinite(target)
        target_mean, target_features = self._daily_mean(target, target_valid)
        self._ewm_update(
            self.bias_mean,
            self.bias_updates,
            target_mean,
            target_features,
        )
        self.bias_mean[:] = np.clip(
            self.bias_mean,
            -float(self.config.bias_clip),
            float(self.config.bias_clip),
        )
        if previous_error is None:
            return
        previous = np.asarray(previous_error, dtype=np.float64)
        if previous.shape != target.shape:
            raise ValueError("matured residual predictor and target dimensions differ")
        pair_valid = target_valid & np.isfinite(previous)
        input_mean, pair_features = self._daily_mean(previous, pair_valid)
        input_square_mean, _ = self._daily_mean(
            np.square(previous), pair_valid
        )
        input_target_mean, _ = self._daily_mean(previous * target, pair_valid)
        before = self.gain_updates.copy()
        self._ewm_update(
            self.input_mean,
            self.gain_updates,
            input_mean,
            pair_features,
        )
        auxiliary_counts = before.copy()
        self._ewm_update(
            self.input_square_mean,
            auxiliary_counts,
            input_square_mean,
            pair_features,
        )
        auxiliary_counts = before.copy()
        self._ewm_update(
            self.input_target_mean,
            auxiliary_counts,
            input_target_mean,
            pair_features,
        )

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        bias = np.clip(
            self.bias_mean,
            -float(self.config.bias_clip),
            float(self.config.bias_clip),
        )
        numerator = self.input_target_mean - bias * self.input_mean
        denominator = self.input_square_mean + float(self.config.ridge)
        gain = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        mature = self.gain_updates >= int(self.config.min_updates)
        gain = np.where(mature, gain, 0.0)
        gain = np.clip(
            gain,
            -float(self.config.gain_clip),
            float(self.config.gain_clip),
        )
        return bias.astype(np.float32), gain.astype(np.float32)

    def correction(
        self,
        previous_error: np.ndarray | None,
        *,
        dynamic: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if previous_error is None:
            previous = np.full((1, self.feature_count), np.nan, dtype=np.float32)
        else:
            previous = np.asarray(previous_error, dtype=np.float32)
            if previous.ndim != 2 or previous.shape[1] != self.feature_count:
                raise ValueError("online residual correction dimensions do not match")
        bias, gain = self.coefficients()
        bias_correction = np.broadcast_to(bias[None, :], previous.shape).copy()
        correction = bias_correction.copy()
        dynamic_features = self.gain_updates >= int(self.config.min_updates)
        if dynamic:
            valid = np.isfinite(previous) & dynamic_features[None, :]
            contribution = previous * gain[None, :]
            correction[valid] += contribution[valid]
        clip = float(self.config.correction_clip)
        return (
            np.clip(bias_correction, -clip, clip),
            np.clip(correction, -clip, clip),
            bool(dynamic and dynamic_features.any() and np.isfinite(previous).any()),
        )

    def advance(
        self,
        errors: np.ndarray,
        origin_index: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        values = np.asarray(errors, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != self.feature_count:
            raise ValueError("online residual errors must be [time, node, feature]")
        index = int(origin_index)
        if index < 0 or index >= len(values):
            raise ValueError("online residual origin index exceeds the error history")
        matured = index - self.horizon
        previous = (
            values[matured]
            if matured >= 0
            else np.full(values.shape[1:], np.nan, dtype=np.float32)
        )
        if matured >= 0:
            predictor = (
                values[matured - self.horizon]
                if matured - self.horizon >= 0
                else None
            )
            self.observe_matured_error(values[matured], predictor)
        return self.correction(previous, dynamic=matured >= 0)

    def flush(self, errors: np.ndarray) -> None:
        values = np.asarray(errors, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != self.feature_count:
            raise ValueError("online residual flush dimensions do not match")
        for matured in range(max(0, len(values) - self.horizon), len(values)):
            predictor = (
                values[matured - self.horizon]
                if matured - self.horizon >= 0
                else None
            )
            self.observe_matured_error(values[matured], predictor)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(getattr(self, key)).copy() for key in self.STATE_KEYS}

    def load_state(self, state: Mapping[str, np.ndarray]) -> None:
        if set(state) != set(self.STATE_KEYS):
            raise ValueError("online residual state keys do not match the contract")
        for key in self.STATE_KEYS:
            expected_dtype = np.int64 if key.endswith("updates") else np.float64
            value = np.asarray(state[key], dtype=expected_dtype)
            if value.shape != (self.feature_count,) or not np.isfinite(value).all():
                raise ValueError(f"online residual state {key} is invalid")
            if key.endswith("updates") and np.any(value < 0):
                raise ValueError(f"online residual state {key} has negative counts")
            setattr(self, key, value.copy())
