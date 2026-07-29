from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from stock_v2.online_residual_adapter import (
    OnlineResidualAdapter,
    OnlineResidualConfig,
)


RESIDUAL_STATE_MODES = ("bias_only", "dynamic", "node_permuted")


@dataclass(frozen=True)
class PostImpactResidualConfig:
    alpha: float = 0.05
    ridge: float = 0.01
    min_updates: int = 10
    gain_clip: float = 1.0
    bias_clip: float = 0.02
    correction_clip: float = 0.05
    mode: str = "dynamic"
    permutation_seed: int = 17

    def validate(self) -> None:
        if self.mode not in RESIDUAL_STATE_MODES:
            raise ValueError(f"unsupported post-impact residual mode: {self.mode}")
        self.adapter_config().validate()

    def adapter_config(self) -> OnlineResidualConfig:
        return OnlineResidualConfig(
            alpha=float(self.alpha),
            ridge=float(self.ridge),
            min_updates=int(self.min_updates),
            gain_clip=float(self.gain_clip),
            bias_clip=float(self.bias_clip),
            correction_clip=float(self.correction_clip),
        )


@dataclass(frozen=True)
class PendingEndpointForecast:
    origin_timestamp_ns: int
    prediction: np.ndarray
    origin_price: np.ndarray
    source_valid: np.ndarray


class CausalPostImpactResidualState:
    """Correct fixed-horizon endpoint forecasts using only matured price errors."""

    def __init__(
        self,
        node_count: int,
        horizon_minutes: Sequence[int],
        config: PostImpactResidualConfig,
        *,
        adapter_states: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        config.validate()
        self.node_count = int(node_count)
        self.horizon_minutes = tuple(int(value) for value in horizon_minutes)
        if self.node_count < 1:
            raise ValueError("post-impact residual node count must be positive")
        if (
            not self.horizon_minutes
            or any(value <= 0 for value in self.horizon_minutes)
            or len(set(self.horizon_minutes)) != len(self.horizon_minutes)
        ):
            raise ValueError("post-impact residual horizons must be unique and positive")
        self.config = config
        supplied = dict(adapter_states or {})
        if supplied and set(supplied) != set(self.horizon_minutes):
            raise ValueError("post-impact residual adapter states do not match horizons")
        self.adapters = {
            horizon: OnlineResidualAdapter(
                horizon=1,
                feature_count=1,
                config=config.adapter_config(),
                state=supplied.get(horizon),
            )
            for horizon in self.horizon_minutes
        }
        generator = np.random.default_rng(int(config.permutation_seed))
        self.node_permutation = generator.permutation(self.node_count)
        self.pending: dict[int, dict[int, PendingEndpointForecast]] = {
            horizon: {} for horizon in self.horizon_minutes
        }
        self.residual_history: dict[int, dict[int, np.ndarray]] = {
            horizon: {} for horizon in self.horizon_minutes
        }
        self.latest_residual: dict[int, np.ndarray | None] = {
            horizon: None for horizon in self.horizon_minutes
        }
        self.session: str | None = None
        self.last_session: str | None = None
        self.last_timestamp_ns: int | None = None
        self.matured_forecasts = 0
        self.matured_cells = 0
        self.dynamic_corrections = 0
        self.dropped_pending = 0

    @staticmethod
    def _horizon_ns(horizon_minutes: int) -> int:
        return int(horizon_minutes) * 60 * 1_000_000_000

    def start_session(self, session: str) -> None:
        name = str(session)
        if not name:
            raise ValueError("post-impact residual session must not be empty")
        if self.session is not None:
            raise ValueError("finish the active residual session before starting another")
        if self.last_session is not None and name <= self.last_session:
            raise ValueError("post-impact residual sessions must be strictly increasing")
        self.session = name
        self.last_timestamp_ns = None
        for horizon in self.horizon_minutes:
            self.pending[horizon].clear()
            self.residual_history[horizon].clear()
            self.latest_residual[horizon] = None

    def _validate_observation(
        self,
        timestamp_ns: int,
        decision_price: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        if self.session is None:
            raise ValueError("start a residual session before observing prices")
        timestamp = int(timestamp_ns)
        if self.last_timestamp_ns is not None and timestamp <= self.last_timestamp_ns:
            raise ValueError("post-impact residual timestamps must be strictly increasing")
        price = np.asarray(decision_price, dtype=np.float32)
        if price.shape != (self.node_count,):
            raise ValueError("post-impact residual decision prices do not match nodes")
        return timestamp, price

    def _mature(self, timestamp_ns: int, decision_price: np.ndarray) -> None:
        for horizon in self.horizon_minutes:
            pending = self.pending[horizon].pop(int(timestamp_ns), None)
            if pending is None:
                continue
            valid = (
                pending.source_valid
                & np.isfinite(decision_price)
                & (decision_price > 0.0)
            )
            realized = np.full(self.node_count, np.nan, dtype=np.float32)
            realized[valid] = (
                decision_price[valid] / pending.origin_price[valid] - 1.0
            )
            residual = realized - pending.prediction
            residual[~valid] = np.nan
            predictor_origin = (
                pending.origin_timestamp_ns - self._horizon_ns(horizon)
            )
            predictor = self.residual_history[horizon].get(predictor_origin)
            self.adapters[horizon].observe_matured_error(
                residual[:, None],
                predictor[:, None] if predictor is not None else None,
            )
            self.residual_history[horizon][pending.origin_timestamp_ns] = residual
            self.latest_residual[horizon] = residual
            minimum_origin = pending.origin_timestamp_ns - 2 * self._horizon_ns(horizon)
            self.residual_history[horizon] = {
                origin: values
                for origin, values in self.residual_history[horizon].items()
                if origin >= minimum_origin
            }
            self.matured_forecasts += 1
            self.matured_cells += int(valid.sum())

    def _correction(self, horizon: int) -> tuple[np.ndarray, bool]:
        previous = self.latest_residual[horizon]
        if previous is None:
            previous_input = np.full((self.node_count, 1), np.nan, dtype=np.float32)
        else:
            selected = previous
            if self.config.mode == "node_permuted":
                selected = selected[self.node_permutation]
            previous_input = selected[:, None]
        _bias, dynamic, used = self.adapters[horizon].correction(
            previous_input,
            dynamic=self.config.mode != "bias_only",
        )
        return dynamic[:, 0], bool(used)

    def step(
        self,
        timestamp_ns: int,
        decision_price: np.ndarray,
        endpoint_prediction: np.ndarray,
    ) -> np.ndarray:
        timestamp, price = self._validate_observation(timestamp_ns, decision_price)
        prediction = np.asarray(endpoint_prediction, dtype=np.float32)
        expected = (self.node_count, len(self.horizon_minutes))
        if prediction.shape != expected:
            raise ValueError("post-impact endpoint predictions do not match state shape")
        self._mature(timestamp, price)
        corrected = prediction.copy()
        for horizon_index, horizon in enumerate(self.horizon_minutes):
            correction, used = self._correction(horizon)
            valid = np.isfinite(prediction[:, horizon_index])
            corrected[valid, horizon_index] += correction[valid]
            target_timestamp = timestamp + self._horizon_ns(horizon)
            if target_timestamp in self.pending[horizon]:
                raise ValueError("duplicate post-impact residual target timestamp")
            source_valid = valid & np.isfinite(price) & (price > 0.0)
            self.pending[horizon][target_timestamp] = PendingEndpointForecast(
                origin_timestamp_ns=timestamp,
                prediction=corrected[:, horizon_index].copy(),
                origin_price=price.copy(),
                source_valid=source_valid,
            )
            self.dynamic_corrections += int(used)
        self.last_timestamp_ns = timestamp
        return corrected

    def finish_session(
        self,
        terminal_timestamp_ns: int | None = None,
        terminal_price: np.ndarray | None = None,
    ) -> None:
        if self.session is None:
            raise ValueError("no active post-impact residual session")
        if (terminal_timestamp_ns is None) != (terminal_price is None):
            raise ValueError("terminal timestamp and price must be supplied together")
        if terminal_timestamp_ns is not None and terminal_price is not None:
            timestamp, price = self._validate_observation(
                int(terminal_timestamp_ns), terminal_price
            )
            self._mature(timestamp, price)
            self.last_timestamp_ns = timestamp
        self.dropped_pending += sum(len(values) for values in self.pending.values())
        for horizon in self.horizon_minutes:
            self.pending[horizon].clear()
            self.residual_history[horizon].clear()
            self.latest_residual[horizon] = None
        self.last_session = self.session
        self.session = None
        self.last_timestamp_ns = None

    def adapter_state_dict(self) -> dict[int, dict[str, np.ndarray]]:
        if self.session is not None:
            raise ValueError("persist post-impact residual state only at a session boundary")
        return {
            horizon: adapter.state_dict()
            for horizon, adapter in self.adapters.items()
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "mode": self.config.mode,
            "node_count": self.node_count,
            "horizon_minutes": list(self.horizon_minutes),
            "matured_forecasts": int(self.matured_forecasts),
            "matured_cells": int(self.matured_cells),
            "dynamic_corrections": int(self.dynamic_corrections),
            "dropped_pending": int(self.dropped_pending),
            "adapters": {
                str(horizon): {
                    "bias_updates": int(adapter.bias_updates.max(initial=0)),
                    "gain_updates": int(adapter.gain_updates.max(initial=0)),
                    "bias": float(adapter.coefficients()[0][0]),
                    "gain": float(adapter.coefficients()[1][0]),
                }
                for horizon, adapter in self.adapters.items()
            },
        }
