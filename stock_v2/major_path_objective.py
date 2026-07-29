from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from stock_v2.market_transition_head import MARKET_FAMILY_TARGETS
from stock_v2.systemic_head import correlation_rank_loss, focal_binary_loss


MAJOR_PATH_LOSS_WEIGHTS = {
    "components": 0.15,
    "families": 0.20,
    "family_rank": 0.10,
    "events": 0.10,
    "trajectory": 0.05,
    "major_rank": 0.15,
    "major_focal": 0.15,
    "peak_horizon": 0.10,
}


@dataclass(frozen=True)
class MajorPathContract:
    event_quantile: float
    event_threshold: float
    fit_event_rate: float
    logit_scale: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def family_threshold_matrix(contracts, horizons: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(
                    contracts[int(horizon)].calibration.family_event_threshold[
                        family
                    ]
                )
                for family in MARKET_FAMILY_TARGETS
            ]
            for horizon in horizons
        ],
        dtype=np.float64,
    )


def target_path_salience(targets, contracts, horizons: Sequence[int]) -> np.ndarray:
    family = np.expm1(np.asarray(targets["family_log"], dtype=np.float64))
    thresholds = family_threshold_matrix(contracts, horizons)
    ratios = family / np.maximum(thresholds[None, :, :], 1e-8)
    return np.nanmax(ratios, axis=(1, 2))


def fit_major_path_contract(
    fit_targets,
    contracts,
    horizons: Sequence[int],
    *,
    event_quantile: float = 0.90,
) -> MajorPathContract:
    if not 0.80 <= float(event_quantile) < 1.0:
        raise ValueError("major path event quantile must be in [0.80, 1.0)")
    salience = target_path_salience(fit_targets, contracts, horizons)
    salience = salience[np.isfinite(salience)]
    if salience.size < 20:
        raise ValueError("too few fit trajectories for major path calibration")
    threshold = float(np.quantile(salience, float(event_quantile)))
    center = float(np.median(salience))
    scale = threshold - center
    if scale <= 1e-8:
        scale = float(np.std(salience))
    if scale <= 1e-8:
        scale = 1.0
    return MajorPathContract(
        event_quantile=float(event_quantile),
        event_threshold=threshold,
        fit_event_rate=float(np.mean(salience >= threshold)),
        logit_scale=float(scale),
    )


def add_major_path_targets(
    targets,
    contracts,
    horizons: Sequence[int],
    contract: MajorPathContract,
) -> dict[str, object]:
    output = dict(targets)
    family = np.expm1(np.asarray(targets["family_log"], dtype=np.float64))
    thresholds = family_threshold_matrix(contracts, horizons)
    horizon_salience = np.nanmax(
        family / np.maximum(thresholds[None, :, :], 1e-8), axis=2
    )
    path_salience = np.nanmax(horizon_salience, axis=1)
    output["horizon_salience"] = horizon_salience.astype(np.float32)
    output["path_salience"] = path_salience.astype(np.float32)
    output["major_label"] = (
        path_salience >= float(contract.event_threshold)
    ).astype(np.float32)
    output["peak_horizon_index"] = np.argmax(horizon_salience, axis=1).astype(
        np.int64
    )
    return output


def major_target_batch(targets, positions, device):
    index = np.asarray(positions, dtype=np.int64)
    return {
        "horizon_salience": torch.as_tensor(
            targets["horizon_salience"][index], device=device
        ),
        "path_salience": torch.as_tensor(
            targets["path_salience"][index], device=device
        ),
        "major_label": torch.as_tensor(targets["major_label"][index], device=device),
        "peak_horizon_index": torch.as_tensor(
            targets["peak_horizon_index"][index], dtype=torch.long, device=device
        ),
    }


def predicted_horizon_salience(
    family_log_prediction: torch.Tensor,
    contracts,
    horizons: Sequence[int],
) -> torch.Tensor:
    if family_log_prediction.ndim != 3:
        raise ValueError("family predictions must be batch-by-horizon-by-family")
    thresholds = torch.as_tensor(
        family_threshold_matrix(contracts, horizons),
        dtype=family_log_prediction.dtype,
        device=family_log_prediction.device,
    )
    family = torch.expm1(torch.clamp(family_log_prediction, -5.0, 5.0)).clamp_min(0.0)
    return (family / thresholds.clamp_min(1e-8)[None, :, :]).amax(dim=2)


def major_path_loss_terms(
    family_log_prediction: torch.Tensor,
    target: Mapping[str, torch.Tensor],
    contracts,
    horizons: Sequence[int],
    path_contract: MajorPathContract,
) -> dict[str, torch.Tensor]:
    horizon_salience = predicted_horizon_salience(
        family_log_prediction, contracts, horizons
    )
    path_salience = horizon_salience.amax(dim=1)
    major_rank = correlation_rank_loss(path_salience, target["path_salience"])
    major_logits = (
        path_salience - float(path_contract.event_threshold)
    ) / max(float(path_contract.logit_scale), 1e-8)
    major_focal = focal_binary_loss(major_logits, target["major_label"])
    major = target["major_label"] > 0.5
    peak_horizon = (
        F.cross_entropy(
            horizon_salience[major], target["peak_horizon_index"][major]
        )
        if major.any()
        else family_log_prediction.new_tensor(0.0)
    )
    return {
        "major_rank": major_rank,
        "major_focal": major_focal,
        "peak_horizon": peak_horizon,
    }
