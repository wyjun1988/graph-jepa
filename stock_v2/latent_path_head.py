from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn


HORIZON_WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}


class LatentTrajectoryPathHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        horizons: Sequence[int],
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.LayerNorm(2 * int(latent_dim)),
                    nn.Linear(2 * int(latent_dim), int(hidden_dim)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), 1),
                )
                for horizon in self.horizons
            }
        )

    def forward(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        features = torch.cat((context, predicted - context), dim=-1)
        return self.heads[str(int(horizon))](features).squeeze(-1)


@dataclass(frozen=True)
class LoadedLatentPathHead:
    model: LatentTrajectoryPathHead
    checkpoint_path: Path
    checkpoint_sha256: str
    parent_model_sha256: str
    train_data_manifest_sha256: str
    train_edge_manifest_sha256: str
    horizons: tuple[int, ...]
    latent_blend_weight: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha(checkpoint: Mapping, key: str) -> str:
    manifest = checkpoint.get(key)
    if not isinstance(manifest, Mapping) or not manifest.get("sha256"):
        raise ValueError(f"parent checkpoint is missing {key}.sha256")
    return str(manifest["sha256"])


def load_latent_path_head(
    checkpoint_path: str | Path,
    parent_model_path: str | Path,
    parent_checkpoint: Mapping,
    device: str | torch.device,
) -> LoadedLatentPathHead:
    path = Path(checkpoint_path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "state_dict",
        "parent_model_sha256",
        "horizons",
        "latent_dim",
        "hidden_dim",
        "dropout",
        "latent_blend_weight",
        "train_data_manifest_sha256",
        "train_edge_manifest_sha256",
        "live_orders_allowed",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"latent path head is missing fields: {missing}")
    if artifact["live_orders_allowed"] is not False:
        raise ValueError("latent path head must remain read-only shadow")

    parent_sha = sha256_file(parent_model_path)
    if str(artifact["parent_model_sha256"]) != parent_sha:
        raise ValueError("latent path head parent checkpoint SHA-256 mismatch")
    data_sha = _manifest_sha(parent_checkpoint, "train_data_manifest")
    edge_sha = _manifest_sha(parent_checkpoint, "train_edge_manifest")
    if str(artifact["train_data_manifest_sha256"]) != data_sha:
        raise ValueError("latent path head data manifest SHA-256 mismatch")
    if str(artifact["train_edge_manifest_sha256"]) != edge_sha:
        raise ValueError("latent path head edge manifest SHA-256 mismatch")

    horizons = tuple(int(value) for value in artifact["horizons"])
    if not horizons or len(set(horizons)) != len(horizons) or any(value < 1 for value in horizons):
        raise ValueError("latent path head horizons must be unique positive integers")
    configured_value = parent_checkpoint.get("args", {}).get("rollout_offsets", [])
    if isinstance(configured_value, str):
        configured_values = [value.strip() for value in configured_value.split(",") if value.strip()]
    else:
        configured_values = list(configured_value)
    configured = tuple(sorted(int(value) for value in configured_values))
    if tuple(sorted(horizons)) != configured:
        raise ValueError("latent path head horizons do not match parent rollout offsets")
    latent_dim = int(artifact["latent_dim"])
    if latent_dim != int(parent_checkpoint.get("args", {}).get("hidden_dim", -1)):
        raise ValueError("latent path head dimension does not match parent hidden dimension")
    blend_weight = float(artifact["latent_blend_weight"])
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("latent path head blend weight must be in [0, 1]")

    model = LatentTrajectoryPathHead(
        latent_dim=latent_dim,
        horizons=horizons,
        hidden_dim=int(artifact["hidden_dim"]),
        dropout=float(artifact["dropout"]),
    ).to(torch.device(device))
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return LoadedLatentPathHead(
        model=model,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        parent_model_sha256=parent_sha,
        train_data_manifest_sha256=data_sha,
        train_edge_manifest_sha256=edge_sha,
        horizons=horizons,
        latent_blend_weight=blend_weight,
    )


def cross_sectional_zscore(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("cross-sectional values must be one-dimensional")
    selected = np.isfinite(array)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError("cross-sectional valid mask must match values")
        selected &= mask
    result = np.full(array.shape, np.nan, dtype=np.float64)
    if not selected.any():
        return result
    centered = array[selected] - array[selected].mean()
    scale = np.sqrt(np.mean(np.square(centered)))
    result[selected] = centered / max(float(scale), 1e-6)
    return result


def blend_latent_path_scores(
    base_paths: np.ndarray,
    latent_paths: np.ndarray,
    horizons: Sequence[int],
    latent_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base = np.asarray(base_paths, dtype=np.float64)
    latent = np.asarray(latent_paths, dtype=np.float64)
    horizon_tuple = tuple(int(value) for value in horizons)
    if base.ndim != 2 or latent.shape != base.shape:
        raise ValueError("base and latent path scores must be aligned matrices")
    if base.shape[1] != len(horizon_tuple):
        raise ValueError("path score columns must match horizons")
    if not 0.0 <= float(latent_weight) <= 1.0:
        raise ValueError("latent path blend weight must be in [0, 1]")

    base_z = np.full(base.shape, np.nan, dtype=np.float64)
    latent_z = np.full(latent.shape, np.nan, dtype=np.float64)
    blended = np.full(base.shape, np.nan, dtype=np.float64)
    for position in range(base.shape[1]):
        valid = np.isfinite(base[:, position]) & np.isfinite(latent[:, position])
        base_z[:, position] = cross_sectional_zscore(base[:, position], valid)
        latent_z[:, position] = cross_sectional_zscore(latent[:, position], valid)
        blended[:, position] = (
            (1.0 - float(latent_weight)) * base_z[:, position]
            + float(latent_weight) * latent_z[:, position]
        )
    weights = np.asarray([HORIZON_WEIGHTS.get(value, 1.0) for value in horizon_tuple])
    finite = np.isfinite(blended)
    denominator = np.where(finite, weights[None, :], 0.0).sum(axis=1)
    scores = np.divide(
        np.where(finite, blended * weights[None, :], 0.0).sum(axis=1),
        denominator,
        out=np.full(base.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )
    return scores, {
        "base_entry_path_zscores": base_z,
        "latent_path_head_zscores": latent_z,
        "blended_entry_path_scores": blended,
    }
