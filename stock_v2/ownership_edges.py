"""Point-in-time ownership edges between listed KRX-500 companies.

Every statistical edge the campaign tried (return correlation, news-theme
similarity, lead-lag, partial correlation) measured neutral-to-harmful. The
working hypothesis is that they are *derived from* observables the 149 node
features already carry, so message passing re-delivers known information while
smearing idiosyncratic signal across neighbours.

Ownership is exogenous: it cannot be recovered from price history, and it
carries a concrete causal channel (a holding company's value literally contains
its subsidiaries', and Korean holding discounts make that repricing slow).

Weights are the disclosed stake itself (20% -> 0.20), so a control relationship
passes a stronger message than a passive 5% position without any hand-tuned
tiering. Edges are symmetric: value flows up from investee to holder, and
parent distress flows back down.

PIT: a relation only exists from the day its filing was received, and then
persists (forward-fill) until superseded by a newer disclosure for that pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_ownership_panel(
    path: str | Path,
    node_tickers: list[str],
    dates: pd.DatetimeIndex,
    min_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pairs, ratios) where pairs is (2, P) node indices and ratios is
    (T, P) the stake known on each date -- 0 before the first disclosure."""
    index = {str(t).zfill(6): i for i, t in enumerate(node_tickers)}
    date_strs = np.array([str(d)[:10] for d in dates])
    n_dates = len(date_strs)

    latest: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        h = index.get(str(row.get("holder") or "").zfill(6))
        t = index.get(str(row.get("target") or "").zfill(6))
        if h is None or t is None or h == t:
            continue
        try:
            ratio = float(str(row.get("ratio")).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ratio) or ratio <= 0.0:
            continue
        latest.setdefault((h, t), []).append((str(row.get("date"))[:10], min(ratio, 100.0) / 100.0))

    pairs: list[tuple[int, int]] = []
    columns: list[np.ndarray] = []
    for (h, t), events in latest.items():
        series = np.zeros(n_dates, dtype=np.float32)
        for day, ratio in sorted(events):
            pos = int(np.searchsorted(date_strs, day, side="left"))
            if pos >= n_dates:
                continue
            series[pos:] = ratio          # forward-fill until a newer filing
        if float(series.max()) < min_ratio:
            continue
        pairs.append((h, t))
        columns.append(series)

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((n_dates, 0), dtype=np.float32)
    return (
        np.asarray(pairs, dtype=np.int64).T,
        np.stack(columns, axis=1).astype(np.float32),
    )


def build_ownership_edge_tensor(
    features,
    step: int,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = getattr(features, "ownership_pairs", None)
    ratios = getattr(features, "ownership_ratios", None)
    if pairs is None or ratios is None or scale <= 0.0 or pairs.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    if step < 0 or step >= ratios.shape[0]:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    active = ratios[step] > 0.0
    if not active.any():
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    holder = pairs[0][active]
    target = pairs[1][active]
    weight = (ratios[step][active] * float(scale)).astype(np.float32)
    edge_index = np.concatenate(
        [np.stack([target, holder]), np.stack([holder, target])], axis=1
    ).astype(np.int64)
    return edge_index, np.concatenate([weight, weight])


def attach_ownership_edges(features, path: str | Path, min_ratio: float = 0.01):
    """Attach the PIT ownership panel to a built FeaturePanel, in place."""
    node_tickers = list(features.node_tickers or features.tickers)
    pairs, ratios = load_ownership_panel(path, node_tickers, features.dates, min_ratio)
    features.ownership_pairs = pairs
    features.ownership_ratios = ratios
    covered = len(set(pairs.flatten().tolist())) if pairs.shape[1] else 0
    live = int((ratios[-1] > 0).sum()) if ratios.shape[1] else 0
    print(
        f"ownership edges: pairs={pairs.shape[1]} nodes={covered} "
        f"active_at_end={live} median_stake="
        f"{float(np.median(ratios[-1][ratios[-1] > 0])) if live else 0.0:.3f}",
        flush=True,
    )
    return features
