from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _normalize_ticker(value: object) -> str:
    text = str(value).strip().replace("A", "")
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def load_industry_codes(paths: Iterable[str | Path]) -> dict[str, str]:
    """Load OpenDART company profile JSONL into a ticker-to-industry mapping."""

    codes: dict[str, str] = {}
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"industry profile path not found: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid industry profile JSON at {path}:{line_number}") from exc
            ticker = _normalize_ticker(row.get("ticker", ""))
            code = str(row.get("industry_code", "")).strip()
            if not ticker or not code.isdigit():
                continue
            prior = codes.get(ticker)
            if prior is not None and prior != code:
                raise ValueError(f"conflicting industry codes for ticker={ticker}")
            codes[ticker] = code
    return codes


def build_industry_edge_arrays(
    tickers: Sequence[str],
    industry_codes: Mapping[str, str],
    prefix_length: int = 2,
    scale: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Create bidirectional same-industry edges among stock nodes.

    The complete intra-industry clique is small for the current 100-stock
    universe and avoids arbitrary ordering when several companies share one
    industry. External factor nodes are deliberately excluded by accepting only
    the stock ticker sequence.
    """

    if prefix_length < 1:
        raise ValueError("industry prefix_length must be >= 1")
    if scale < 0.0:
        raise ValueError("industry edge scale must be >= 0")
    groups: dict[str, list[int]] = {}
    matched = 0
    for index, ticker in enumerate(tickers):
        code = str(industry_codes.get(_normalize_ticker(ticker), ""))
        if not code.isdigit() or len(code) < prefix_length:
            continue
        matched += 1
        groups.setdefault(code[:prefix_length], []).append(index)

    src: list[int] = []
    dst: list[int] = []
    for indices in groups.values():
        if len(indices) < 2 or scale == 0.0:
            continue
        for source in indices:
            for target in indices:
                if source != target:
                    src.append(source)
                    dst.append(target)
    edge_index = (
        np.asarray([src, dst], dtype=np.int64)
        if src
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_weight = np.full(len(src), float(scale), dtype=np.float32)
    stats = {
        "profile_tickers": len(industry_codes),
        "matched_tickers": matched,
        "industry_groups": sum(len(indices) >= 2 for indices in groups.values()),
        "edges": len(src),
    }
    return edge_index, edge_weight, stats
