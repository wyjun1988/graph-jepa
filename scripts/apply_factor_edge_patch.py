"""Apply the selective factor-to-stock edge builder to a stock-v2 source tree.

This patch is applied ONLY to the disposable A4000 pilot tree. It must not be
applied to the M1 Max authoritative tree: `stock_v2/real_features.py` sits on
the daily prospective chain's import path
(`build_stale_jepa_rollout_cache.py` -> `run_real_backtest.py` /
`stock_v2/ops/signals.py` -> `stock_v2/real_features.py`), and the 06:05 KST
cycle rebuilds the prospective cache from the working tree.

Hypothesis under test: external macro factors move only a subset of stocks.
Every previously rejected design connected factors to stocks indiscriminately
(repeated features, full external nodes, uniform one-way broadcast). A causal,
per-factor top-k edge selected by trailing return sensitivity has never been
tested.

Idempotent: re-running detects the existing marker and exits without changes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

MARKER = "def build_factor_sensitivity_edge_tensor("

BUILDER = '''

def build_factor_sensitivity_edge_tensor(
    features: "FeaturePanel",
    history: np.ndarray,
    top_k: int = 0,
    min_abs_corr: float = 0.15,
    mode: str = "signed",
    scale: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Selective external-factor -> stock edges by trailing return sensitivity.

    Unlike `build_policy_rate_broadcast_edge_tensor`, which sends every policy
    node to every stock with one uniform weight, this connects each external
    factor only to the `top_k` stocks whose trailing returns actually co-move
    with it. `history` is the same window the sibling builders receive, so the
    causality convention is unchanged.

    The matrix is factors x stocks and therefore rectangular, so it cannot use
    `_edge_arrays_from_matrix` (that helper zeroes a diagonal, which is
    meaningless here and would silently drop real factor/stock pairs).
    """

    if top_k <= 0 or scale <= 0.0 or mode == "none":
        return _empty_edge_arrays()
    if not features.node_tickers or features.tradable_count <= 0:
        return _empty_edge_arrays()
    if history.ndim != 2 or history.shape[0] < 3:
        return _empty_edge_arrays()
    stock_count = int(features.tradable_count)
    external_indices = [
        index
        for index, node_id in enumerate(features.node_tickers)
        if str(node_id).startswith("EXT:")
    ]
    if not external_indices or history.shape[1] <= stock_count:
        return _empty_edge_arrays()
    external_indices = [index for index in external_indices if index < history.shape[1]]
    if not external_indices:
        return _empty_edge_arrays()

    values = np.nan_to_num(history.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    centered = values - values.mean(axis=0, keepdims=True)
    deviation = centered.std(axis=0, keepdims=True)
    normalized = np.divide(
        centered, deviation, out=np.zeros_like(centered), where=deviation > 1e-8
    )
    factors = normalized[:, external_indices]
    stocks = normalized[:, :stock_count]
    matrix = factors.T @ stocks / max(1, normalized.shape[0] - 1)
    matrix = np.clip(matrix, -1.0, 1.0)
    matrix = (
        np.rint(matrix / EDGE_WEIGHT_QUANTIZATION) * EDGE_WEIGHT_QUANTIZATION
    ).astype(np.float32)

    srcs: list[int] = []
    dsts: list[int] = []
    weights: list[float] = []
    for row, source in enumerate(external_indices):
        line = matrix[row]
        if mode == "positive":
            candidates = np.flatnonzero(line >= float(min_abs_corr))
            scores = line[candidates]
        elif mode == "negative":
            candidates = np.flatnonzero(line <= -float(min_abs_corr))
            scores = np.abs(line[candidates])
        else:
            candidates = np.flatnonzero(np.abs(line) >= float(min_abs_corr))
            scores = np.abs(line[candidates])
        if candidates.size == 0:
            continue
        order = np.lexsort((candidates, scores))
        ranked = np.sort(candidates[order[-int(top_k) :]])
        for destination in ranked:
            weight = float(line[int(destination)])
            if mode == "abs":
                weight = abs(weight)
            if abs(weight) <= 0.0:
                continue
            srcs.append(int(source))
            dsts.append(int(destination))
            weights.append(weight * float(scale))
    if not weights:
        return _empty_edge_arrays()
    return (
        np.asarray([srcs, dsts], dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )
'''

BUILD_EDGE_SIGNATURE_OLD = """    policy_rate_edge_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(0, step - edge_window)"""
BUILD_EDGE_SIGNATURE_NEW = """    policy_rate_edge_scale: float = 0.0,
    factor_sensitivity_top_k: int = 0,
    factor_sensitivity_min_abs_corr: float = 0.15,
    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(0, step - edge_window)"""

CALL_OLD = """    edge_parts.append(
        build_policy_rate_broadcast_edge_tensor(
            features,
            scale=policy_rate_edge_scale,
        )
    )"""
CALL_NEW = """    edge_parts.append(
        build_policy_rate_broadcast_edge_tensor(
            features,
            scale=policy_rate_edge_scale,
        )
    )
    edge_parts.append(
        build_factor_sensitivity_edge_tensor(
            features,
            history,
            top_k=factor_sensitivity_top_k,
            min_abs_corr=factor_sensitivity_min_abs_corr,
            mode=factor_sensitivity_mode,
            scale=factor_sensitivity_scale,
        )
    )"""

SNAPSHOT_SIGNATURE_OLD = """    policy_rate_edge_scale: float = 0.0,
    seed: int | None = None,"""
SNAPSHOT_SIGNATURE_NEW = """    policy_rate_edge_scale: float = 0.0,
    factor_sensitivity_top_k: int = 0,
    factor_sensitivity_min_abs_corr: float = 0.15,
    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
    seed: int | None = None,"""

SNAPSHOT_CALL_OLD = """            policy_rate_edge_scale=policy_rate_edge_scale,
        )
    else:
        edge_index, edge_weight = cached_edges"""
SNAPSHOT_CALL_NEW = """            policy_rate_edge_scale=policy_rate_edge_scale,
            factor_sensitivity_top_k=factor_sensitivity_top_k,
            factor_sensitivity_min_abs_corr=factor_sensitivity_min_abs_corr,
            factor_sensitivity_mode=factor_sensitivity_mode,
            factor_sensitivity_scale=factor_sensitivity_scale,
        )
    else:
        edge_index, edge_weight = cached_edges"""


KWARGS_OLD = """        "policy_rate_edge_scale": float(getattr(args, "policy_rate_edge_scale", 0.0)),
    }"""
KWARGS_NEW = """        "policy_rate_edge_scale": float(getattr(args, "policy_rate_edge_scale", 0.0)),
        "factor_sensitivity_top_k": int(
            getattr(args, "factor_sensitivity_top_k", 0) or 0
        ),
        "factor_sensitivity_min_abs_corr": float(
            getattr(args, "factor_sensitivity_min_abs_corr", 0.15)
        ),
        "factor_sensitivity_mode": str(
            getattr(args, "factor_sensitivity_mode", "signed")
        ),
        "factor_sensitivity_scale": float(
            getattr(args, "factor_sensitivity_scale", 0.50)
        ),
    }"""

ARGS_OLD = '''    parser.add_argument("--policy-rate-edge-scale", type=float, default=0.0)'''
ARGS_NEW = '''    parser.add_argument("--policy-rate-edge-scale", type=float, default=0.0)
    parser.add_argument("--factor-sensitivity-top-k", type=int, default=0)
    parser.add_argument("--factor-sensitivity-min-abs-corr", type=float, default=0.15)
    parser.add_argument(
        "--factor-sensitivity-mode",
        default="signed",
        choices=["signed", "positive", "negative", "abs", "none"],
    )
    parser.add_argument("--factor-sensitivity-scale", type=float, default=0.50)'''

IMPORT_OLD = """    build_edge_tensor,"""
IMPORT_NEW = """    build_edge_tensor,
    build_factor_sensitivity_edge_tensor,"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_backtest(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "factor_sensitivity_top_k" in text:
        print(f"already patched: {path}")
        return False
    for name, fragment in (
        ("graph_edge_kwargs", KWARGS_OLD),
        ("policy-rate CLI argument", ARGS_OLD),
    ):
        if fragment not in text:
            raise SystemExit(f"{name} not found in {path}")
    text = text.replace(KWARGS_OLD, KWARGS_NEW, 1)
    text = text.replace(ARGS_OLD, ARGS_NEW, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched run_real_backtest.py -> {sha256(path)}")
    return True


def patch_real_features(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {path}")
        return False
    if BUILD_EDGE_SIGNATURE_OLD not in text:
        raise SystemExit(f"build_edge_tensor signature not found in {path}")
    if CALL_OLD not in text:
        raise SystemExit(f"policy-rate call site not found in {path}")
    for name, fragment in (
        ("make_real_snapshot signature", SNAPSHOT_SIGNATURE_OLD),
        ("make_real_snapshot call site", SNAPSHOT_CALL_OLD),
    ):
        if fragment not in text:
            raise SystemExit(f"{name} not found in {path}")
    anchor = "def build_edge_tensor("
    text = text.replace(anchor, BUILDER.strip() + "\n\n\n" + anchor, 1)
    text = text.replace(BUILD_EDGE_SIGNATURE_OLD, BUILD_EDGE_SIGNATURE_NEW, 1)
    text = text.replace(CALL_OLD, CALL_NEW, 1)
    text = text.replace(SNAPSHOT_SIGNATURE_OLD, SNAPSHOT_SIGNATURE_NEW, 1)
    text = text.replace(SNAPSHOT_CALL_OLD, SNAPSHOT_CALL_NEW, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched real_features.py -> {sha256(path)}")
    return True


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    features = root / "stock_v2/real_features.py"
    backtest = root / "scripts/run_real_backtest.py"
    for path in (features, backtest):
        if not path.is_file():
            raise SystemExit(f"not a stock-v2 tree: {root}")
    guard = root / "scripts/run_post_impact_chain_session_cycle.py"
    if guard.is_file() and (root / "ops/prospective_live").is_dir():
        raise SystemExit(
            "refusing to patch a tree that runs the prospective chain; this patch "
            "is for the disposable A4000 pilot tree only"
        )
    patch_real_features(features)
    patch_backtest(backtest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
