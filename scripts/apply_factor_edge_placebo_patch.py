"""Add a destination-permuted placebo to the selective factor-edge builder.

Pilot tree only (same rationale as factor_edge_patch.py).

Why this placebo: simply adding factor->stock edges also adds graph
connectivity, so any gain could come from "more edges" rather than "edges
routed to the stocks that actually co-move". This placebo holds the factor
sources, the edge count per factor, and the weight multiset exactly fixed and
permutes only which stock each edge lands on. A gain that survives against it
is attributable to correct routing; a gain that does not is connectivity.

The permutation is a deterministic function of the seed and the factor's source
index, so it is stable across steps, workers, and reruns (the edge cache is
hashed, so any nondeterminism would surface as a hash mismatch).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

BUILDER_SIGNATURE_OLD = '''    top_k: int = 0,
    min_abs_corr: float = 0.15,
    mode: str = "signed",
    scale: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Selective external-factor -> stock edges by trailing return sensitivity.'''
BUILDER_SIGNATURE_NEW = '''    top_k: int = 0,
    min_abs_corr: float = 0.15,
    mode: str = "signed",
    scale: float = 0.50,
    permute_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Selective external-factor -> stock edges by trailing return sensitivity.'''

APPEND_OLD = """            srcs.append(int(source))
            dsts.append(int(destination))
            weights.append(weight * float(scale))
    if not weights:
        return _empty_edge_arrays()
    return (
        np.asarray([srcs, dsts], dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )"""
APPEND_NEW = """            srcs.append(int(source))
            dsts.append(int(destination))
            weights.append(weight * float(scale))
    if not weights:
        return _empty_edge_arrays()
    if int(permute_seed) > 0:
        # Placebo: keep sources, per-factor edge counts, and the weight multiset
        # exactly as selected, and permute only the destination stocks. Each
        # factor gets one permutation table keyed by the seed and its source
        # index, so the mapping is identical across steps, workers, and reruns,
        # and distinct destinations stay distinct.
        tables = {
            source: np.random.default_rng(
                abs(int(permute_seed)) * 1_000_003 + int(source)
            ).permutation(int(stock_count))
            for source in sorted(set(srcs))
        }
        dsts = [
            int(tables[source][destination])
            for source, destination in zip(srcs, dsts)
        ]
    return (
        np.asarray([srcs, dsts], dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )"""

CALL_OLD = """            mode=factor_sensitivity_mode,
            scale=factor_sensitivity_scale,
        )
    )"""
CALL_NEW = """            mode=factor_sensitivity_mode,
            scale=factor_sensitivity_scale,
            permute_seed=factor_sensitivity_permute_seed,
        )
    )"""

BUILD_EDGE_SIG_OLD = '''    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(0, step - edge_window)'''
BUILD_EDGE_SIG_NEW = '''    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
    factor_sensitivity_permute_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(0, step - edge_window)'''

SNAPSHOT_SIG_OLD = '''    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
    seed: int | None = None,'''
SNAPSHOT_SIG_NEW = '''    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
    factor_sensitivity_permute_seed: int = 0,
    seed: int | None = None,'''

SNAPSHOT_CALL_OLD = """            factor_sensitivity_mode=factor_sensitivity_mode,
            factor_sensitivity_scale=factor_sensitivity_scale,
        )
    else:
        edge_index, edge_weight = cached_edges"""
SNAPSHOT_CALL_NEW = """            factor_sensitivity_mode=factor_sensitivity_mode,
            factor_sensitivity_scale=factor_sensitivity_scale,
            factor_sensitivity_permute_seed=factor_sensitivity_permute_seed,
        )
    else:
        edge_index, edge_weight = cached_edges"""

KWARGS_OLD = """        "factor_sensitivity_scale": float(
            getattr(args, "factor_sensitivity_scale", 0.50)
        ),
    }"""
KWARGS_NEW = """        "factor_sensitivity_scale": float(
            getattr(args, "factor_sensitivity_scale", 0.50)
        ),
        "factor_sensitivity_permute_seed": int(
            getattr(args, "factor_sensitivity_permute_seed", 0) or 0
        ),
    }"""

ARGS_OLD = '''    parser.add_argument("--factor-sensitivity-scale", type=float, default=0.50)'''
ARGS_NEW = '''    parser.add_argument("--factor-sensitivity-scale", type=float, default=0.50)
    parser.add_argument("--factor-sensitivity-permute-seed", type=int, default=0)'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(path: Path, pairs, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"already patched: {path}")
        return
    for name, old, new in pairs:
        if old not in text:
            raise SystemExit(f"{name} not found in {path}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path.name} -> {sha256(path)}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if (root / "scripts/run_post_impact_chain_session_cycle.py").is_file() and (
        root / "ops/prospective_live"
    ).is_dir():
        raise SystemExit("refusing to patch the prospective-chain tree")
    apply(
        root / "stock_v2/real_features.py",
        [
            ("builder signature", BUILDER_SIGNATURE_OLD, BUILDER_SIGNATURE_NEW),
            ("builder return", APPEND_OLD, APPEND_NEW),
            ("build_edge_tensor signature", BUILD_EDGE_SIG_OLD, BUILD_EDGE_SIG_NEW),
            ("build_edge_tensor call", CALL_OLD, CALL_NEW),
            ("make_real_snapshot signature", SNAPSHOT_SIG_OLD, SNAPSHOT_SIG_NEW),
            ("make_real_snapshot call", SNAPSHOT_CALL_OLD, SNAPSHOT_CALL_NEW),
        ],
        marker="permute_seed",
    )
    apply(
        root / "scripts/run_real_backtest.py",
        [
            ("graph_edge_kwargs", KWARGS_OLD, KWARGS_NEW),
            ("CLI argument", ARGS_OLD, ARGS_NEW),
        ],
        marker="factor_sensitivity_permute_seed",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
