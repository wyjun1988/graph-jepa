from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


KEYS = ["split", "date", "target_date", "step", "horizon"]
EVENT_COLUMNS = [
    "systemic_event",
    "price_transition",
    "activity_transition",
    "node_state_transition",
    "topology_transition",
]
BROADNESS_COLUMNS = [
    "return_concentration",
    "return_breadth",
    "volume_participation_z1",
    "value_participation_z1",
    "state_change_participation",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def _finite_summary(values: pd.Series) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "median": float("nan"), "q75": float("nan")}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _event_comparison(frame: pd.DataFrame, event: str) -> dict[str, float | int]:
    old = _as_bool(frame[f"old:{event}"])
    new = _as_bool(frame[f"new:{event}"])
    intersection = int((old & new).sum())
    union = int((old | new).sum())
    return {
        "rows": int(len(frame)),
        "old_count": int(old.sum()),
        "new_count": int(new.sum()),
        "old_rate": float(old.mean()) if len(frame) else float("nan"),
        "new_rate": float(new.mean()) if len(frame) else float("nan"),
        "intersection": intersection,
        "old_only": int((old & ~new).sum()),
        "new_only": int((~old & new).sum()),
        "jaccard": float(intersection / union) if union else 1.0,
    }


def _broadness(frame: pd.DataFrame, prefix: str, selector: pd.Series) -> dict[str, Any]:
    selected = frame.loc[selector]
    return {
        column: _finite_summary(selected[f"{prefix}:{column}"])
        for column in BROADNESS_COLUMNS
    }


def _examples(frame: pd.DataFrame, selector: pd.Series, prefix: str, limit: int) -> list[dict[str, Any]]:
    selected = frame.loc[selector].copy()
    selected = selected.sort_values(f"{prefix}:systemic_energy", ascending=False).head(limit)
    columns = [
        *KEYS,
        f"{prefix}:systemic_energy",
        f"{prefix}:family:price_co_movement",
        f"{prefix}:family:market_activity",
        f"{prefix}:family:node_state",
        f"{prefix}:family:topology",
        f"{prefix}:return_concentration",
        f"{prefix}:return_breadth",
        f"{prefix}:volume_participation_z1",
        f"{prefix}:value_participation_z1",
        f"{prefix}:state_change_participation",
    ]
    return selected[columns].replace({np.nan: None}).to_dict(orient="records")


def compare(old_path: Path, new_path: Path, *, examples: int) -> dict[str, Any]:
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    if old.duplicated(KEYS).any() or new.duplicated(KEYS).any():
        raise ValueError("target rows must be unique by split/date/target/horizon")
    merged = old[KEYS].merge(new[KEYS], on=KEYS, how="outer", indicator=True)
    if not bool((merged["_merge"] == "both").all()):
        raise ValueError("old and new target rows do not align exactly")

    old = old.rename(columns={column: f"old:{column}" for column in old.columns if column not in KEYS})
    new = new.rename(columns={column: f"new:{column}" for column in new.columns if column not in KEYS})
    frame = old.merge(new, on=KEYS, how="inner", validate="one_to_one")
    test = frame.loc[frame["split"] == "test"].copy()
    old_event = _as_bool(test["old:systemic_event"])
    new_event = _as_bool(test["new:systemic_event"])

    by_horizon = {}
    for horizon, selected in test.groupby("horizon", sort=True):
        by_horizon[str(int(horizon))] = {
            event: _event_comparison(selected, event) for event in EVENT_COLUMNS
        }

    return {
        "status": "complete",
        "role": "market_transition_target_version_comparison",
        "old_path": str(old_path),
        "new_path": str(new_path),
        "aligned_rows": int(len(frame)),
        "test_rows": int(len(test)),
        "events": {event: _event_comparison(test, event) for event in EVENT_COLUMNS},
        "events_by_horizon": by_horizon,
        "broadness": {
            "old_events_old_values": _broadness(test, "old", old_event),
            "new_events_new_values": _broadness(test, "new", new_event),
            "old_only_old_values": _broadness(test, "old", old_event & ~new_event),
            "new_only_new_values": _broadness(test, "new", ~old_event & new_event),
        },
        "old_only_examples": _examples(test, old_event & ~new_event, "old", examples),
        "new_only_examples": _examples(test, ~old_event & new_event, "new", examples),
        "live_orders_allowed": False,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    event = summary["events"]["systemic_event"]
    lines = [
        "# Market transition target version comparison",
        "",
        f"- Aligned test rows: {summary['test_rows']}",
        f"- Old events: {event['old_count']} ({event['old_rate']:.3f})",
        f"- New events: {event['new_count']} ({event['new_rate']:.3f})",
        f"- Common / old-only / new-only: {event['intersection']} / {event['old_only']} / {event['new_only']}",
        f"- Jaccard: {event['jaccard']:.3f}",
        "- Live orders allowed: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two aligned market transition target audits.")
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compare(Path(args.old), Path(args.new), examples=int(args.examples))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary["events"]["systemic_event"], ensure_ascii=False))


if __name__ == "__main__":
    main()
