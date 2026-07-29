from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def parse_date(record: dict[str, Any]) -> datetime | None:
    article = record.get("article") if isinstance(record.get("article"), dict) else {}
    for value in [record.get("published"), article.get("published"), article.get("updated"), record.get("ts")]:
        if not value:
            continue
        text = str(value)
        try:
            if "," in text:
                return parsedate_to_datetime(text).replace(tzinfo=None)
        except Exception:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pass
    return None


def event_type(record: dict[str, Any]) -> str:
    event = record.get("event") if isinstance(record.get("event"), dict) else {}
    raw = str(event.get("event_type") or "unknown").strip().lower()
    return re.sub(r"[^a-z0-9_가-힣-]+", "_", raw)[:32] or "unknown"


def score(record: dict[str, Any]) -> float:
    try:
        return float(record.get("score_contribution") or 0.0)
    except Exception:
        return 0.0


def sign_bucket(value: float, threshold: float) -> str:
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "neutral"


def take_round_robin(
    buckets: dict[tuple[str, str], list[dict[str, Any]]],
    target: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    keys = sorted(buckets)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < target and keys:
        progressed = False
        for key in list(keys):
            rows = buckets.get(key) or []
            while rows:
                row = rows.pop()
                rid = str(row.get("id", ""))
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                selected.append(row)
                progressed = True
                break
            if not rows:
                keys.remove(key)
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced Qwen calibration sample from news-event JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument("--positive-frac", type=float, default=0.35)
    parser.add_argument("--negative-frac", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--exclude", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    exclude_ids: set[str] = set()
    for raw_path in args.exclude:
        path = Path(raw_path)
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            if row.get("id"):
                exclude_ids.add(str(row["id"]))

    rows = []
    for record in iter_jsonl(Path(args.input)):
        rid = str(record.get("id", ""))
        ticker = str(record.get("ticker", "")).replace("A", "").strip()
        if not rid or rid in exclude_ids or not re.fullmatch(r"\d{6}", ticker):
            continue
        date = parse_date(record)
        year = "unknown" if date is None else str(date.year)
        value = score(record)
        row = dict(record)
        row["calibration_sampling"] = {
            "year": year,
            "sign_bucket": sign_bucket(value, args.score_threshold),
            "event_type": event_type(record),
            "heuristic_score": value,
        }
        rows.append(row)

    target_pos = int(round(args.sample_size * args.positive_frac))
    target_neg = int(round(args.sample_size * args.negative_frac))
    target_neu = max(0, args.sample_size - target_pos - target_neg)
    targets = {"positive": target_pos, "negative": target_neg, "neutral": target_neu}

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    counts: dict[str, int] = {}
    for sign, target in targets.items():
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            meta = row["calibration_sampling"]
            if meta["sign_bucket"] != sign:
                continue
            buckets[(str(meta["year"]), str(meta["event_type"]))].append(row)
        part = take_round_robin(buckets, target, rng)
        counts[sign] = len(part)
        for row in part:
            rid = str(row.get("id", ""))
            if rid not in selected_ids:
                selected_ids.add(rid)
                selected.append(row)

    if len(selected) < args.sample_size:
        leftovers = [row for row in rows if str(row.get("id", "")) not in selected_ids]
        rng.shuffle(leftovers)
        for row in leftovers[: args.sample_size - len(selected)]:
            selected.append(row)
            selected_ids.add(str(row.get("id", "")))

    rng.shuffle(selected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected[: args.sample_size]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": args.output,
        "sample_size": len(selected[: args.sample_size]),
        "requested": args.sample_size,
        "targets": targets,
        "selected_by_initial_bucket": counts,
        "final_counts": defaultdict(int),
        "years": defaultdict(int),
    }
    for row in selected[: args.sample_size]:
        meta = row["calibration_sampling"]
        summary["final_counts"][meta["sign_bucket"]] += 1
        summary["years"][meta["year"]] += 1
    summary["final_counts"] = dict(sorted(summary["final_counts"].items()))
    summary["years"] = dict(sorted(summary["years"].items()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
