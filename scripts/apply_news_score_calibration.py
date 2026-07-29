from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def median_ratio(pairs: list[tuple[float, float]], fallback: float) -> float:
    ratios = []
    for heuristic, qwen in pairs:
        if abs(heuristic) < 1e-12:
            continue
        ratio = qwen / heuristic
        if ratio > 0 and ratio == ratio and abs(ratio) < 20:
            ratios.append(ratio)
    if not ratios:
        return fallback
    return max(0.05, min(5.0, float(statistics.median(ratios))))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def fit_calibration(samples: list[dict[str, Any]]) -> dict[str, Any]:
    pos_pairs: list[tuple[float, float]] = []
    neg_pairs: list[tuple[float, float]] = []
    zero_qwen: list[float] = []
    all_pairs: list[tuple[float, float]] = []
    used = 0
    errors = 0
    for record in samples:
        if not record.get("llm_used"):
            errors += 1
            continue
        h = safe_float((record.get("rescore") or {}).get("previous_score_contribution"))
        q = safe_float(record.get("score_contribution"))
        all_pairs.append((h, q))
        used += 1
        if h > 1e-12:
            pos_pairs.append((h, q))
        elif h < -1e-12:
            neg_pairs.append((h, q))
        else:
            zero_qwen.append(q)
    pos_scale = median_ratio(pos_pairs, 1.0)
    neg_scale = median_ratio(neg_pairs, 1.0)
    abs_h = [abs(h) for h, _ in all_pairs if abs(h) > 1e-12]
    abs_q = [abs(q) for _, q in all_pairs]
    return {
        "sample_rows": len(samples),
        "llm_used_rows": used,
        "llm_error_rows": errors,
        "positive_pairs": len(pos_pairs),
        "negative_pairs": len(neg_pairs),
        "zero_heuristic_pairs": len(zero_qwen),
        "positive_scale_median": pos_scale,
        "negative_scale_median": neg_scale,
        "heuristic_abs_mean_nonzero": mean(abs_h),
        "qwen_abs_mean": mean(abs_q),
        "qwen_zero_mean": mean(zero_qwen),
        "qwen_zero_abs_mean": mean([abs(v) for v in zero_qwen]),
    }


def ticker_score_from_payload(record: dict[str, Any], ticker: str) -> float:
    event = record.get("event") if isinstance(record.get("event"), dict) else {}
    score = 0.0
    for delta in event.get("node_deltas", []) or []:
        if not isinstance(delta, dict):
            continue
        node = str(delta.get("node", "")).replace("A", "")
        if node != ticker:
            continue
        if str(delta.get("field", "news_score")) != "news_score":
            continue
        score += safe_float(delta.get("delta")) * safe_float(delta.get("confidence", event.get("confidence", 0.0)))
    return score


def scale_event_payload(record: dict[str, Any], calibrated_score: float) -> dict[str, Any]:
    ticker = str(record.get("ticker", "")).replace("A", "")
    event = record.get("event")
    if not isinstance(event, dict):
        return record
    old_score = ticker_score_from_payload(record, ticker)
    if abs(old_score) < 1e-12:
        return record
    factor = calibrated_score / old_score
    updated_event = dict(event)
    deltas = []
    for delta in event.get("node_deltas", []) or []:
        if not isinstance(delta, dict):
            deltas.append(delta)
            continue
        new_delta = dict(delta)
        node = str(new_delta.get("node", "")).replace("A", "")
        if node == ticker and str(new_delta.get("field", "news_score")) == "news_score":
            new_delta["delta"] = safe_float(new_delta.get("delta")) * factor
        deltas.append(new_delta)
    updated_event["node_deltas"] = deltas
    edge_deltas = []
    for delta in event.get("edge_deltas", []) or []:
        if not isinstance(delta, dict):
            edge_deltas.append(delta)
            continue
        new_delta = dict(delta)
        dst = str(new_delta.get("dst", "")).replace("A", "")
        if dst == ticker and "delta_weight" in new_delta:
            new_delta["delta_weight"] = safe_float(new_delta.get("delta_weight")) * abs(factor)
        edge_deltas.append(new_delta)
    updated_event["edge_deltas"] = edge_deltas
    updated = dict(record)
    updated["event"] = updated_event
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Qwen sample calibration to a full news-event JSONL file.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--qwen-sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-scale", type=float, default=3.0)
    parser.add_argument("--min-scale", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    sample_path = Path(args.qwen_sample)
    output_path = Path(args.output)
    report_path = Path(args.report)
    samples = list(iter_jsonl(sample_path))
    by_id = {str(record.get("id")): record for record in samples if record.get("id") and record.get("llm_used")}
    cal = fit_calibration(samples)
    pos_scale = max(args.min_scale, min(args.max_scale, float(cal["positive_scale_median"])))
    neg_scale = max(args.min_scale, min(args.max_scale, float(cal["negative_scale_median"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "qwen_exact": 0, "scaled_positive": 0, "scaled_negative": 0, "zero_kept": 0}
    score_before = []
    score_after = []
    with output_path.open("w", encoding="utf-8") as output:
        for record in iter_jsonl(base_path):
            counts["rows"] += 1
            rid = str(record.get("id", ""))
            old = safe_float(record.get("score_contribution"))
            score_before.append(old)
            if rid in by_id:
                updated = dict(by_id[rid])
                updated["source"] = "qwen_calibration_exact_sample"
                counts["qwen_exact"] += 1
            else:
                if old > 1e-12:
                    new_score = old * pos_scale
                    counts["scaled_positive"] += 1
                elif old < -1e-12:
                    new_score = old * neg_scale
                    counts["scaled_negative"] += 1
                else:
                    new_score = 0.0
                    counts["zero_kept"] += 1
                updated = scale_event_payload(record, new_score)
                updated["score_contribution"] = new_score
                updated["source"] = "heuristic_qwen_calibrated"
                updated["calibration"] = {
                    "method": "signed_median_ratio_from_qwen_sample",
                    "positive_scale": pos_scale,
                    "negative_scale": neg_scale,
                    "qwen_sample": str(sample_path),
                    "calibrated_at": datetime.now().isoformat(timespec="seconds"),
                }
            score_after.append(safe_float(updated.get("score_contribution")))
            output.write(json.dumps(updated, ensure_ascii=False) + "\n")

    report = {
        "base": str(base_path),
        "qwen_sample": str(sample_path),
        "output": str(output_path),
        "calibration": cal,
        "applied_scales": {"positive": pos_scale, "negative": neg_scale},
        "counts": counts,
        "score_abs_mean_before": mean([abs(v) for v in score_before]),
        "score_abs_mean_after": mean([abs(v) for v in score_after]),
        "score_nonzero_before": sum(abs(v) > 1e-12 for v in score_before),
        "score_nonzero_after": sum(abs(v) > 1e-12 for v in score_after),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
