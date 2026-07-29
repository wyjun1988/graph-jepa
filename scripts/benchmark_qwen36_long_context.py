#!/usr/bin/env python3
"""Measure deterministic long-context retrieval through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


NEEDLES = {
    73: ("alpha", "ALPHA-7D3F91"),
    211: ("bravo", "BRAVO-42C8AE"),
    389: ("charlie", "CHARLIE-19B7D0"),
    577: ("delta", "DELTA-88E21C"),
    743: ("echo", "ECHO-05AF64"),
}


def build_prompt(records: int) -> tuple[str, dict[str, str]]:
    if records <= max(NEEDLES):
        raise ValueError(f"records must be greater than {max(NEEDLES)}")
    expected = {name: value for _, (name, value) in NEEDLES.items()}
    rows = []
    for index in range(records):
        row = (
            f"Ledger row {index:04d}: division={index % 17:02d}; "
            f"batch={index % 29:02d}; status=ordinary; "
            "instruction=retain this row for audit comparison."
        )
        if index in NEEDLES:
            name, value = NEEDLES[index]
            row += f" SECRET_{name.upper()}={value}."
        rows.append(row)
    prompt = (
        "Read the complete ledger. Return only one compact JSON object whose keys are "
        "alpha, bravo, charlie, delta, and echo and whose values are the corresponding "
        "SECRET codes. Do not add markdown or explanation.\n\n" + "\n".join(rows)
    )
    return prompt, expected


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    return json.loads(text[start : end + 1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", type=int, default=900)
    parser.add_argument("--output")
    args = parser.parse_args()

    prompt, expected = build_prompt(args.records)
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Follow the retrieval instruction exactly. Do not reason aloud.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    elapsed = time.perf_counter() - started
    text = payload["choices"][0]["message"]["content"]
    error = None
    try:
        actual = extract_json(text)
        passed = actual == expected
        if not passed:
            error = f"expected={expected!r}, actual={actual!r}"
    except Exception as exc:
        actual = None
        passed = False
        error = repr(exc)

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "records": args.records,
        "passed": passed,
        "elapsed_seconds": round(elapsed, 3),
        "usage": payload.get("usage", {}),
        "expected": expected,
        "actual": actual,
        "error": error,
        "raw_response": text,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        Path(args.output).write_text(f"{encoded}\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
