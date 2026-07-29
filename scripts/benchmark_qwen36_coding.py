#!/usr/bin/env python3
"""Run a small executable coding benchmark against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


TASKS = [
    {
        "name": "merge_intervals",
        "prompt": """Implement this exact Python function:

def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:

Requirements: do not mutate the input; sort the result; merge intervals exactly when
next_start <= current_end (so (1, 2) and (3, 4) stay separate); raise ValueError when
an interval has start > end; empty input returns [].
Output only complete Python source, with no markdown or explanation.""",
        "tests": """
assert merge_intervals([]) == []
source = [(5, 7), (1, 2), (2, 4), (9, 9)]
assert merge_intervals(source) == [(1, 4), (5, 7), (9, 9)]
assert source == [(5, 7), (1, 2), (2, 4), (9, 9)]
assert merge_intervals([(-3, -1), (-2, 2), (8, 10)]) == [(-3, 2), (8, 10)]
try:
    merge_intervals([(3, 2)])
except ValueError:
    pass
else:
    raise AssertionError('invalid interval was accepted')
""",
    },
    {
        "name": "topological_sort",
        "prompt": """Implement this exact Python function:

def topological_sort(graph: dict[str, set[str]]) -> list[str]:

The mapping is node -> outgoing successors. Include nodes that occur only as successors.
Return a deterministic topological ordering, choosing the lexicographically smallest
available node each step. Raise ValueError on a cycle. Do not mutate graph.
Output only complete Python source, with no markdown or explanation.""",
        "tests": """
g = {'build': {'test', 'package'}, 'test': {'package'}, 'orphan': set()}
assert topological_sort(g) == ['build', 'orphan', 'test', 'package']
assert g == {'build': {'test', 'package'}, 'test': {'package'}, 'orphan': set()}
assert topological_sort({'b': {'c'}, 'a': {'c'}}) == ['a', 'b', 'c']
assert topological_sort({}) == []
try:
    topological_sort({'a': {'b'}, 'b': {'a'}})
except ValueError:
    pass
else:
    raise AssertionError('cycle was accepted')
""",
    },
    {
        "name": "lru_cache",
        "prompt": """Implement a production-quality Python class LRUCache with methods:

LRUCache(capacity: int), get(key, default=None), put(key, value), and __len__().

All operations must be O(1), put returns None, get refreshes recency, updates refresh
recency, capacity <= 0 raises ValueError, and all public operations must be thread-safe.
Do not use functools.lru_cache. Output only complete Python source, no markdown.""",
        "tests": """
c = LRUCache(2)
assert len(c) == 0 and c.put('a', 1) is None
c.put('b', 2)
assert c.get('a') == 1
c.put('c', 3)
assert c.get('b') is None and c.get('c') == 3 and len(c) == 2
c.put('a', 9)
assert c.get('a') == 9 and len(c) == 2
sentinel = object()
assert c.get('missing', sentinel) is sentinel
try:
    LRUCache(0)
except ValueError:
    pass
else:
    raise AssertionError('non-positive capacity was accepted')

import threading
errors = []
def worker(offset):
    try:
        for i in range(200):
            c.put((offset, i), i)
            c.get((offset, i))
    except BaseException as exc:
        errors.append(exc)
threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]
assert not errors and len(c) <= 2
""",
    },
    {
        "name": "sliding_window_max",
        "prompt": """Implement this exact Python function:

def sliding_window_max(values: list[int], k: int) -> list[int]:

Return the maximum for every contiguous window of length k in O(n) time and O(k)
extra space. Do not mutate values. Raise ValueError when k <= 0 or k > len(values).
Output only complete Python source, with no markdown or explanation.""",
        "tests": """
source = [1, 3, -1, -3, 5, 3, 6, 7]
assert sliding_window_max(source, 3) == [3, 3, 5, 5, 6, 7]
assert source == [1, 3, -1, -3, 5, 3, 6, 7]
assert sliding_window_max([4, 4, 2, 4], 2) == [4, 4, 4]
assert sliding_window_max([-2], 1) == [-2]
for values, k in [([], 1), ([1], 0), ([1, 2], 3)]:
    try:
        sliding_window_max(values, k)
    except ValueError:
        pass
    else:
        raise AssertionError((values, k))
""",
    },
    {
        "name": "shortest_path",
        "prompt": """Implement this exact Python function:

def shortest_path(graph: dict[str, dict[str, float]], start: str, end: str) -> tuple[float, list[str]]:

The mapping is node -> outgoing neighbor weights. Include nodes occurring only as
neighbors. Use non-negative finite weights; raise ValueError if any weight is negative
or non-finite. Return (distance, path). For equal distances choose the
lexicographically smallest entire path. Return (float('inf'), []) if unreachable.
Do not mutate graph. Output only complete Python source, no markdown.""",
        "tests": """
g = {'a': {'b': 1.0, 'c': 1.0}, 'b': {'d': 1.0}, 'c': {'d': 1.0}}
assert shortest_path(g, 'a', 'd') == (2.0, ['a', 'b', 'd'])
assert shortest_path({'a': {'c': 2, 'b': 1}, 'b': {'c': 1}}, 'a', 'c') == (2.0, ['a', 'b', 'c'])
assert shortest_path({'a': {'b': 1}}, 'b', 'b') == (0.0, ['b'])
d, p = shortest_path({'a': {}, 'b': {}}, 'a', 'b')
assert d == float('inf') and p == []
assert g == {'a': {'b': 1.0, 'c': 1.0}, 'b': {'d': 1.0}, 'c': {'d': 1.0}}
for bad in [-1.0, float('inf'), float('nan')]:
    try:
        shortest_path({'a': {'b': bad}}, 'a', 'b')
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
""",
    },
    {
        "name": "deep_merge",
        "prompt": """Implement this exact Python function:

def deep_merge(base: dict, override: dict) -> dict:

Recursively merge only when both corresponding values are dictionaries. Otherwise the
override value replaces the base value. Keys present in only one input are retained.
The result must share no mutable dict/list/set objects with either input, including
objects nested inside tuples. Preserve dictionary insertion order: base keys first,
then new override keys. Do not mutate inputs. Output only complete Python source.""",
        "tests": """
base = {'db': {'host': 'a', 'opts': {'retry': 2}}, 'xs': [1], 'keep': {'z': {1}}}
over = {'db': {'port': 9, 'opts': {'retry': 4}}, 'xs': [2], 'new': ({'k': []},)}
r = deep_merge(base, over)
assert r == {'db': {'host': 'a', 'opts': {'retry': 4}, 'port': 9}, 'xs': [2], 'keep': {'z': {1}}, 'new': ({'k': []},)}
assert list(r) == ['db', 'xs', 'keep', 'new']
r['db']['opts']['retry'] = 99
r['xs'].append(3)
r['keep']['z'].add(2)
r['new'][0]['k'].append(1)
assert base == {'db': {'host': 'a', 'opts': {'retry': 2}}, 'xs': [1], 'keep': {'z': {1}}}
assert over == {'db': {'port': 9, 'opts': {'retry': 4}}, 'xs': [2], 'new': ({'k': []},)}
""",
    },
    {
        "name": "retry_decorator",
        "prompt": """Implement this exact Python decorator factory:

def retry(attempts: int, exceptions=(Exception,), base_delay: float = 0.0, sleeper=time.sleep):

The decorated function is called at most attempts times. Retry only the supplied
exception types. Before retry number n (first retry is n=1), call sleeper with
base_delay * 2**(n-1). Re-raise the final caught exception with its traceback. Reject
attempts <= 0 and base_delay < 0 with ValueError at decoration-factory creation time.
Preserve function metadata and support arbitrary args/kwargs. Output only Python source.""",
        "tests": """
import time
calls, sleeps = [], []
@retry(3, exceptions=(KeyError,), base_delay=0.25, sleeper=sleeps.append)
def f(x=0):
    'doc'
    calls.append(x)
    if len(calls) < 3:
        raise KeyError('x')
    return x + 1
assert f(x=4) == 5 and calls == [4, 4, 4] and sleeps == [0.25, 0.5]
assert f.__name__ == 'f' and f.__doc__ == 'doc'
try:
    retry(0)
except ValueError:
    pass
else:
    raise AssertionError('attempts')
try:
    retry(1, base_delay=-1)
except ValueError:
    pass
else:
    raise AssertionError('delay')
@retry(4, exceptions=(KeyError,), sleeper=lambda _: None)
def g():
    raise TypeError('do not retry')
try:
    g()
except TypeError:
    pass
else:
    raise AssertionError('wrong exception handling')
""",
    },
    {
        "name": "ttl_cache",
        "prompt": """Implement a thread-safe Python class TTLCache with methods:

TTLCache(capacity: int, ttl: float, clock=time.monotonic), get(key, default=None),
put(key, value), purge(), and __len__().

Capacity and ttl must be positive. Expiry time is insertion/update time + ttl; get does
not extend it. Expired entries behave as absent. Among live entries, get and update
refresh LRU recency. put evicts expired entries before the least-recently-used live
entry. purge returns the number removed. Public operations must be thread-safe.
Output only complete Python source, no markdown.""",
        "tests": """
now = [10.0]
c = TTLCache(2, 5.0, clock=lambda: now[0])
assert c.put('a', 1) is None
now[0] = 11.0
c.put('b', 2)
assert c.get('a') == 1
c.put('c', 3)
assert c.get('b') is None and c.get('a') == 1 and c.get('c') == 3
now[0] = 15.0
assert c.get('a') is None and len(c) == 1
now[0] = 17.0
assert c.purge() == 1 and len(c) == 0
for args in [(0, 1.0), (1, 0.0), (1, -1.0)]:
    try:
        TTLCache(*args)
    except ValueError:
        pass
    else:
        raise AssertionError(args)
""",
    },
    {
        "name": "csv_records",
        "prompt": """Implement this exact Python function using the standard csv module:

def parse_csv_records(text: str, required: set[str]) -> list[dict[str, str]]:

Parse CSV with a header. Strip surrounding whitespace from header names, reject empty
or duplicate normalized headers, and raise ValueError if required columns are absent.
Strip surrounding whitespace from unquoted and quoted field values. Blank lines are
ignored. Every data row must have exactly the header's field count or raise ValueError.
Return records in input order. An empty/blank input raises ValueError. Output only source.""",
        "tests": """
s = ' name ,note,age\\n Alice ,"x,y", 10 \\n\\n Bob , " hi " ,20\\n'
assert parse_csv_records(s, {'name', 'age'}) == [
    {'name': 'Alice', 'note': 'x,y', 'age': '10'},
    {'name': 'Bob', 'note': 'hi', 'age': '20'},
]
for bad, req in [('', set()), ('a,a\\n1,2\\n', set()), ('a,b\\n1\\n', set()), ('a\\n1,2\\n', set()), ('a\\n1\\n', {'b'}), (',b\\n1,2\\n', set())]:
    try:
        parse_csv_records(bad, req)
    except ValueError:
        pass
    else:
        raise AssertionError((bad, req))
""",
    },
    {
        "name": "async_map_ordered",
        "prompt": """Implement this exact async Python function:

async def async_map_ordered(func, items, limit: int):

func is an async callable. Run at most limit calls concurrently and return results in
input order. Consume any finite iterable exactly once. If one call raises, cancel and
await all outstanding calls, then re-raise that exception. An empty iterable returns
[]. Raise ValueError for limit <= 0. Do not call func until the coroutine is awaited.
Output only complete Python source, no markdown.""",
        "tests": """
import asyncio

async def main():
    active = peak = 0
    async def f(x):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001 * (4 - x))
        active -= 1
        return x * 2
    assert await async_map_ordered(f, iter([1, 2, 3]), 2) == [2, 4, 6]
    assert peak <= 2
    assert await async_map_ordered(f, [], 1) == []
    try:
        await async_map_ordered(f, [1], 0)
    except ValueError:
        pass
    else:
        raise AssertionError('limit')
    cancelled = asyncio.Event()
    async def bad(x):
        if x == 0:
            await asyncio.sleep(0.002)
            raise RuntimeError('boom')
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.set()
            raise
    try:
        await async_map_ordered(bad, [0, 1], 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError('missing error')
    assert cancelled.is_set()

asyncio.run(main())
""",
    },
]


def extract_source(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def request_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, dict, float]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only the requested source code. Do not reason aloud."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    elapsed = time.perf_counter() - started
    return payload["choices"][0]["message"]["content"], payload.get("usage", {}), elapsed


def request_lmstudio_completion(
    base_url: str, model: str, prompt: str, max_tokens: int
) -> tuple[str, dict, float]:
    body = {
        "model": model,
        "input": prompt,
        "system_prompt": "Return only the requested source code. Do not reason aloud.",
        "temperature": 0,
        "max_output_tokens": max_tokens,
        "reasoning": "off",
        "store": False,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/v1/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    elapsed = time.perf_counter() - started
    messages = [item["content"] for item in payload["output"] if item["type"] == "message"]
    stats = payload.get("stats", {})
    usage = {
        "prompt_tokens": stats.get("input_tokens"),
        "completion_tokens": stats.get("total_output_tokens"),
        "reasoning_tokens": stats.get("reasoning_output_tokens"),
        "tokens_per_second": stats.get("tokens_per_second"),
        "time_to_first_token_seconds": stats.get("time_to_first_token_seconds"),
    }
    return "\n".join(messages), usage, elapsed


def run_tests(source: str, tests: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="qwen36-code-") as temp_dir:
        path = Path(temp_dir) / "candidate.py"
        path.write_text(f"{source}\n\n{tests}\n", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    detail = (completed.stderr or completed.stdout).strip()
    return completed.returncode == 0, detail[-1200:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-style", choices=("openai", "lmstudio"), default="openai")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--task", choices=[task["name"] for task in TASKS])
    parser.add_argument("--output")
    args = parser.parse_args()

    results = []
    selected_tasks = [task for task in TASKS if not args.task or task["name"] == args.task]
    requester = request_lmstudio_completion if args.api_style == "lmstudio" else request_completion
    for task in selected_tasks:
        try:
            raw, usage, elapsed = requester(
                args.base_url, args.model, task["prompt"], args.max_tokens
            )
            source = extract_source(raw)
            passed, detail = run_tests(source, task["tests"])
            results.append(
                {
                    "name": task["name"],
                    "passed": passed,
                    "elapsed_seconds": round(elapsed, 3),
                    "usage": usage,
                    "error": detail or None,
                    "source": source,
                }
            )
        except Exception as exc:
            results.append({"name": task["name"], "passed": False, "error": repr(exc)})

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "api_style": args.api_style,
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "results": results,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        Path(args.output).write_text(f"{encoded}\n", encoding="utf-8")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
