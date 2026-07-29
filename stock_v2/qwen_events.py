from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import requests

from stock_v2.event_schema import MarketEvent


def _clean_text(text: str, limit: int = 500) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _extract_json_object(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]


def _clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(low, min(high, numeric))


def _load_json_payload(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Qwen sometimes copies Korean news titles into a summary string without
        # escaping embedded quotes. The model scores are still usable, so drop
        # that optional field and parse the remaining compact object.
        repaired = re.sub(
            r'"summary"\s*:\s*.*?,\s*"polarity"',
            '"polarity"',
            payload,
            count=1,
            flags=re.DOTALL,
        )
        return json.loads(repaired)


def _first_code(universe: Sequence[str]) -> str:
    for item in universe:
        code = str(item).replace("A", "").strip()
        if re.fullmatch(r"\d{6}", code):
            return code
    return str(universe[0]).replace("A", "").strip() if universe else "UNKNOWN"


@dataclass
class QwenEventExtractor:
    """Qwen extractor for news-to-graph updates.

    The llama.cpp Qwen server is started with `enable_thinking=false`; use the
    chat endpoint so that the chat template option is actually applied. The LLM
    is asked for compact scores only, and this class normalizes them into the
    stricter MarketEvent schema used by the graph state updater.
    """

    model: str = "auto"
    endpoint: str = "http://127.0.0.1:8001/v1/chat/completions"
    timeout: float = 120.0

    def build_prompt(self, title: str, summary: str, universe: Sequence[str]) -> str:
        ticker = _first_code(universe)
        return (
            "/no_think\n"
            "Korean stock-news scoring. Output one compact JSON object only. "
            "No markdown, no explanation, no <think>.\n"
            "Use the target ticker exactly; do not guess another stock code.\n"
            "Required keys only: event_type, polarity, magnitude, confidence, horizon_days, themes. "
            "Do not output title or summary fields.\n"
            "event_type one of earnings,policy,supply,contract,lawsuit,macro,theme,analyst,other. "
            "polarity -1..1, magnitude/confidence 0..1, horizon_days 1..30, themes Korean strings array.\n"
            f"Target ticker: {ticker}\n"
            f"Title: {_clean_text(title, 240)}\n"
            f"Summary: {_clean_text(summary, 360)}"
        )

    def _openai_base(self) -> str:
        if "/v1/" in self.endpoint:
            return self.endpoint.split("/v1/", 1)[0]
        return self.endpoint.rstrip("/")

    def _resolve_model(self) -> str:
        if self.model and self.model != "auto":
            return self.model
        if "/v1/" not in self.endpoint:
            return "qwen3:4b"
        response = requests.get(f"{self._openai_base()}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models:
            raise RuntimeError("OpenAI-compatible endpoint returned no models")
        return str(models[0].get("id"))

    def _extract_text(self, response_json: dict) -> str:
        if "response" in response_json:
            return str(response_json.get("response", ""))
        choices = response_json.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        if "text" in choice:
            return str(choice.get("text", ""))
        message = choice.get("message", {})
        return str(message.get("content", ""))

    def _normalize_payload(self, payload: dict[str, Any], title: str, summary: str, universe: Sequence[str]) -> dict[str, Any]:
        ticker = _first_code(universe)
        polarity = _clamp(payload.get("polarity"), -1.0, 1.0)
        magnitude = _clamp(payload.get("magnitude"), 0.0, 1.0, 0.1)
        confidence = _clamp(payload.get("confidence"), 0.0, 1.0, 0.5)
        try:
            horizon_days = max(1, min(30, int(payload.get("horizon_days", 3))))
        except Exception:
            horizon_days = 3
        themes = payload.get("themes", [])
        if isinstance(themes, str):
            themes = [themes]
        themes = [str(theme).strip() for theme in themes if str(theme).strip()][:4]
        delta = polarity * magnitude
        return {
            "event_type": str(payload.get("event_type", "other"))[:32] or "other",
            "summary": str(payload.get("summary") or _clean_text(title or summary, 120))[:180],
            "polarity": polarity,
            "magnitude": magnitude,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "affected_nodes": [ticker] + themes,
            "node_deltas": [
                {
                    "node": ticker,
                    "field": "news_score",
                    "delta": delta,
                    "confidence": confidence,
                    "half_life_days": horizon_days,
                }
            ],
            "edge_deltas": [
                {
                    "src": theme,
                    "dst": ticker,
                    "edge_type": "theme_exposure",
                    "delta_weight": abs(delta),
                    "confidence": confidence,
                    "half_life_days": horizon_days,
                }
                for theme in themes
            ],
            "raw_llm": payload,
        }

    def extract_one(self, title: str, summary: str, universe: Sequence[str]) -> MarketEvent:
        prompt = self.build_prompt(title=title, summary=summary, universe=universe)
        if "/v1/chat/completions" in self.endpoint:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self._resolve_model(),
                    "messages": [
                        {
                            "role": "system",
                            "content": "You classify Korean financial news. Do not think. Output only JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 192,
                    "stream": False,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
        elif "/v1/completions" in self.endpoint:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self._resolve_model(),
                    "prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": 192,
                    "stream": False,
                },
                timeout=self.timeout,
            )
        else:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model if self.model != "auto" else "qwen3:4b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=self.timeout,
            )
        response.raise_for_status()
        text = self._extract_text(response.json())
        payload = _load_json_payload(text)
        normalized = self._normalize_payload(payload, title=title, summary=summary, universe=universe)
        return MarketEvent.from_dict(normalized)

    def extract_many(
        self,
        articles: Iterable[tuple[str, str]],
        universe: Sequence[str],
    ) -> List[MarketEvent]:
        events = []
        for title, summary in articles:
            events.append(self.extract_one(title=title, summary=summary, universe=universe))
        return events
