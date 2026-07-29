from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from stock_v2.edge_update import EdgeDelta


@dataclass(frozen=True)
class NodeDelta:
    node: str
    field: str
    delta: float
    confidence: float = 1.0
    half_life_days: float = 3.0


@dataclass(frozen=True)
class MarketEvent:
    """Structured event emitted by an LLM/news parser."""

    event_type: str
    summary: str
    polarity: float
    magnitude: float
    confidence: float
    horizon_days: int
    affected_nodes: List[str] = field(default_factory=list)
    node_deltas: List[NodeDelta] = field(default_factory=list)
    edge_deltas: List[EdgeDelta] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketEvent":
        node_deltas = [
            NodeDelta(
                node=str(item.get("node", "")),
                field=str(item.get("field", "news_score")),
                delta=float(item.get("delta", 0.0)),
                confidence=float(item.get("confidence", data.get("confidence", 1.0))),
                half_life_days=float(item.get("half_life_days", data.get("horizon_days", 3))),
            )
            for item in data.get("node_deltas", [])
            if item.get("node")
        ]
        edge_deltas = [
            EdgeDelta(
                src=str(item.get("src", "")),
                dst=str(item.get("dst", "")),
                edge_type=str(item.get("edge_type", "llm_event")),
                delta_weight=float(item.get("delta_weight", 0.0)),
                confidence=float(item.get("confidence", data.get("confidence", 1.0))),
                half_life_days=float(item.get("half_life_days", data.get("horizon_days", 3))),
            )
            for item in data.get("edge_deltas", [])
            if item.get("src") and item.get("dst")
        ]
        return cls(
            event_type=str(data.get("event_type", "unknown")),
            summary=str(data.get("summary", "")),
            polarity=max(-1.0, min(1.0, float(data.get("polarity", 0.0)))),
            magnitude=max(0.0, min(1.0, float(data.get("magnitude", 0.0)))),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            horizon_days=max(1, int(data.get("horizon_days", 3))),
            affected_nodes=[str(node) for node in data.get("affected_nodes", [])],
            node_deltas=node_deltas,
            edge_deltas=edge_deltas,
            raw=data,
        )

    def implied_node_deltas(self, field: str = "news_score") -> List[NodeDelta]:
        if self.node_deltas:
            return list(self.node_deltas)
        shock = self.polarity * self.magnitude
        return [
            NodeDelta(
                node=node,
                field=field,
                delta=shock,
                confidence=self.confidence,
                half_life_days=float(self.horizon_days),
            )
            for node in self.affected_nodes
        ]

    def implied_edge_deltas(self) -> List[EdgeDelta]:
        if self.edge_deltas:
            return list(self.edge_deltas)
        if len(self.affected_nodes) < 2:
            return []
        deltas: List[EdgeDelta] = []
        base = self.magnitude * self.confidence
        for src in self.affected_nodes:
            for dst in self.affected_nodes:
                if src == dst:
                    continue
                deltas.append(
                    EdgeDelta(
                        src=src,
                        dst=dst,
                        edge_type=f"event:{self.event_type}",
                        delta_weight=base,
                        confidence=self.confidence,
                        half_life_days=float(self.horizon_days),
                    )
                )
        return deltas


def merge_events(events: Iterable[MarketEvent]) -> tuple[List[NodeDelta], List[EdgeDelta]]:
    node_deltas: List[NodeDelta] = []
    edge_deltas: List[EdgeDelta] = []
    for event in events:
        node_deltas.extend(event.implied_node_deltas())
        edge_deltas.extend(event.implied_edge_deltas())
    return node_deltas, edge_deltas
