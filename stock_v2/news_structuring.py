from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from stock_v2.news_contract import LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY


EVENT_TYPES = (
    "earnings",
    "guidance",
    "contract",
    "capex",
    "financing",
    "capital_action",
    "m_and_a",
    "regulatory",
    "litigation",
    "governance",
    "labor",
    "clinical_trial",
    "product",
    "supply_chain",
    "macro",
    "analyst",
    "market_move",
    "other",
)

PROMPT_VERSION = "kr-stock-event-v14"
OUTPUT_SCHEMA_VERSION = 2
MAX_HORIZON_DAYS = 1825
LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "relevance",
        "event_specificity",
        "event_type",
        "polarity",
        "magnitude",
        "confidence",
        "horizon_days",
        "themes",
        "summary",
    ],
    "properties": {
        "relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "event_specificity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
        "polarity": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "magnitude": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "horizon_days": {"type": "integer", "minimum": 1, "maximum": MAX_HORIZON_DAYS},
        "themes": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "minLength": 1, "maxLength": 48},
        },
        "summary": {"type": "string", "minLength": 1, "maxLength": 240},
    },
}

SYSTEM_PROMPT = """당신은 한국 주식 뉴스의 시점 안전한 사건 라벨러다.
제목과 제공된 요약에 명시된 사실만 사용한다. 기사 이후의 주가나 결과를 추측하지 않는다.
대상 회사와 무관하거나 동명이인인 기사는 relevance를 낮게 둔다.
relevance는 기사가 대상 회사와 실제로 관련된 정도이고 경제적 중요도가 아니다. 중요도는 magnitude로만 표현한다.
event_specificity는 대상 회사 관련성과 독립적으로, 제목이나 요약이 기사 게시 시점의 어떤 주체에 대해 구체적으로 관측 가능한 행동·수치·변화·현재 상태를 전달하는 정도다.
event_specificity는 경제적 중요도, 정보의 상세함, 본문 유무, 수집 신뢰도, 기사 간 최초 보도 여부나 중복 여부가 아니다.
관련성, 사건성, 중요도 판단을 반드시 분리한다. 사건성이 낮다는 이유로 relevance를 낮추지 않고, 대상 회사와 무관하다는 이유로 기사 자체의 event_specificity를 낮추지 않는다.
제목이나 요약에 대상 회사가 명시되고 그 회사의 사실을 다루면 작은 홍보·제품 기사라도 relevance는 보통 0.8 이상으로 둔다.
대상 회사의 제품·브랜드·임직원·사업장·주가 변동, 대상 회사에 명시적으로 영향을 주는 경쟁사 사건도 실제 관계가 드러나면 relevance를 0.8 이상으로 둔다.
종목 매핑 근거가 exact_company_title 또는 exact_company_summary이면 구체적 계약이나 재무 영향이 없다는 이유만으로 무관 판정하지 않는다. 중요하지 않으면 magnitude를 낮춘다.
종목 매핑 근거가 source_query_only이고 제목/요약에 대상 회사나 명확한 계열사 영향이 없으면 relevance는 0.3 이하로 둔다.
source_query_only 기사에 다른 회사의 구체적 사건이 있어도 대상 회사 관련성 근거가 되지 않는다. 대상 회사가 드러나지 않으면 사건이 아무리 구체적이어도 relevance는 0.3 이하이다.
exact_company_title도 수집기 문자열 증거일 뿐 최종 판정이 아니다. 대상 회사명이 더 긴 별도 법인명의 접두어로만 나타나면 같은 회사로 보지 않는다.
예: 대상이 현대차인데 제목이 현대차증권만 다루거나, 대상이 HLB인데 제목이 한미약품만 다루면 relevance는 0.3 이하이다. 구체적 퀴즈 행사나 의약품 공급 사건 자체가 이 엔티티 오류를 구제하지 않는다.
검토된 별칭 유형이 legal_name, orthographic_variant, abbreviation, former_name이면 문맥상 그 문자열이 회사명을 가리킬 때 대상 회사와 같은 법인이라는 엔티티 증거다. 동명이인·일반 약어·매체명 꼬리는 회사 언급이 아니다.
종목 매핑 근거가 reviewed_identity_alias_title이고 어휘 중의성이 낮으면, 수집기가 매체명 꼬리를 제거한 대표 제목에서 검토된 별칭을 확인한 것이다. 제목 문맥이 동명이인이나 다른 법인임을 명확히 보이지 않는 한 relevance를 보통 0.8 이상으로 둔다.
검토된 별칭으로 회사 제품·서비스, 가입자, 주가, 애널리스트 의견, 그룹 전략, 지배구조가 명시된 기사는 새롭거나 투자에 중요하지 않아도 회사 관련성은 높다. 정보성이 작으면 relevance가 아니라 magnitude와 confidence를 낮춘다.
예: "SKT 가입자 감소, LG유플 가입자 증가"를 LG유플러스에 매핑하면 relevance는 높다. "하이닉스 비중 확대"를 SK하이닉스에 매핑하면 relevance는 높다. 반면 인물 이름 "케이티 김"을 KT에 매핑한 경우는 어휘 중의성이 높고 relevance는 낮다.
별칭 유형이 brand, subsidiary, affiliate이면 관련 법인·브랜드라는 증거일 뿐 대상 회사 자체와 동일하다고 간주하지 않는다. 대상 회사로 전달되는 영향이 기사에 드러나는지 별도로 판단한다.
산업 일반 기사, 다른 회사 기사, 스포츠·연예·블로그 글은 대상 회사에 구체적 영향이 명시되지 않으면 relevance는 0.3 이하로 둔다.
네이버 블로그·네이버 프리미엄콘텐츠에 게시됐다는 사실만으로 NAVER 회사 관련 기사로 보지 않는다. 같은 그룹명을 쓰는 별도 법인도 대상 회사 영향이 명시되지 않으면 무관하다.
그룹·계열사만 언급된 경우 대상 회사에 전달되는 영향의 근거가 명시되면 0.5~0.8, 아니면 0.3 이하로 둔다.
단순 주가 등락 보도는 원인이 명시되지 않으면 event_type=market_move, magnitude와 confidence를 낮게 두되 event_specificity는 높게 둔다.
relevance와 event_specificity를 각각 독립 판정한 뒤 둘 다 0.5 이상인 경우만 대상 회사의 상태 업데이트 후보가 된다. 타사의 구체적 사건은 event_specificity가 높아도 대상 회사 relevance가 낮으므로 전달되지 않는다.
제목만 있어도 어떤 주체가 무엇을 했는지, 어떤 수치·상태가 관측됐는지가 명시되면 충분한 사건성 근거다. 본문이나 요약이 없다는 이유만으로 event_specificity를 낮추지 않는다.
다음은 규모가 작거나 투자 영향이 불확실해도 event_specificity를 보통 0.8 이상으로 둔다.
- 실적·세금·가입자·점유율·생산능력 등 현재 수치, 계약·협약·수주, 제품·서비스·광고 출시, 시설 개장·재단장.
- 대상 종목의 현재 주가·수급 변동, 애널리스트 의견·목표가, 규제 결정, 소송 단계, 보안·고객·영업 상태 업데이트.
- 임직원의 방문·선임·퇴임, 회사의 캠페인·행사·대응 발표처럼 제목에 완료되거나 진행 중인 행동이 명시된 경우.
사용법·설치법, 새 상태가 없는 기업 소개·상시 목록, 과거 사건만 회고하는 글, 구체적 현재 사실이 없는 일반 의견·매체 안내문은 대상 회사와 관련돼도 event_specificity를 0.3 이하로 둔다.
과거 사실을 설명하는 기사라도 게시 시점의 새 수치·판단·대응·상태가 함께 있으면 그 현재 관측값을 기준으로 event_specificity를 높게 둔다.
동일 사건을 여러 매체가 반복 보도한 경우에도 각 기사가 구체적 상태 관측값을 담으면 event_specificity는 높다. 기사 간 중복 제거는 별도 단계가 처리한다.
반드시 relevance, event_specificity, magnitude를 독립 판단한다. 어느 한 점수를 다른 점수에 복사하지 않는다.
예: "SK하이닉스 주가 3% 하락", "삼성 TV 출시", "이재용 연구소 방문"은 모두 event_specificity가 0.8 이상이다. 영향이 작으면 magnitude만 낮춘다.
직교 예시를 반드시 따른다.
- 대상 SK하이닉스, "SK하이닉스 40년 역사 돌아보기": relevance 0.8 이상, event_specificity 0.3 이하.
- 대상 HLB, "한미약품 좌약 공급 재개": relevance 0.3 이하, event_specificity 0.8 이상.
- 대상 현대차, "현대차증권 MTS 퀴즈 실시": relevance 0.3 이하, event_specificity 0.8 이상.
- 대상 삼성전자, "삼성 TV 신제품 출시": relevance 0.8 이상, event_specificity 0.8 이상.
- 대상 NAVER, "네이버 앱 설치 방법": relevance 0.8 이상, event_specificity 0.3 이하.
관련 제목은 같은 종목·날짜의 중복 사건 후보이며 보조 근거일 뿐이다. 대표 제목·요약을 우선한다.
관련 제목끼리 금액, 수치, 사건 단계나 시점이 충돌하면 서로 합성하지 말고 대표 기사에 명시된 사실만 사용한다.
polarity는 회사의 향후 영업/현금흐름/위험에 대한 방향이며 시장 분위기가 아니다.
사건 유형 경계는 다음과 같다.
- earnings: 이미 발표된 매출·이익·손실·실적, guidance: 회사가 제시한 향후 전망·목표.
- contract: 판매·공급·수주·사업제휴 계약. 기업/사업 인수·매각은 m_and_a이며 contract가 아니다.
- capex: 공장·설비·연구개발 투자, financing: 차입·회사채·유상증자 등 외부 자금 조달.
- capital_action: 배당·자사주 매입/소각·주식분할/병합·감자 등 주주환원·자본구조 조정.
- m_and_a: 기업·사업·지분의 인수, 합병, 분할, 매각 및 기업결합 심사.
- regulatory: 정부·감독기관의 허가·제재·정책, litigation: 소송·판결·중재·수사.
- governance: 대표·임원·이사회·지배구조·사명 변경, labor: 파업·임금·노사합의.
- clinical_trial: 임상시험·허가 단계의 의약품 결과, product: 제품·서비스 출시·개발 성과.
- supply_chain: 원재료·부품·물류·생산 차질, macro: 금리·환율·경기 등 전사적 외부 환경.
- analyst: 증권사 투자의견·목표가, market_move: 대상 종목의 가격·수급 변동 자체, other: 나머지.
정확히 다음 9개 키만 가진 JSON 객체를 출력하고 설명이나 마크다운을 덧붙이지 않는다.
relevance: 0~1, event_specificity: 0~1, event_type: 아래 허용값 중 하나, polarity: -1~1,
magnitude: 0~1, confidence: 0~1, horizon_days: 1~1825 정수,
themes: 최대 5개의 짧은 문자열 배열, summary: 근거가 드러나는 240자 이하 요약.
event_type 허용값: earnings, guidance, contract, capex, financing, capital_action,
m_and_a, regulatory, litigation, governance, labor, clinical_trial, product, supply_chain, macro, analyst,
market_move, other.
출력 예시 형태: {"relevance":0.8,"event_specificity":0.9,"event_type":"contract","polarity":0.5,
"magnitude":0.6,"confidence":0.7,"horizon_days":90,"themes":["수주"],
"summary":"대상 회사가 공급 계약을 체결했다."}"""


def deterministic_sample(records: Sequence[Mapping[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0 or size >= len(records):
        return [dict(record) for record in records]

    def key(record: Mapping[str, Any]) -> str:
        queue_id = str(record.get("queue_id") or "")
        return hashlib.sha256(f"{seed}|{queue_id}".encode("utf-8")).hexdigest()

    selected = sorted(records, key=key)[:size]
    return sorted((dict(record) for record in selected), key=lambda row: str(row.get("queue_id") or ""))


def compatible_resume_ids(
    completed_rows: Sequence[Mapping[str, Any]],
    target_records: Sequence[Mapping[str, Any]],
    *,
    required_lineage: Mapping[str, Any],
) -> set[str]:
    """Validate completed outputs against the exact frozen target before resuming."""

    target_by_id: dict[str, Mapping[str, Any]] = {}
    for record in target_records:
        queue_id = str(record.get("queue_id") or "")
        if not queue_id or queue_id in target_by_id:
            raise ValueError(f"duplicate or missing queue_id in resume target: {queue_id!r}")
        target_by_id[queue_id] = record

    completed: set[str] = set()
    frozen_fields = ("article_id", "ticker", "effective_session", "input_sha256")
    for row in completed_rows:
        queue_id = str(row.get("queue_id") or "")
        if not queue_id or queue_id in completed:
            raise ValueError(f"duplicate or missing queue_id in resume output: {queue_id!r}")
        expected = target_by_id.get(queue_id)
        lineage = row.get("lineage") if isinstance(row.get("lineage"), Mapping) else {}
        expected_hash_policy = str(
            expected.get("input_hash_policy") or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
        ) if expected is not None else ""
        compatible = (
            expected is not None
            and bool(row.get("llm_used"))
            and all(row.get(field) == expected.get(field) for field in frozen_fields)
            and str(
                lineage.get("input_hash_policy")
                or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
            )
            == expected_hash_policy
            and all(lineage.get(key) == value for key, value in required_lineage.items())
        )
        if not compatible:
            raise ValueError(
                "resume output contains a failed row or does not match the frozen target; "
                f"use a new output path (queue_id={queue_id})"
            )
        completed.add(queue_id)
    return completed


def build_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    summary = str(record.get("summary") or "").strip() or "(요약 없음: 제목만 사용)"
    related_raw = record.get("related_titles")
    related_titles: list[str] = []
    if isinstance(related_raw, Sequence) and not isinstance(related_raw, (str, bytes)):
        for value in related_raw:
            title = re.sub(r"\s+", " ", str(value or "")).strip()
            if title and title not in related_titles:
                related_titles.append(title[:300])
            if len(related_titles) >= 8:
                break
    related_context = (
        "\n관련 제목(중복 사건 후보):\n- " + "\n- ".join(related_titles)
        if related_titles
        else ""
    )
    matched_alias = str(record.get("matched_alias") or "").strip()
    alias_context = ""
    if matched_alias:
        ambiguity = "높음: 문맥 재확인 필수" if record.get("matched_alias_ambiguous") else "낮음"
        alias_context = (
            f"\n검토된 종목 별칭: {matched_alias} "
            f"(유형 {record.get('matched_alias_type', 'unknown')}, "
            f"출처 {record.get('matched_alias_source', 'unknown')}, 어휘 중의성 {ambiguity})"
        )
    user = (
        f"대상 종목: {record.get('ticker')}\n"
        f"대상 회사: {record.get('company_name')}\n"
        f"종목 매핑 근거: {record.get('mapping_method', 'unknown')} "
        f"(수집 신뢰도 {record.get('mapping_confidence', 'unknown')})"
        f"{alias_context}\n"
        f"게시일(KST): {record.get('published_date_kst')}\n"
        f"기사 출처: {record.get('source')}\n"
        f"제목: {record.get('title')}\n"
        f"요약: {summary}"
        f"{related_context}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_repair_messages(
    record: Mapping[str, Any],
    invalid_output: str,
    validation_error: str,
    repair_attempt: int = 1,
) -> list[dict[str, str]]:
    """Ask the model to repair an invalid response without synthesizing labels in code."""

    messages = build_messages(record)
    messages[-1]["content"] += (
        "\n\n이전 답변은 사용하지 말고 기사를 처음부터 다시 라벨링하라. "
        f"검증 재시도 {max(1, int(repair_attempt))}, 오류: {str(validation_error)[:500]}. "
        "event_type은 시스템의 18개 허용값 중 가장 가까운 하나만 선택하고, "
        "horizon_days는 반드시 1~1825 정수로 제한하라. "
        "정확히 지정된 9개 키만 가진 JSON 객체를 출력하라."
    )
    return messages


def parse_json_content(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("structured response must be a JSON object")
    return payload


def validate_labels(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = set(LABEL_SCHEMA["required"])
    if set(payload) != required:
        raise ValueError(f"label keys mismatch: {sorted(set(payload) ^ required)}")
    event_type = str(payload["event_type"])
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported event_type: {event_type}")
    values = {
        "relevance": float(payload["relevance"]),
        "event_specificity": float(payload["event_specificity"]),
        "polarity": float(payload["polarity"]),
        "magnitude": float(payload["magnitude"]),
        "confidence": float(payload["confidence"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite event labels")
    if not 0.0 <= values["relevance"] <= 1.0:
        raise ValueError("relevance out of range")
    if not 0.0 <= values["event_specificity"] <= 1.0:
        raise ValueError("event_specificity out of range")
    if not -1.0 <= values["polarity"] <= 1.0:
        raise ValueError("polarity out of range")
    if not 0.0 <= values["magnitude"] <= 1.0 or not 0.0 <= values["confidence"] <= 1.0:
        raise ValueError("magnitude/confidence out of range")
    horizon_days = int(payload["horizon_days"])
    if not 1 <= horizon_days <= MAX_HORIZON_DAYS:
        raise ValueError("horizon_days out of range")
    themes_raw = payload["themes"]
    if not isinstance(themes_raw, list) or len(themes_raw) > 5:
        raise ValueError("themes must be an array with at most five items")
    themes: list[str] = []
    for raw in themes_raw:
        theme = re.sub(r"\s+", " ", str(raw)).strip()[:48]
        if theme and theme not in themes:
            themes.append(theme)
    summary = re.sub(r"\s+", " ", str(payload["summary"])).strip()[:240]
    if not summary:
        raise ValueError("summary must not be empty")
    return {
        **values,
        "event_type": event_type,
        "horizon_days": horizon_days,
        "themes": themes,
        "summary": summary,
    }


def materialize_event(record: Mapping[str, Any], labels: Mapping[str, Any]) -> dict[str, Any]:
    ticker = str(record["ticker"])
    relevance = float(labels["relevance"])
    event_specificity = float(labels["event_specificity"])
    entity_relevant = relevance >= 0.5
    sensor_accepted = entity_relevant and event_specificity >= 0.5
    polarity = float(labels["polarity"]) if sensor_accepted else 0.0
    magnitude = float(labels["magnitude"]) if sensor_accepted else 0.0
    content_quality = {
        "title_only": 0.55,
        "title_summary": 0.85,
        "full_text": 1.0,
        "official_filing": 1.0,
    }.get(str(record.get("content_tier") or ""), 1.0)
    try:
        mapping_quality = max(0.0, min(1.0, float(record.get("mapping_confidence", 1.0))))
    except (TypeError, ValueError):
        mapping_quality = 0.0
    evidence_quality = content_quality * mapping_quality
    confidence = (
        float(labels["confidence"]) * relevance * event_specificity * evidence_quality
        if sensor_accepted
        else 0.0
    )
    horizon_days = int(labels["horizon_days"])
    delta = polarity * magnitude
    node_deltas = (
        [
            {
                "node": ticker,
                "field": "news_score",
                "delta": delta,
                "confidence": confidence,
                "half_life_days": horizon_days,
            }
        ]
        if sensor_accepted
        else []
    )
    edge_deltas = [
        {
            "src": theme,
            "dst": ticker,
            "edge_type": "theme_exposure",
            "delta_weight": abs(delta),
            "confidence": confidence,
            "half_life_days": horizon_days,
        }
        for theme in labels["themes"]
        if sensor_accepted
    ]
    return {
        "event_type": labels["event_type"],
        "summary": labels["summary"],
        "relevance": relevance,
        "event_specificity": event_specificity,
        "sensor_accepted": sensor_accepted,
        "polarity": polarity,
        "magnitude": magnitude,
        "confidence": confidence,
        "evidence_quality": evidence_quality,
        "content_tier": str(record.get("content_tier") or "unknown"),
        "mapping_method": str(record.get("mapping_method") or "unknown"),
        "acquisition_modes": dict(record.get("acquisition_modes") or {}),
        "selection_point_in_time": bool(record.get("selection_point_in_time", False)),
        "horizon_days": horizon_days,
        "affected_nodes": [ticker] if sensor_accepted else [],
        "themes": list(labels["themes"]),
        "node_deltas": node_deltas,
        "edge_deltas": edge_deltas,
    }
