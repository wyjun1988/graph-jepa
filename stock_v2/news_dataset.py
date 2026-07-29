from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import pandas as pd

from stock_v2.dataset_integrity import (
    canonical_url,
    clean_text,
    load_json,
    load_jsonl,
    normalize_ticker,
    parse_timestamp,
    sha256_file,
)
from stock_v2.news_contract import (
    BASE_NEWS_QUEUE_ID_POLICY,
    NEWS_QUEUE_INPUT_HASH_POLICY,
    news_queue_id,
    news_queue_input_sha256,
)
from stock_v2.news_aliases import (
    best_alias_match,
    is_lexically_ambiguous_short_name,
    text_contains_alias,
    validate_alias_registry,
)


def _normalized_title(value: Any, source: Any = "") -> str:
    title = clean_text(value).lower()
    source_text = clean_text(source).lower()
    if source_text:
        title = re.sub(rf"\s*[-|]\s*{re.escape(source_text)}\s*$", "", title)
    title = re.sub(r"[^0-9a-z가-힣]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _usable_summary(value: Any, title: str, source: str) -> str:
    summary = clean_text(value)
    normalized = _normalized_title(summary, source)
    title_normalized = _normalized_title(title, source)
    if not normalized or normalized == title_normalized:
        return ""
    remainder = normalized.replace(title_normalized, "", 1).strip() if title_normalized in normalized else normalized
    source_normalized = _normalized_title(source)
    if source_normalized:
        remainder = remainder.replace(source_normalized, "").strip()
    return "" if len(remainder) < 12 else summary


def _article_key(
    title: str,
    source: str,
    published_date: str,
    url: str,
    source_provider: str = "",
) -> str:
    if source_provider == "opendart" and url:
        identity = f"official-url-v1|{canonical_url(url)}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
    normalized_title = _normalized_title(title, source)
    if normalized_title:
        identity = f"title-v1|{normalized_title}|{_normalized_title(source)}|{published_date}"
    else:
        identity = f"url-v1|{canonical_url(url)}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _next_session(calendar: pd.DatetimeIndex, published_date: pd.Timestamp) -> str | None:
    candidates = calendar[calendar > published_date.normalize()]
    return str(candidates[0].date()) if len(candidates) else None


def _source_provider(record: Mapping[str, Any]) -> str:
    source = str(record.get("source") or "").lower()
    if "opendart" in source or source.startswith("dart"):
        return "opendart"
    if "google" in source:
        return "google_news_rss"
    if "naver" in source:
        return "naver_search"
    return source or "unknown"


def _selection_lineage(
    record: Mapping[str, Any],
    source_provider: str,
    published: pd.Timestamp | None,
) -> tuple[str, bool]:
    if source_provider == "opendart":
        return "official_retrospective_ledger", True
    if isinstance(record.get("query_window"), Mapping):
        return "retrospective_discovery_search", False
    collected_raw = record.get("collected_at_utc") or record.get("ts")
    collected = pd.to_datetime(collected_raw, errors="coerce", utc=True)
    if published is not None and not pd.isna(collected):
        lag = pd.Timestamp(collected) - published
        if pd.Timedelta(0) <= lag <= pd.Timedelta(days=7):
            return "live_capture", True
    return "unknown_selection", False


def _mapping_evidence(
    ticker: str,
    company_name: str,
    title: str,
    raw_summary: str,
    alias_rows: Sequence[Mapping[str, Any]] = (),
    published_date: Any = None,
    source: str = "",
) -> tuple[str, float, str | None, str | None, str | None, bool]:
    title_for_mapping = clean_text(title)
    source_text = clean_text(source)
    if source_text:
        title_for_mapping = re.sub(
            rf"\s*[-|]\s*{re.escape(source_text)}\s*$",
            "",
            title_for_mapping,
            flags=re.IGNORECASE,
        ).strip()
        if re.sub(r"\s+", "", source_text).lower() == "네이버프리미엄콘텐츠":
            title_for_mapping = re.sub(
                r"(?:\s*:\s*)?네이버\s*프리미엄\s*콘텐츠",
                " ",
                title_for_mapping,
                flags=re.IGNORECASE,
            ).strip()
    title_text = re.sub(r"[^0-9a-z가-힣]+", "", title_for_mapping.lower())
    summary_text = re.sub(r"[^0-9a-z가-힣]+", "", clean_text(raw_summary).lower())
    company_aliases = {
        clean_text(company_name),
        clean_text(re.sub(r"^(?:주식회사|\(주\))\s*", "", company_name)),
    }
    company_aliases.discard("")
    for company_alias in sorted(company_aliases, key=len, reverse=True):
        if len(company_alias) >= 2 and text_contains_alias(title_for_mapping, company_alias):
            ambiguous = is_lexically_ambiguous_short_name(company_alias)
            return (
                "ambiguous_short_company_title" if ambiguous else "exact_company_title",
                0.60 if ambiguous else 1.0,
                company_name,
                "security_name",
                "universe_manifest",
                ambiguous,
            )
    if ticker in title_text:
        return "ticker_title", 0.95, ticker, "ticker", "universe_manifest", False
    title_alias = best_alias_match(title_for_mapping, alias_rows, published_date)
    if title_alias is not None:
        identity = title_alias["relationship"] == "identity"
        ambiguous = bool(title_alias.get("lexically_ambiguous", False))
        confidence = float(title_alias["confidence"]) * (
            0.55 if identity and ambiguous else (0.95 if identity else (0.45 if ambiguous else 0.65))
        )
        return (
            (
                "reviewed_ambiguous_identity_alias_title"
                if identity and ambiguous
                else ("reviewed_identity_alias_title" if identity else "reviewed_related_entity_title")
            ),
            confidence,
            str(title_alias["alias"]),
            str(title_alias["alias_type"]),
            str(title_alias["source"]),
            ambiguous,
        )
    for company_alias in sorted(company_aliases, key=len, reverse=True):
        if len(company_alias) >= 2 and text_contains_alias(raw_summary, company_alias):
            ambiguous = is_lexically_ambiguous_short_name(company_alias)
            return (
                "ambiguous_short_company_summary" if ambiguous else "exact_company_summary",
                0.45 if ambiguous else 0.80,
                company_name,
                "security_name",
                "universe_manifest",
                ambiguous,
            )
    if ticker in summary_text:
        return "ticker_summary", 0.75, ticker, "ticker", "universe_manifest", False
    summary_alias = best_alias_match(raw_summary, alias_rows, published_date)
    if summary_alias is not None:
        identity = summary_alias["relationship"] == "identity"
        ambiguous = bool(summary_alias.get("lexically_ambiguous", False))
        confidence = float(summary_alias["confidence"]) * (
            0.45 if identity and ambiguous else (0.80 if identity else (0.35 if ambiguous else 0.50))
        )
        return (
            (
                "reviewed_ambiguous_identity_alias_summary"
                if identity and ambiguous
                else ("reviewed_identity_alias_summary" if identity else "reviewed_related_entity_summary")
            ),
            confidence,
            str(summary_alias["alias"]),
            str(summary_alias["alias_type"]),
            str(summary_alias["source"]),
            ambiguous,
        )
    return "source_query_only", 0.25, None, None, None, False


def _mapping_rank(method: str) -> int:
    return {
        "source_query_only": 0,
        "ambiguous_short_company_summary": 1,
        "reviewed_ambiguous_identity_alias_summary": 1,
        "reviewed_related_entity_summary": 1,
        "ambiguous_short_company_title": 2,
        "reviewed_ambiguous_identity_alias_title": 2,
        "reviewed_related_entity_title": 2,
        "ticker_summary": 3,
        "reviewed_identity_alias_summary": 4,
        "exact_company_summary": 5,
        "ticker_title": 6,
        "reviewed_identity_alias_title": 7,
        "exact_company_title": 8,
        "official_corp_code": 9,
    }.get(method, -1)


def dart_neutral_family(title: Any) -> tuple[str, str] | None:
    """Map every title-only filing to a non-directional typed count sensor."""

    normalized = clean_text(title)
    normalized = re.sub(r"^(?:\[[^]]+\]\s*)+", "", normalized).strip()
    if not normalized:
        return None
    policies = (
        (r"^(?:임원ㆍ주요주주특정증권등소유상황보고서|주식등의대량보유상황보고서|최대주주등소유주식변동신고서)", "ownership_filing", "governance"),
        (r"^(?:의결권대리행사권유참고서류|주주총회소집공고|주주명부폐쇄기간또는기준일설정)", "shareholder_administration", "governance"),
        (r"^(?:투자설명서|일괄신고추가서류|증권발행실적보고서)", "securities_document", "financing"),
        (r"^(?:기업설명회\(IR\)개최|결산실적공시예고)", "scheduled_investor_relations", "other"),
        (r"^(?:대규모기업집단현황공시|기업지배구조보고서공시)", "periodic_governance_filing", "governance"),
    )
    for pattern, family, event_type in policies:
        if re.search(pattern, normalized):
            return family, event_type
    event_patterns = (
        (r"실적|매출액|손익|재무제표|감사보고", "earnings"),
        (r"단일판매|공급계약|수주", "contract"),
        (r"신규시설투자|시설투자", "capex"),
        (r"유상증자|회사채|전환사채|신주인수권부사채|차입|담보제공|채무보증|증권신고서", "financing"),
        (r"배당|자기주식|자사주|주식분할|주식병합|감자|소각", "capital_action"),
        (r"합병|분할|영업양수|영업양도|타법인주식.*(?:취득|처분)|공개매수", "m_and_a"),
        (r"과징금|제재|허가|승인|불성실공시|거래정지", "regulatory"),
        (r"소송|판결|중재|수사", "litigation"),
        (r"대표이사|이사회|주주총회|이사|최대주주|지배구조|사명변경", "governance"),
        (r"파업|노사|임금", "labor"),
        (r"임상|품목허가|의약품", "clinical_trial"),
        (r"제품|서비스|특허", "product"),
        (r"생산중단|생산재개|공급중단", "supply_chain"),
    )
    event_type = next(
        (candidate for pattern, candidate in event_patterns if re.search(pattern, normalized)),
        "other",
    )
    title_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"dart_{event_type}_{title_digest}", event_type


def _is_direct_url(url: str) -> bool:
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return False
    return bool(host) and host not in {"news.google.com", "search.naver.com"}


def _candidate_rank(article: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        1 if article.get("source_provider") == "opendart" else 0,
        1 if article["published_precision"] == "datetime" else 0,
        1 if _is_direct_url(str(article["url"])) else 0,
        len(str(article["summary"])),
        str(article["source_provider"]),
    )


def _cluster_title(article: Mapping[str, Any]) -> str:
    title = _normalized_title(article.get("title"), article.get("source"))
    title = re.sub(r"^(?:속보|단독|종합|공시|특징주|업데이트)\s+", "", title)
    return re.sub(r"\s+", " ", title).strip()


def _title_tokens(title: str) -> set[str]:
    return {token for token in re.findall(r"[0-9a-z가-힣]+", title) if len(token) >= 2}


def _character_ngrams(title: str, size: int = 3) -> set[str]:
    compact = title.replace(" ", "")
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _set_similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _near_duplicate_titles(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < 12:
        return False
    sequence_ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
    token_ratio = _set_similarity(_title_tokens(left), _title_tokens(right))
    character_ratio = _set_similarity(_character_ngrams(left), _character_ngrams(right))
    return sequence_ratio >= 0.90 or (
        sequence_ratio >= 0.82 and token_ratio >= 0.55 and character_ratio >= 0.68
    )


def cluster_news_articles(
    article_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic same-day near-duplicate clusters without dropping articles."""

    rows = [dict(row) for row in article_rows]
    if not rows:
        return [], []
    rows.sort(key=lambda row: (str(row.get("published_date_kst")), str(row.get("article_id"))))
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    token_index: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    url_index: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    normalized_titles: list[str] = []
    for index, article in enumerate(rows):
        date = str(article.get("published_date_kst") or "")
        title = _cluster_title(article)
        normalized_titles.append(title)
        url = canonical_url(article.get("url"))
        candidates: set[int] = set(url_index.get((date, url), [])) if url else set()
        is_official = article.get("source_provider") == "opendart"
        tokens = [] if is_official else sorted(_title_tokens(title), key=lambda token: (-len(token), token))[:8]
        for token in tokens:
            candidates.update(token_index.get((date, token), []))
        for candidate_index in sorted(candidates):
            candidate_url = canonical_url(rows[candidate_index].get("url"))
            same_url = bool(url and candidate_url and url == candidate_url)
            official_pair = (
                article.get("source_provider") == "opendart"
                or rows[candidate_index].get("source_provider") == "opendart"
            )
            if same_url or (not official_pair and _near_duplicate_titles(title, normalized_titles[candidate_index])):
                union(index, candidate_index)
        if url:
            url_index[(date, url)].append(index)
        if not is_official:
            for token in tokens:
                token_index[(date, token)].append(index)

    groups: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)

    cluster_rows: list[dict[str, Any]] = []
    for member_indices in groups.values():
        members = sorted(str(rows[index]["article_id"]) for index in member_indices)
        cluster_id = hashlib.sha256(f"news-cluster-v1|{members[0]}".encode("utf-8")).hexdigest()
        representative_index = max(
            member_indices,
            key=lambda index: (_candidate_rank(rows[index]), str(rows[index]["article_id"])),
        )
        representative_id = str(rows[representative_index]["article_id"])
        for index in member_indices:
            rows[index]["event_cluster_id"] = cluster_id
            rows[index]["cluster_size"] = len(member_indices)
            rows[index]["is_cluster_representative"] = index == representative_index
        cluster_rows.append(
            {
                "schema_version": 2,
                "event_cluster_id": cluster_id,
                "published_date_kst": str(rows[member_indices[0]]["published_date_kst"]),
                "representative_article_id": representative_id,
                "article_ids": members,
                "article_count": len(members),
                "clustering_policy": "same_day_url_or_title_similarity_v1",
            }
        )
    cluster_rows.sort(key=lambda row: (str(row["published_date_kst"]), str(row["event_cluster_id"])))
    rows.sort(key=lambda row: (str(row["published_date_kst"]), str(row["article_id"])))
    return rows, cluster_rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_calendar(paths: Sequence[Path], start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex([])
    for path in paths:
        frame = pd.read_csv(path)
        date_column = "Date" if "Date" in frame.columns else frame.columns[0]
        dates = pd.DatetimeIndex(pd.to_datetime(frame[date_column], errors="coerce").dropna()).normalize()
        result = result.union(dates)
    return result[(result >= start) & (result <= end)].sort_values().unique()


def filter_universe(
    universe: Mapping[str, Mapping[str, Any]],
    include_tickers_raw: Any,
) -> dict[str, Mapping[str, Any]]:
    if include_tickers_raw is None:
        return dict(universe)
    if not isinstance(include_tickers_raw, Sequence) or isinstance(
        include_tickers_raw, (str, bytes)
    ):
        raise ValueError("include_tickers must be an array")
    include_tickers = {normalize_ticker(value) for value in include_tickers_raw}
    if "" in include_tickers:
        raise ValueError("include_tickers contains an invalid ticker")
    unknown_tickers = include_tickers - set(universe)
    if unknown_tickers:
        raise ValueError(f"include_tickers contains unknown tickers: {sorted(unknown_tickers)}")
    return {ticker: row for ticker, row in universe.items() if ticker in include_tickers}


def curate_news_dataset(
    repo_root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    universe_payload = load_json(repo_root / str(config["universe_manifest"]))
    full_universe = {
        normalize_ticker(row.get("ticker")): row
        for row in universe_payload.get("universe", [])
        if normalize_ticker(row.get("ticker"))
    }
    universe = filter_universe(full_universe, config.get("include_tickers"))
    release_start = pd.Timestamp(config["start"]).normalize()
    release_end = pd.Timestamp(config["end"]).normalize()
    calendar = load_calendar(
        [repo_root / str(path) for path in config["calendar_paths"]],
        release_start,
        release_end,
    )
    if not len(calendar):
        raise ValueError("news curation requires a non-empty trading calendar")

    input_files: list[dict[str, Any]] = []
    combined_alias_payload: dict[str, Any] = {"schema_version": 1, "aliases": []}
    for alias_path_value in config.get("alias_registry_paths", []):
        alias_path = repo_root / str(alias_path_value)
        alias_payload = load_json(alias_path)
        rows = alias_payload.get("aliases") if isinstance(alias_payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError(f"invalid company alias registry: {alias_path}")
        combined_alias_payload["aliases"].extend(rows)
        input_files.append(
            {
                "path": str(alias_path_value),
                "sha256": sha256_file(alias_path),
                "rows": len(rows),
                "role": "company_alias_registry",
            }
        )
    full_aliases_by_ticker = validate_alias_registry(combined_alias_payload, full_universe)
    aliases_by_ticker = {
        ticker: rows for ticker, rows in full_aliases_by_ticker.items() if ticker in universe
    }

    articles: dict[str, dict[str, Any]] = {}
    pit_articles: dict[str, dict[str, Any]] = {}
    article_source_ids: defaultdict[str, set[str]] = defaultdict(set)
    article_acquisition_modes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    article_pit_acquisition_modes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    article_pit_occurrences: Counter[str] = Counter()
    article_non_pit_occurrences: Counter[str] = Counter()
    mappings: dict[tuple[str, str], dict[str, Any]] = {}
    pit_mappings: dict[tuple[str, str], dict[str, Any]] = {}
    occurrences: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    load_reports: dict[str, Any] = {}
    for raw_path in config.get("raw_paths", []):
        path = repo_root / str(raw_path)
        rows, load_report = load_jsonl(path)
        load_reports[str(raw_path)] = load_report
        input_files.append({"path": str(raw_path), "sha256": sha256_file(path), "rows": load_report["rows"]})
        if load_report["invalid_json"] or load_report["invalid_rows"]:
            raise ValueError(f"invalid JSONL input: {path}")
        for row_number, record in enumerate(rows, start=1):
            raw_counts["rows"] += 1
            source_record_id = str(record.get("id") or "").strip()
            ticker = normalize_ticker(record.get("ticker"))
            if not ticker:
                raw_counts["invalid_ticker"] += 1
                quarantine.append({"source_record_id": source_record_id, "reason": "invalid_ticker"})
                continue
            if ticker not in full_universe:
                raw_counts["outside_universe"] += 1
                quarantine.append({"source_record_id": source_record_id, "ticker": ticker, "reason": "outside_universe"})
                continue
            if ticker not in universe:
                raw_counts["excluded_by_include_tickers"] += 1
                continue
            payload = record.get("article") if isinstance(record.get("article"), Mapping) else record
            source_provider = _source_provider(record)
            published_raw = payload.get("published") or record.get("published")
            published, precision = parse_timestamp(published_raw)
            if published is None:
                raw_counts["invalid_published"] += 1
                quarantine.append({"source_record_id": source_record_id, "ticker": ticker, "reason": "invalid_published"})
                continue
            acquisition_mode, selection_point_in_time = _selection_lineage(
                record,
                source_provider,
                published,
            )
            if source_provider == "google_news_rss" and record.get("query_window"):
                # Historical Google RSS searches commonly expose a synthetic date-level
                # timestamp. Treating 08:00 GMT as an intraday availability time leaks
                # precision that the archive does not actually guarantee.
                precision = "date"
            published_date = published.tz_convert("Asia/Seoul").normalize().tz_localize(None)
            query_window = record.get("query_window")
            if isinstance(query_window, Mapping):
                query_start = pd.to_datetime(query_window.get("start"), errors="coerce")
                query_end = pd.to_datetime(query_window.get("end"), errors="coerce")
                if pd.isna(query_start) or pd.isna(query_end) or query_start >= query_end:
                    raw_counts["invalid_query_window"] += 1
                    quarantine.append(
                        {
                            "source_record_id": source_record_id,
                            "ticker": ticker,
                            "reason": "invalid_query_window",
                        }
                    )
                    continue
                if (
                    published_date < pd.Timestamp(query_start).normalize()
                    or published_date >= pd.Timestamp(query_end).normalize()
                ):
                    raw_counts["outside_query_window"] += 1
                    quarantine.append(
                        {
                            "source_record_id": source_record_id,
                            "ticker": ticker,
                            "reason": "outside_query_window",
                        }
                    )
                    continue
            if published_date < release_start or published_date > release_end:
                raw_counts["outside_release_window"] += 1
                quarantine.append({"source_record_id": source_record_id, "ticker": ticker, "reason": "outside_release_window"})
                continue
            listing = pd.to_datetime(universe[ticker].get("listing_date"), errors="coerce")
            delisting = pd.to_datetime(universe[ticker].get("delisting_date"), errors="coerce")
            if (
                (not pd.isna(listing) and published_date < pd.Timestamp(listing).normalize())
                or (not pd.isna(delisting) and published_date > pd.Timestamp(delisting).normalize())
            ):
                raw_counts["outside_security_lifecycle"] += 1
                quarantine.append(
                    {"source_record_id": source_record_id, "ticker": ticker, "reason": "outside_security_lifecycle"}
                )
                continue
            title = clean_text(payload.get("title") or record.get("title"))
            source = clean_text(payload.get("source") or record.get("publisher"))
            url = canonical_url(payload.get("link") or payload.get("url") or record.get("url"))
            raw_summary = clean_text(
                payload.get("body")
                or payload.get("content")
                or payload.get("summary")
                or record.get("summary")
            )
            summary = _usable_summary(
                raw_summary,
                title,
                source,
            )
            if not title:
                raw_counts["missing_title"] += 1
                quarantine.append({"source_record_id": source_record_id, "ticker": ticker, "reason": "missing_title"})
                continue
            article_id = _article_key(
                title,
                source,
                str(published_date.date()),
                url,
                source_provider=source_provider,
            )
            availability_date = published_date
            if acquisition_mode == "live_capture":
                collected = pd.to_datetime(
                    record.get("collected_at_utc") or record.get("ts"),
                    errors="coerce",
                    utc=True,
                )
                if not pd.isna(collected):
                    collected_date = (
                        pd.Timestamp(collected)
                        .tz_convert("Asia/Seoul")
                        .normalize()
                        .tz_localize(None)
                    )
                    availability_date = max(published_date, collected_date)
            effective_session = _next_session(calendar, availability_date)
            if effective_session is None:
                raw_counts["not_yet_effective"] += 1
            candidate = {
                "schema_version": 1,
                "article_id": article_id,
                "title": title,
                "summary": summary,
                "source": source,
                "url": url,
                "source_provider": source_provider,
                "published_at_raw": str(published_raw),
                "published_at_utc": (
                    published.astimezone(timezone.utc).isoformat() if precision == "datetime" else None
                ),
                "published_date_kst": str(published_date.date()),
                "published_precision": precision,
                "effective_session": effective_session,
                "availability_policy": "next_krx_session",
                "content_tier": (
                    "official_filing"
                    if source_provider == "opendart"
                    else ("title_summary" if summary else "title_only")
                ),
                "acquisition_mode": acquisition_mode,
                "selection_point_in_time": selection_point_in_time,
            }
            existing = articles.get(article_id)
            if existing is None or _candidate_rank(candidate) > _candidate_rank(existing):
                articles[article_id] = candidate
            if selection_point_in_time:
                existing_pit = pit_articles.get(article_id)
                if existing_pit is None or _candidate_rank(candidate) > _candidate_rank(
                    existing_pit
                ):
                    pit_articles[article_id] = candidate
            if source_record_id:
                article_source_ids[article_id].add(source_record_id)
            article_acquisition_modes[article_id][acquisition_mode] += 1
            if selection_point_in_time:
                article_pit_occurrences[article_id] += 1
                article_pit_acquisition_modes[article_id][acquisition_mode] += 1
            else:
                article_non_pit_occurrences[article_id] += 1
            occurrence_id = hashlib.sha256(
                f"news-occurrence-v1|{raw_path}|{row_number}|{source_record_id}".encode("utf-8")
            ).hexdigest()
            occurrences.append(
                {
                    "schema_version": 1,
                    "occurrence_id": occurrence_id,
                    "article_id": article_id,
                    "ticker": ticker,
                    "source_record_id": source_record_id,
                    "source_provider": source_provider,
                    "source_path": str(raw_path),
                    "source_row_number": row_number,
                    "query_window": record.get("query_window"),
                    "acquisition": record.get("acquisition"),
                    "collected_at_raw": record.get("collected_at_utc") or record.get("ts"),
                    "published_at_raw": str(published_raw),
                    "acquisition_mode": acquisition_mode,
                    "selection_point_in_time": selection_point_in_time,
                }
            )
            mapping_key = (article_id, ticker)
            if source_provider == "opendart":
                mapping_method, mapping_confidence = "official_corp_code", 1.0
                matched_alias, matched_alias_type, matched_alias_source = None, None, None
                matched_alias_ambiguous = False
            else:
                (
                    mapping_method,
                    mapping_confidence,
                    matched_alias,
                    matched_alias_type,
                    matched_alias_source,
                    matched_alias_ambiguous,
                ) = _mapping_evidence(
                    ticker,
                    str(universe[ticker].get("name") or ""),
                    title,
                    raw_summary,
                    aliases_by_ticker.get(ticker, ()),
                    published_date,
                    source,
                )
            mapping_candidate = {
                "schema_version": 1,
                "article_id": article_id,
                "ticker": ticker,
                "company_name": str(universe[ticker].get("name") or ""),
                "mapping_method": mapping_method,
                "mapping_confidence": mapping_confidence,
                "matched_alias": matched_alias,
                "matched_alias_type": matched_alias_type,
                "matched_alias_source": matched_alias_source,
                "matched_alias_ambiguous": matched_alias_ambiguous,
                "requires_relevance_classification": source_provider != "opendart",
            }
            if mapping_key in mappings:
                raw_counts["duplicate_mapping"] += 1
                if _mapping_rank(mapping_method) > _mapping_rank(str(mappings[mapping_key]["mapping_method"])):
                    mappings[mapping_key] = mapping_candidate
            else:
                mappings[mapping_key] = mapping_candidate
            if selection_point_in_time:
                existing_pit_mapping = pit_mappings.get(mapping_key)
                if (
                    existing_pit_mapping is None
                    or _mapping_rank(mapping_method)
                    > _mapping_rank(str(existing_pit_mapping["mapping_method"]))
                ):
                    pit_mappings[mapping_key] = mapping_candidate

    article_rows: list[dict[str, Any]] = []
    for article_id, default_article in articles.items():
        article = pit_articles.get(article_id, default_article)
        acquisition_modes = (
            article_pit_acquisition_modes[article_id]
            if article_pit_occurrences[article_id] > 0
            else article_acquisition_modes[article_id]
        )
        article_rows.append(
            {
                **article,
                "source_record_ids": sorted(article_source_ids[article_id]),
                "acquisition_modes": dict(sorted(acquisition_modes.items())),
                "selection_point_in_time": article_pit_occurrences[article_id] > 0,
                "pit_occurrence_count": int(article_pit_occurrences[article_id]),
                "retrospective_or_unknown_occurrence_count": int(
                    article_non_pit_occurrences[article_id]
                ),
            }
        )
    article_rows, cluster_rows = cluster_news_articles(article_rows)
    curated_articles = {str(row["article_id"]): row for row in article_rows}
    mappings.update(pit_mappings)
    mapping_rows = sorted(mappings.values(), key=lambda row: (str(row["article_id"]), str(row["ticker"])))
    occurrence_rows = sorted(
        occurrences,
        key=lambda row: (str(row["article_id"]), str(row["ticker"]), str(row["occurrence_id"])),
    )

    queue_rows: list[dict[str, Any]] = []
    neutral_event_groups: defaultdict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    cluster_ticker_articles: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for mapping in mapping_rows:
        article = curated_articles[str(mapping["article_id"])]
        neutral_policy = (
            dart_neutral_family(article["title"])
            if article.get("source_provider") == "opendart"
            else None
        )
        if neutral_policy is not None and article["effective_session"] is not None:
            family, event_type = neutral_policy
            neutral_event_groups[
                (str(mapping["ticker"]), str(article["effective_session"]), family, event_type)
            ].append(str(mapping["article_id"]))
            continue
        cluster_ticker_articles[(str(article["event_cluster_id"]), str(mapping["ticker"]))].append(
            str(mapping["article_id"])
        )
    for (event_cluster_id, ticker), candidate_article_ids in sorted(cluster_ticker_articles.items()):
        unique_candidate_article_ids = sorted(set(candidate_article_ids))
        pit_candidate_article_ids = [
            value
            for value in unique_candidate_article_ids
            if bool(curated_articles[value].get("selection_point_in_time"))
        ]
        representative_candidate_ids = (
            pit_candidate_article_ids
            if pit_candidate_article_ids
            else unique_candidate_article_ids
        )
        article_id = max(
            representative_candidate_ids,
            key=lambda value: (
                _mapping_rank(str(mappings[(value, ticker)]["mapping_method"])),
                _candidate_rank(curated_articles[value]),
                value,
            ),
        )
        article = curated_articles[article_id]
        selected_mapping = mappings[(article_id, ticker)]
        cluster_acquisition_modes: Counter[str] = Counter()
        for candidate_article_id in representative_candidate_ids:
            cluster_acquisition_modes.update(
                {
                    str(mode): int(count)
                    for mode, count in dict(
                        curated_articles[candidate_article_id].get("acquisition_modes") or {}
                    ).items()
                }
            )
        cluster_selection_point_in_time = bool(pit_candidate_article_ids)
        if article["effective_session"] is None:
            continue
        queue_id = news_queue_id(event_cluster_id, ticker)
        company_name = str(universe[ticker].get("name") or "")
        queue_row = {
            "schema_version": 4,
            "queue_id": queue_id,
            "article_id": article_id,
            "event_cluster_id": event_cluster_id,
            "cluster_size": (
                len(representative_candidate_ids)
                if cluster_selection_point_in_time
                else int(article["cluster_size"])
            ),
            "ticker": ticker,
            "company_name": company_name,
            "mapping_method": selected_mapping["mapping_method"],
            "mapping_confidence": selected_mapping["mapping_confidence"],
            "matched_alias": selected_mapping.get("matched_alias"),
            "matched_alias_type": selected_mapping.get("matched_alias_type"),
            "matched_alias_source": selected_mapping.get("matched_alias_source"),
            "matched_alias_ambiguous": bool(
                selected_mapping.get("matched_alias_ambiguous", False)
            ),
            "title": article["title"],
            "summary": article["summary"],
            "related_titles": [],
            "source": article["source"],
            "published_date_kst": article["published_date_kst"],
            "published_precision": article["published_precision"],
            "effective_session": article["effective_session"],
            "content_tier": article["content_tier"],
            "acquisition_modes": dict(sorted(cluster_acquisition_modes.items())),
            "selection_point_in_time": cluster_selection_point_in_time,
            "queue_identity_policy": BASE_NEWS_QUEUE_ID_POLICY,
            "input_hash_policy": NEWS_QUEUE_INPUT_HASH_POLICY,
        }
        queue_row["input_sha256"] = news_queue_input_sha256(queue_row)
        queue_rows.append(queue_row)

    neutral_event_rows: list[dict[str, Any]] = []
    for (ticker, effective_session, family, event_type), article_ids in sorted(neutral_event_groups.items()):
        unique_article_ids = sorted(set(article_ids))
        representative = curated_articles[unique_article_ids[0]]
        neutral_acquisition_modes: Counter[str] = Counter()
        for source_article_id in unique_article_ids:
            neutral_acquisition_modes.update(
                {
                    str(mode): int(count)
                    for mode, count in dict(
                        curated_articles[source_article_id].get("acquisition_modes") or {}
                    ).items()
                }
            )
        neutral_selection_point_in_time = all(
            bool(curated_articles[source_article_id].get("selection_point_in_time"))
            for source_article_id in unique_article_ids
        )
        neutral_id = hashlib.sha256(
            f"dart-neutral-v1|{ticker}|{effective_session}|{family}".encode("utf-8")
        ).hexdigest()
        neutral_event_rows.append(
            {
                "schema_version": 1,
                "queue_id": neutral_id,
                "ticker": ticker,
                "published": representative["published_date_kst"],
                "effective_session": effective_session,
                "source": "opendart_neutral_count_policy_v1",
                "source_article_ids": unique_article_ids,
                "source_article_count": len(unique_article_ids),
                "content_tier": "official_filing",
                "acquisition_modes": dict(sorted(neutral_acquisition_modes.items())),
                "selection_point_in_time": neutral_selection_point_in_time,
                "lineage": {
                    "method": "deterministic_standardized_filing_policy",
                    "policy_version": "opendart-neutral-v1",
                    "llm_used": False,
                },
                "event": {
                    "event_type": event_type,
                    "summary": f"{representative['title']} ({len(unique_article_ids)}건)",
                    "relevance": 1.0,
                    "event_specificity": 1.0,
                    "sensor_accepted": True,
                    "polarity": 0.0,
                    "magnitude": 0.0,
                    "confidence": 1.0,
                    "evidence_quality": 1.0,
                    "content_tier": "official_filing",
                    "mapping_method": "official_corp_code",
                    "acquisition_modes": dict(sorted(neutral_acquisition_modes.items())),
                    "selection_point_in_time": neutral_selection_point_in_time,
                    "horizon_days": 1,
                    "affected_nodes": [ticker],
                    "themes": [f"dart:{event_type}", family],
                    "node_deltas": [],
                    "edge_deltas": [],
                },
            }
        )

    coverage_rows: list[dict[str, Any]] = []
    coverage_tickers: set[str] = set()
    for raw_path in config.get("coverage_paths", []):
        path = repo_root / str(raw_path)
        rows, load_report = load_jsonl(path)
        input_files.append({"path": str(raw_path), "sha256": sha256_file(path), "rows": load_report["rows"]})
        if load_report["invalid_json"] or load_report["invalid_rows"]:
            raise ValueError(f"invalid coverage JSONL input: {path}")
        for row in rows:
            ticker = normalize_ticker(row.get("ticker"))
            if ticker not in universe:
                continue
            start = pd.to_datetime(row.get("start"), errors="coerce")
            end = pd.to_datetime(row.get("end"), errors="coerce")
            if pd.isna(start) or pd.isna(end) or start > release_start or end < release_end:
                continue
            if int(row.get("request_errors", 0) or 0) > 0:
                continue
            if int(row.get("saturated_leaf_windows", 0) or 0) > 0:
                continue
            if str(row.get("status") or "complete") != "complete":
                continue
            coverage_rows.append({**row, "ticker": ticker, "status": "complete"})
            coverage_tickers.add(ticker)
    coverage_rows.sort(key=lambda row: (str(row.get("ticker")), str(row.get("source"))))

    coverage_window_rows: list[dict[str, Any]] = []
    coverage_window_ids: set[str] = set()
    for raw_path in config.get("window_coverage_paths", []):
        path = repo_root / str(raw_path)
        rows, load_report = load_jsonl(path)
        input_files.append({"path": str(raw_path), "sha256": sha256_file(path), "rows": load_report["rows"]})
        if load_report["invalid_json"] or load_report["invalid_rows"]:
            raise ValueError(f"invalid window coverage JSONL input: {path}")
        for row in rows:
            window_id = str(row.get("window_id") or "").strip()
            ticker = normalize_ticker(row.get("ticker"))
            start = pd.to_datetime(row.get("start"), errors="coerce")
            end_exclusive = pd.to_datetime(row.get("end_exclusive"), errors="coerce")
            status = str(row.get("status") or "")
            valid = (
                bool(window_id)
                and window_id not in coverage_window_ids
                and ticker in universe
                and not pd.isna(start)
                and not pd.isna(end_exclusive)
                and pd.Timestamp(start).normalize() >= release_start
                and pd.Timestamp(end_exclusive).normalize() <= release_end + pd.Timedelta(days=1)
                and pd.Timestamp(start).normalize() < pd.Timestamp(end_exclusive).normalize()
                and status in {"complete", "incomplete_saturated"}
            )
            if not valid:
                raise ValueError(f"invalid or duplicate window coverage row: {window_id or '<missing>'}")
            coverage_window_ids.add(window_id)
            coverage_window_rows.append({**row, "ticker": ticker})
    coverage_window_rows.sort(
        key=lambda row: (
            str(row.get("provider")),
            str(row.get("ticker")),
            str(row.get("start")),
            str(row.get("end_exclusive")),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "source_occurrences.jsonl", occurrence_rows)
    _write_jsonl(output_dir / "articles.jsonl", article_rows)
    _write_jsonl(output_dir / "event_clusters.jsonl", cluster_rows)
    _write_jsonl(output_dir / "article_ticker_mappings.jsonl", mapping_rows)
    _write_jsonl(output_dir / "structure_queue.jsonl", queue_rows)
    _write_jsonl(output_dir / "neutral_events.jsonl", neutral_event_rows)
    _write_jsonl(output_dir / "coverage.jsonl", coverage_rows)
    _write_jsonl(output_dir / "coverage_windows.jsonl", coverage_window_rows)
    _write_jsonl(output_dir / "quarantine.jsonl", quarantine)
    report = {
        "schema_version": 1,
        "source_universe_tickers": len(full_universe),
        "universe_tickers": len(universe),
        "policy": {
            "deduplication": "normalized_title+publisher+published_date",
            "event_clustering": "same-day canonical URL or conservative title similarity",
            "availability": "next_krx_session",
            "universe": "frozen point-in-time manifest with security lifecycle filter",
            "source_completeness": "search-index sample; no exhaustive archive claim",
            "historical_selection": "retrospective discovery searches are explicitly non-PIT",
            "relevance": "retain query-only mappings for explicit negative classification",
            "company_aliases": "only explicitly reviewed, unambiguous aliases with typed relationships",
            "standardized_filings": "directionless DART forms become grouped neutral count sensors",
        },
        "input_files": input_files,
        "load_reports": load_reports,
        "raw_counts": dict(raw_counts),
        "articles": len(article_rows),
        "source_occurrences": len(occurrence_rows),
        "event_clusters": len(cluster_rows),
        "duplicate_cluster_articles": sum(max(0, int(row["article_count"]) - 1) for row in cluster_rows),
        "mappings": len(mapping_rows),
        "mapping_evidence": dict(Counter(str(row["mapping_method"]) for row in mapping_rows)),
        "alias_registry_rows": sum(len(rows) for rows in aliases_by_ticker.values()),
        "alias_registry_tickers": len(aliases_by_ticker),
        "queue_rows": len(queue_rows),
        "neutral_event_rows": len(neutral_event_rows),
        "neutral_source_articles": sum(int(row["source_article_count"]) for row in neutral_event_rows),
        "queue_tickers": len({row["ticker"] for row in queue_rows}),
        "title_only_rows": sum(row["content_tier"] == "title_only" for row in article_rows),
        "point_in_time_selection_rows": sum(
            bool(row.get("selection_point_in_time")) for row in article_rows
        ),
        "retrospective_or_unknown_selection_rows": sum(
            not bool(row.get("selection_point_in_time")) for row in article_rows
        ),
        "date_only_rows": sum(row["published_precision"] == "date" for row in article_rows),
        "coverage_rows": len(coverage_rows),
        "coverage_tickers": len(coverage_tickers),
        "coverage_window_rows": len(coverage_window_rows),
        "complete_coverage_window_rows": sum(
            row.get("status") == "complete" for row in coverage_window_rows
        ),
        "saturated_coverage_window_rows": sum(
            row.get("status") == "incomplete_saturated" for row in coverage_window_rows
        ),
        "quarantine_rows": len(quarantine),
    }
    for filename in (
        "source_occurrences.jsonl",
        "articles.jsonl",
        "event_clusters.jsonl",
        "article_ticker_mappings.jsonl",
        "structure_queue.jsonl",
        "neutral_events.jsonl",
        "coverage.jsonl",
        "coverage_windows.jsonl",
        "quarantine.jsonl",
    ):
        report.setdefault("output_files", {})[filename] = {
            "sha256": sha256_file(output_dir / filename),
            "size_bytes": (output_dir / filename).stat().st_size,
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
