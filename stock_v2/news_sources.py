from __future__ import annotations

import re
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup


_NAVER_SEARCH_URL = "https://search.naver.com/search.naver"
_NAVER_SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://search.naver.com/",
}
_NAVER_SESSION = requests.Session()
_NAVER_INTERNAL_HOSTS = {
    "media.naver.com",
    "n.news.naver.com",
    "keep.naver.com",
    "search.naver.com",
    "www.naver.com",
}
_NAVER_DATE_RE = re.compile(r"\b(20\d{2})\.(\d{2})\.(\d{2})\.")


def _clean_naver_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for marker in ("새 창 열림", "네이버뉴스", "Keep에 저장", "Keep에 바로가기"):
        text = text.replace(marker, " ")
    return re.sub(r"\s+", " ", text).strip()


def _is_external_naver_article_link(href: str) -> bool:
    parsed = urllib.parse.urlparse(href)
    host = parsed.netloc.lower()
    return (
        parsed.scheme in {"http", "https"}
        and host not in _NAVER_INTERNAL_HOSTS
        and host != "naver.com"
        and not host.endswith(".naver.com")
    )


def _looks_like_article_url(href: str) -> bool:
    parsed = urllib.parse.urlparse(href)
    return parsed.path not in {"", "/"} or bool(parsed.query)


def parse_naver_search_results(
    html: str,
    limit: int,
    relative_date: pd.Timestamp | None = None,
) -> list[dict[str, str]]:
    """Extract the primary article from each public Naver news result cluster."""

    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".fds-news-item-list-tab")
    if root is None:
        return []

    rows: list[dict[str, str]] = []
    for item in root.find_all("div", recursive=False):
        if "sds-comps-vertical-layout" not in item.get("class", []):
            continue
        candidates: list[tuple[str, str]] = []
        for anchor in item.select("a[href]"):
            href = str(anchor.get("href") or "").strip()
            text = _clean_naver_text(anchor.get_text(" ", strip=True))
            if (
                _is_external_naver_article_link(href)
                and _looks_like_article_url(href)
                and len(text) >= 4
            ):
                candidates.append((href, text))
        if not candidates:
            continue

        link = candidates[0][0]
        texts = [text for href, text in candidates if href == link]
        title = texts[0]
        summary = next((text for text in texts[1:] if text != title), "")
        item_text = item.get_text(" ", strip=True)
        date_match = _NAVER_DATE_RE.search(item_text)
        if date_match is not None:
            published = "-".join(date_match.groups())
        elif relative_date is not None and re.search(r"(?:\d+\s*(?:분|시간)\s*전|방금\s*전|어제)", item_text):
            observed = pd.Timestamp(relative_date).normalize()
            if "어제" in item_text:
                observed -= pd.Timedelta(days=1)
            published = observed.strftime("%Y-%m-%d")
        else:
            continue

        source = ""
        for anchor in item.select("a[href]"):
            href = str(anchor.get("href") or "")
            if urllib.parse.urlparse(href).netloc.lower() == "media.naver.com":
                source = _clean_naver_text(anchor.get_text(" ", strip=True))
                if source:
                    break
        if not source:
            for anchor in item.select("a[href]"):
                href = str(anchor.get("href") or "")
                if _is_external_naver_article_link(href) and not _looks_like_article_url(href):
                    source = _clean_naver_text(anchor.get_text(" ", strip=True))
                    if source:
                        break

        rows.append(
            {
                "title": title,
                "summary": summary,
                "link": link,
                "published": published,
                "source": source,
            }
        )
        if len(rows) >= max(0, int(limit)):
            break
    return rows


def fetch_naver_search(
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int,
    timeout_sec: float,
    sort: str = "0",
    start_index: int = 1,
) -> list[dict[str, str]]:
    """Fetch date-bounded Korean news from Naver's public search results."""

    if sort not in {"0", "1"}:
        raise ValueError("sort must be '0' (relevance) or '1' (newest)")
    if start_index < 1:
        raise ValueError("start_index must be positive")
    params = {
        "where": "news",
        "query": query,
        "sm": "tab_opt",
        "sort": sort,
        "photo": "0",
        "field": "0",
        "pd": "3",
        "ds": start.strftime("%Y.%m.%d"),
        "de": (end - pd.Timedelta(days=1)).strftime("%Y.%m.%d"),
        "start": str(int(start_index)),
    }
    response = _NAVER_SESSION.get(
        _NAVER_SEARCH_URL,
        params=params,
        timeout=max(float(timeout_sec), 0.1),
        headers=_NAVER_SEARCH_HEADERS,
    )
    response.raise_for_status()
    return parse_naver_search_results(
        response.text,
        limit=limit,
        relative_date=(end - pd.Timedelta(days=1)).normalize(),
    )
