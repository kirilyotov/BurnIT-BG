"""Project Gutenberg source.

Gutenberg's OPDS *search* endpoint returns navigation entries plus book
entries that do **not** carry acquisition links directly — each book has its
own ``/ebooks/{id}.opds`` page that lists the actual files. We resolve those
on the fly so each yielded :class:`BookEntry` already has its
acquisition links populated.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Iterator, List, Tuple
from urllib.parse import urljoin

import feedparser

from ..http import USER_AGENT, request_with_retries
from .base import AcquisitionLink, BookEntry, BookSource

BASE_URL = "https://www.gutenberg.org"

TOPICS: Dict[str, str] = {
    "psychology": "/ebooks/search.opds/?query=psychology&sort_order=downloads",
    "self-improvement": "/ebooks/search.opds/?query=self+improvement&sort_order=downloads",
    "mental-health": "/ebooks/search.opds/?query=mental+health&sort_order=downloads",
    "philosophy-of-mind": "/ebooks/search.opds/?query=philosophy+of+mind&sort_order=downloads",
}

MIME_FORMATS: Dict[str, Tuple[str, str]] = {
    "application/epub+zip": ("epub", ".epub"),
    "text/plain": ("txt", ".txt"),
    "text/plain; charset=utf-8": ("txt", ".txt"),
    "text/html": ("html", ".html"),
    "application/x-mobipocket-ebook": ("mobi", ".mobi"),
    "application/pdf": ("pdf", ".pdf"),
}

BOOK_ID_RE = re.compile(r"/ebooks/(\d+)\.opds")


def absolutize(href: str) -> str:
    if not href:
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(BASE_URL, href)
    return href


def _fetch_feed(url: str, delay: float = 0.0) -> feedparser.FeedParserDict:
    resp = request_with_retries(
        url,
        headers={"Accept": "application/atom+xml,application/xml,text/xml"},
        retries=5,
        delay=max(delay, 1.0),
    )
    if delay > 0:
        time.sleep(delay)
    return feedparser.parse(resp.content)


def _book_links_from_feed(feed: feedparser.FeedParserDict) -> List[AcquisitionLink]:
    """Each book's OPDS page contains multiple ``<entry>`` blocks (with /
    without images). We collate all acquisition links across them and prefer
    no-images / smaller variants implicitly via the link title hint."""
    links: List[AcquisitionLink] = []
    seen: set = set()
    for entry in feed.entries:
        for link in entry.get("links", []):
            rel = link.get("rel") or ""
            if "acquisition" not in rel:
                continue
            mime = link.get("type") or ""
            href = link.get("href")
            if not href or mime not in MIME_FORMATS:
                continue
            key = (mime, href)
            if key in seen:
                continue
            seen.add(key)
            fmt, ext = MIME_FORMATS[mime]
            links.append(
                AcquisitionLink(
                    url=absolutize(href),
                    mime=mime,
                    format_label=fmt,
                    ext=ext,
                    title=link.get("title", ""),
                )
            )

    # When we have multiple epub variants, prefer the smaller / no-images one.
    def _link_rank(a: AcquisitionLink) -> int:
        title_lc = (a.title or "").lower()
        url_lc = a.url.lower()
        if "no images" in title_lc or "noimages" in url_lc or ".epub.noimages" in url_lc:
            return 0
        return 1

    links.sort(key=lambda l: (l.format_label, _link_rank(l)))
    return links


def _subjects(feed: feedparser.FeedParserDict) -> List[str]:
    out: List[str] = []
    for entry in feed.entries:
        for tag in entry.get("tags", []) or []:
            term = tag.get("term")
            if term and term not in out:
                out.append(term)
    return out


def _author(entry) -> str:
    if entry.get("authors"):
        name = entry["authors"][0].get("name")
        if name:
            return name
    content = entry.get("content") or entry.get("summary") or ""
    if isinstance(content, list):
        content = content[0].get("value", "") if content else ""
    return (content or "").strip()


class GutenbergSource(BookSource):
    name = "project_gutenberg"
    default_formats = ("epub", "txt", "html", "mobi", "pdf")

    def category_choices(self) -> List[str]:
        return list(TOPICS.keys())

    def iter_books(
        self,
        category: str,
        limit: int = 20,
        delay: float = 1.0,
    ) -> Iterator[BookEntry]:
        if category not in TOPICS:
            raise ValueError(
                f"Unknown gutenberg topic '{category}'. "
                f"Choose one of: {', '.join(sorted(TOPICS))}"
            )
        search_url = absolutize(TOPICS[category])
        # Walk paginated search results until we have enough book entries.
        yielded = 0
        next_url = search_url
        while next_url and yielded < limit:
            feed = _fetch_feed(next_url, delay=delay)
            for entry in feed.entries:
                if yielded >= limit:
                    break
                entry_id = entry.get("id", "") or ""
                m = BOOK_ID_RE.search(entry_id)
                if not m:
                    continue  # navigation row (subjects / bookshelves)
                book_id = m.group(1)
                book_opds = f"{BASE_URL}/ebooks/{book_id}.opds"
                try:
                    book_feed = _fetch_feed(book_opds, delay=delay)
                except Exception:
                    continue
                links = _book_links_from_feed(book_feed)
                if not links:
                    continue
                subjects = _subjects(book_feed)
                yield BookEntry(
                    source=self.name,
                    book_id=book_id,
                    entry_id=entry_id,
                    title=(entry.get("title") or "").strip(),
                    authors=[_author(entry)] if _author(entry) else [],
                    language="en",
                    summary="",
                    categories=[category] + subjects,
                    catalog_url=search_url,
                    page_url=f"{BASE_URL}/ebooks/{book_id}",
                    published=entry.get("published") or entry.get("updated"),
                    acquisition_links=links,
                )
                yielded += 1

            next_link = None
            for link in feed.feed.get("links", []):
                if link.get("rel") == "next":
                    next_link = absolutize(link.get("href"))
                    break
            next_url = next_link
