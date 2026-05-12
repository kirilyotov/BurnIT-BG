"""Chitanka.info OPDS source.

Wraps the generic :class:`OPDSSource` walker with chitanka-specific knowledge:
fixed category slugs (Bulgarian-language psychology / self-help / mental
health categories), MIME → format mapping, and URL absolutisation
(chitanka serves relative ``/book/{id}.epub`` hrefs).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple
from urllib.parse import urljoin

from .base import AcquisitionLink, BookEntry, BookSource
from .opds import OPDSSource

BASE_URL = "https://chitanka.info"

# Bulgarian categories on https://chitanka.info/books/category.opds that
# relate to psychology, self-improvement, mental health, well-being.
CATEGORIES: Dict[str, str] = {
    "psychology": "/books/category/psihologia.opds",
    "applied-psychology": "/books/category/prilozhna-psichologia.opds",
    "self-improvement": "/books/category/samousavarshenstvane-i-alternativno-poznanie.opds",
    "health-and-alt-medicine": "/books/category/zdrave-i-alternativna-medicina.opds",
    "self-help-manuals": "/books/category/rakovodstva-i-samouchiteli.opds",
}

# MIME → (canonical format label, filename extension)
MIME_FORMATS: Dict[str, Tuple[str, str]] = {
    "application/epub+zip": ("epub", ".epub"),
    "application/x-fictionbook+xml": ("fb2", ".fb2.zip"),
    "application/zip": ("txt", ".txt.zip"),
    "application/pdf": ("pdf", ".pdf")
}


def absolutize(href: str) -> str:
    if not href:
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin(BASE_URL, href)
    return href


def _entry_id_to_book_id(entry_id: str) -> str:
    # 'urn:x-chitanka:book:6023' -> '6023'
    return entry_id.rsplit(":", 1)[-1] if entry_id else "unknown"


class ChitankaSource(BookSource):
    name = "chitanka"
    default_formats = ("epub", "fb2", "txt", "pdf")

    def __init__(self) -> None:
        self._opds = OPDSSource()

    def category_choices(self) -> List[str]:
        return list(CATEGORIES.keys())

    def iter_books(
        self,
        category: str,
        limit: int = 20,
        delay: float = 1.0,
    ) -> Iterator[BookEntry]:
        if category not in CATEGORIES:
            raise ValueError(
                f"Unknown chitanka category '{category}'. "
                f"Choose one of: {', '.join(sorted(CATEGORIES))}"
            )
        feed_url = absolutize(CATEGORIES[category])
        for entry in self._opds.fetch_entries(feed_url, limit=limit, delay=delay):
            entry_id = entry.get("id") or ""
            book_id = _entry_id_to_book_id(entry_id)
            authors = [a.get("name") for a in entry.get("authors", []) if a.get("name")]
            if not authors and entry.get("author"):
                authors = [entry.get("author")]
            links: List[AcquisitionLink] = []
            for link in entry.get("links", []):
                rel = link.get("rel") or ""
                if "acquisition" not in rel:
                    continue
                mime = link.get("type") or ""
                href = link.get("href")
                if not href or mime not in MIME_FORMATS:
                    continue
                fmt, ext = MIME_FORMATS[mime]
                links.append(
                    AcquisitionLink(
                        url=absolutize(href),
                        mime=mime,
                        format_label=fmt,
                        ext=ext,
                    )
                )
            yield BookEntry(
                source=self.name,
                book_id=book_id,
                entry_id=entry_id,
                title=(entry.get("title") or "").strip(),
                authors=[a.strip() for a in authors if a],
                language=entry.get("dc_language") or entry.get("language") or "bg",
                summary=(entry.get("summary") or entry.get("description") or "").strip(),
                categories=[category],
                catalog_url=feed_url,
                page_url=absolutize(f"/book/{book_id}"),
                published=entry.get("dc_issued") or entry.get("updated"),
                acquisition_links=links,
            )
