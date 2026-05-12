"""Shared types for book sources.

A *source* is a catalog (chitanka, gutenberg, …) the downloader can walk.
Each source yields :class:`BookEntry` records and the pipeline in
``data_scraping.download_books`` is source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional


@dataclass
class AcquisitionLink:
    """One downloadable representation of a book (epub, txt, fb2, …)."""

    url: str
    mime: str
    format_label: str   # canonical short tag: 'epub', 'txt', 'fb2'
    ext: str            # filename suffix, e.g. '.epub', '.fb2.zip'
    title: str = ""


@dataclass
class BookEntry:
    """Normalised metadata for one book, regardless of source."""

    source: str
    book_id: str
    entry_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    language: str = ""
    summary: str = ""
    categories: List[str] = field(default_factory=list)
    catalog_url: str = ""
    page_url: str = ""
    published: Optional[str] = None
    acquisition_links: List[AcquisitionLink] = field(default_factory=list)


class BookSource:
    """Abstract base for catalog sources."""

    name: str = ""
    default_formats: tuple = ("epub", "txt", "fb2")

    def category_choices(self) -> List[str]:
        """Return the list of category/topic names this source supports."""
        raise NotImplementedError

    def iter_books(
        self,
        category: str,
        limit: int = 20,
        delay: float = 1.0,
    ) -> Iterator[BookEntry]:
        """Yield :class:`BookEntry` records for *category*, up to *limit*."""
        raise NotImplementedError


def pick_acquisition(
    links: Iterable[AcquisitionLink],
    preferred: Iterable[str],
) -> Optional[AcquisitionLink]:
    """Return the first link whose ``format_label`` matches *preferred* order."""
    by_fmt = {}
    for link in links:
        by_fmt.setdefault(link.format_label, link)
    for fmt in preferred:
        if fmt in by_fmt:
            return by_fmt[fmt]
    return None
