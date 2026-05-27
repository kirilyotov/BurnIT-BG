"""EPUB → paragraph stream via ``ebooklib`` + BeautifulSoup."""

from __future__ import annotations

import io
from typing import Iterator

from .base import BookExtractor, BookPassage
from ._clean import looks_like_noise, normalize_paragraph


class EpubExtractor(BookExtractor):
    """Walk every HTML document in an EPUB, yield ``<p>`` paragraphs."""
    name = "epub"

    def __init__(self, min_chars: int = 40) -> None:
        self.min_chars = min_chars

    def extract(self, raw_bytes: bytes) -> Iterator[BookPassage]:
        # Local imports — keep extractors importable without the heavy deps.
        from ebooklib import epub, ITEM_DOCUMENT  # type: ignore[import-not-found]
        from bs4 import BeautifulSoup            # type: ignore[import-not-found]

        # ebooklib only takes file paths; tee through an in-memory buffer.
        book = epub.read_epub(io.BytesIO(raw_bytes))

        paragraph_index = 0
        char_offset = 0
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            try:
                content = item.get_body_content().decode("utf-8", errors="ignore")
            except Exception:
                continue
            soup = BeautifulSoup(content, "html.parser")
            chapter_title = None
            heading = soup.find(["h1", "h2", "h3"])
            if heading and heading.get_text(strip=True):
                chapter_title = normalize_paragraph(heading.get_text(" ", strip=True))[:140]

            for elem in soup.find_all(["p", "div"]):
                # Skip nested duplicates — only emit leaf-ish paragraphs.
                if elem.find(["p", "div"]):
                    continue
                raw_text = elem.get_text(" ", strip=True)
                text = normalize_paragraph(raw_text)
                if looks_like_noise(text, min_chars=self.min_chars):
                    continue
                yield BookPassage(
                    text=text,
                    paragraph_index=paragraph_index,
                    char_offset=char_offset,
                    chapter_title=chapter_title,
                )
                paragraph_index += 1
                char_offset += len(text) + 1
