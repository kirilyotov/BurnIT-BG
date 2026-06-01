"""EPUB → paragraph stream via ``ebooklib`` + BeautifulSoup."""

import io
import logging
import zipfile
from typing import Iterator

from .base import BookExtractor, BookPassage
from ._clean import looks_like_noise, normalize_paragraph

log = logging.getLogger(__name__)


def _sniff_file_type(raw_bytes: bytes) -> str:
    """Guess file format from magic bytes."""
    head = raw_bytes[:20]
    if head.startswith(b"<?xml"): return "fb2"
    h = head.lower()
    if h.startswith(b"<!doctype") or h.startswith(b"<html"): return "html"
    if head.startswith(b"PK"): return "zip"
    try:
        raw_bytes[:200].decode("utf-8", errors="strict")
        return "txt"
    except UnicodeDecodeError:
        return "unknown"


class EpubExtractor(BookExtractor):
    """Walk every HTML document in an EPUB, yield ``<p>`` paragraphs."""
    name = "epub"

    def __init__(self, min_chars: int = 40) -> None:
        self.min_chars = min_chars

    def extract(self, raw_bytes: bytes) -> Iterator[BookPassage]:
        # Local imports — keep extractors importable without the heavy deps.
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup

        # ebooklib only takes file paths; tee through an in-memory buffer.
        try:
            book = epub.read_epub(io.BytesIO(raw_bytes))
        except (zipfile.BadZipFile, zipfile.LargeZipFile, KeyError, OSError, Exception) as exc:
            log.warning(
                "[epub] %s: %s — sniffed=%s — skipping",
                type(exc).__name__, str(exc)[:80], _sniff_file_type(raw_bytes),
            )
            return

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
