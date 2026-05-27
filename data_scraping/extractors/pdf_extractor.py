"""PDF → paragraph stream via ``pypdf``."""

from __future__ import annotations

import io
import re
from typing import Iterator

from .base import BookExtractor, BookPassage
from ._clean import looks_like_noise, normalize_paragraph


# PDF text-extraction often inserts hard line-breaks every ~80 chars.
# Re-join wrapped lines: a single newline that's NOT followed by an empty
# line is treated as a soft wrap.
_SOFT_WRAP = re.compile(r"(?<!\n)\n(?!\n)")


class PdfExtractor(BookExtractor):
    """Page-level extraction with paragraph reflow."""
    name = "pdf"

    def __init__(self, min_chars: int = 40) -> None:
        self.min_chars = min_chars

    def extract(self, raw_bytes: bytes) -> Iterator[BookPassage]:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(io.BytesIO(raw_bytes))
        paragraph_index = 0
        char_offset = 0
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            text = _SOFT_WRAP.sub(" ", text)
            for chunk in text.split("\n\n"):
                cleaned = normalize_paragraph(chunk)
                if looks_like_noise(cleaned, min_chars=self.min_chars):
                    continue
                yield BookPassage(
                    text=cleaned, paragraph_index=paragraph_index, char_offset=char_offset,
                )
                paragraph_index += 1
                char_offset += len(cleaned) + 1
