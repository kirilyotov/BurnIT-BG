"""Plain ``.txt`` (or ``.txt.zip``) → paragraph stream."""

from __future__ import annotations

import io
import zipfile
from typing import Iterator

from .base import BookExtractor, BookPassage
from ._clean import looks_like_noise, normalize_paragraph


class TxtExtractor(BookExtractor):
    """Split on blank lines, treat each non-empty block as a paragraph.

    Auto-handles ``.txt.zip`` (the Chitanka convention) by reading the
    first ``.txt`` member of the archive.
    """
    name = "txt"

    def __init__(self, min_chars: int = 40, encodings: tuple[str, ...] = ("utf-8", "cp1251", "windows-1251")) -> None:
        self.min_chars = min_chars
        self.encodings = encodings

    def _maybe_unzip(self, raw_bytes: bytes) -> bytes:
        # ZIP files start with "PK\x03\x04".
        if raw_bytes[:4] != b"PK\x03\x04":
            return raw_bytes
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            txt_members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not txt_members:
                # Fall through to first member to be lenient.
                txt_members = zf.namelist()[:1]
            if not txt_members:
                return raw_bytes
            return zf.read(txt_members[0])

    def _decode(self, raw_bytes: bytes) -> str:
        for enc in self.encodings:
            try:
                return raw_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="ignore")

    def extract(self, raw_bytes: bytes) -> Iterator[BookPassage]:
        text = self._decode(self._maybe_unzip(raw_bytes))
        # Paragraphs separated by 1+ blank lines.
        paragraph_index = 0
        char_offset = 0
        buffer: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if line.strip():
                buffer.append(line)
                continue
            if not buffer:
                continue
            joined = normalize_paragraph(" ".join(buffer))
            buffer.clear()
            if looks_like_noise(joined, min_chars=self.min_chars):
                continue
            yield BookPassage(text=joined, paragraph_index=paragraph_index, char_offset=char_offset)
            paragraph_index += 1
            char_offset += len(joined) + 1
        if buffer:
            joined = normalize_paragraph(" ".join(buffer))
            if not looks_like_noise(joined, min_chars=self.min_chars):
                yield BookPassage(text=joined, paragraph_index=paragraph_index, char_offset=char_offset)
