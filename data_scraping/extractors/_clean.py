"""Shared cleanup helpers used by every extractor.

Paragraph cleanup is intentionally minimal — just enough to make
keyword-matching robust and not produce garbage in the dataset.
"""

from __future__ import annotations

import re
import unicodedata

# Collapses runs of any kind of whitespace (incl. NBSP, IDEOGRAPHIC SPACE).
_WS_RE = re.compile(r"\s+", re.UNICODE)

# Lines that are just page numbers / chapter labels — common in PDFs.
_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")


def normalize_paragraph(text: str) -> str:
    """Strip control chars, normalize Unicode, collapse whitespace."""
    if not text:
        return ""
    # NFC normalization gives us canonical Bulgarian glyphs (й / Й composed).
    text = unicodedata.normalize("NFC", text)
    # Drop control chars and zero-width spaces.
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
    # Collapse whitespace.
    text = _WS_RE.sub(" ", text).strip()
    return text


def looks_like_noise(text: str, *, min_chars: int = 40) -> bool:
    """Return True for paragraphs we should skip (page numbers, copyright lines, etc.)."""
    if not text:
        return True
    if len(text) < min_chars:
        return True
    if _NUMBER_LINE.match(text):
        return True
    # Lines that are 80%+ digits/punctuation aren't prose.
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.5:
        return True
    return False
