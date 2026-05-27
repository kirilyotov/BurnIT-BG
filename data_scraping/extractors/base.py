"""Shared types for the book content extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class BookPassage:
    """A single paragraph-sized snippet extracted from a book.

    ``paragraph_index`` is monotonic per book (starts at 0). ``char_offset``
    is the cumulative character count of *all preceding paragraphs joined*
    — useful for locating the passage inside the full text later.
    """
    text: str
    paragraph_index: int
    char_offset: int
    chapter_title: str | None = None

    @property
    def length_chars(self) -> int:
        return len(self.text)


class BookExtractor(ABC):
    """Stream a book's paragraphs as :class:`BookPassage` objects."""

    name: str = "abstract"

    @abstractmethod
    def extract(self, raw_bytes: bytes) -> Iterator[BookPassage]:
        """Yield paragraphs as they're read from the in-memory book."""
        raise NotImplementedError
