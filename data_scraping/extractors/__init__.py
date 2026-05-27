"""Book content extractors — turn raw book bytes into a stream of paragraphs.

Every extractor implements :class:`BookExtractor.extract(bytes) → Iterator[BookPassage]`.
Use :func:`pick_extractor` to get one by Chitanka's ``format_label``
(``epub`` / ``fb2`` / ``txt`` / ``pdf``).
"""

from .base import BookExtractor, BookPassage
from .factory import pick_extractor

__all__ = ["BookExtractor", "BookPassage", "pick_extractor"]
