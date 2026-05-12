"""Catalog source implementations for the book downloader pipeline."""

from .base import AcquisitionLink, BookEntry, BookSource, pick_acquisition
from .chitanka import ChitankaSource
from .gutenberg import GutenbergSource
from .opds import OPDSSource

SOURCES = {
    ChitankaSource.name: ChitankaSource,
    GutenbergSource.name: GutenbergSource,
}

__all__ = [
    "AcquisitionLink",
    "BookEntry",
    "BookSource",
    "ChitankaSource",
    "GutenbergSource",
    "OPDSSource",
    "SOURCES",
    "pick_acquisition",
]
