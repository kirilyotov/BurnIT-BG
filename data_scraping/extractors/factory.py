"""Map ``format_label`` strings to a concrete :class:`BookExtractor`."""

from .base import BookExtractor
from .epub_extractor import EpubExtractor
from .fb2_extractor import Fb2Extractor
from .pdf_extractor import PdfExtractor
from .txt_extractor import TxtExtractor


_REGISTRY: dict[str, type[BookExtractor]] = {
    "epub": EpubExtractor,
    "fb2": Fb2Extractor,
    "txt": TxtExtractor,
    "pdf": PdfExtractor,
}


def pick_extractor(format_label: str, **kwargs) -> BookExtractor:
    """Return a fresh extractor matching ``format_label`` (case-insensitive)."""
    key = (format_label or "").lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"No extractor for format '{format_label}'. "
            f"Supported: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)
