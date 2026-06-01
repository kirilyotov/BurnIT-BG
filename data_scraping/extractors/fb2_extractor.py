"""FictionBook (.fb2 / .fb2.zip) → paragraph stream."""

import io
import xml.etree.ElementTree as ET
import zipfile
from typing import Iterator

from .base import BookExtractor, BookPassage
from ._clean import looks_like_noise, normalize_paragraph


# FB2 documents declare default namespaces — we strip them on the fly so
# XPath/find() can use plain tag names.
def _strip_namespaces(elem: ET.Element) -> None:
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


class Fb2Extractor(BookExtractor):
    """Walk ``<section>/<p>`` elements in a FB2 book; auto-handles ``.fb2.zip``."""
    name = "fb2"

    def __init__(self, min_chars: int = 40) -> None:
        self.min_chars = min_chars

    def _maybe_unzip(self, raw_bytes: bytes) -> bytes:
        if raw_bytes[:4] != b"PK\x03\x04":
            return raw_bytes
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            fb2_members = [n for n in zf.namelist() if n.lower().endswith(".fb2")]
            if not fb2_members and zf.namelist():
                fb2_members = [zf.namelist()[0]]
            if not fb2_members:
                return raw_bytes
            return zf.read(fb2_members[0])

    def extract(self, raw_bytes: bytes) -> Iterator[BookPassage]:
        xml_bytes = self._maybe_unzip(raw_bytes)
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            # Some FB2 files have a BOM or weird preamble; try a more
            # forgiving decode.
            text = xml_bytes.decode("utf-8", errors="ignore")
            root = ET.fromstring(text)
        _strip_namespaces(root)

        paragraph_index = 0
        char_offset = 0
        for body in root.findall("body"):
            for section in body.iter("section"):
                title_el = section.find("title")
                chapter_title = None
                if title_el is not None:
                    title_text = " ".join((t.text or "") for t in title_el.iter()).strip()
                    if title_text:
                        chapter_title = normalize_paragraph(title_text)[:140]
                for p in section.iter("p"):
                    # Concatenate all descendant text — preserves italics/emphasis content.
                    raw_text = "".join(p.itertext())
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
