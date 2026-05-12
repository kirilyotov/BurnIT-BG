"""Free-tier translation helper.

Thin wrapper over :class:`deep_translator.GoogleTranslator` that:

* Chunks long strings to fit the provider's per-request size cap.
* Caches translations on disk (JSON file) so a re-run of the same dataset
  is mostly free.
* Retries transient errors with exponential backoff.

The CLI command in :mod:`data_transformation.cli` drives this against any
file format supported by :mod:`data_transformation.io_utils`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    NotValidLength,
    NotValidPayload,
    RequestError,
    TranslationNotFound,
)

log = logging.getLogger(__name__)

# GoogleTranslator's hard limit is 5000 chars per request; we leave headroom.
MAX_CHUNK_CHARS = 4500
PARAGRAPH_SPLIT = re.compile(r"\n{2,}")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TranslatorConfig:
    source: str = "en"
    target: str = "bg"
    cache_path: Optional[Path] = None
    delay: float = 0.0
    retries: int = 4
    backoff: float = 1.5


class Translator:
    """Stateful translator with on-disk cache."""

    def __init__(self, cfg: TranslatorConfig) -> None:
        self.cfg = cfg
        self._engine = GoogleTranslator(source=cfg.source, target=cfg.target)
        self._cache: dict[str, str] = {}
        if cfg.cache_path and Path(cfg.cache_path).exists():
            try:
                self._cache = json.loads(Path(cfg.cache_path).read_text(encoding="utf-8"))
            except Exception:
                log.warning("Failed to read translation cache at %s; starting empty", cfg.cache_path)
                self._cache = {}

    def _cache_key(self, text: str) -> str:
        return f"{self.cfg.source}|{self.cfg.target}|{text}"

    def _flush_cache(self) -> None:
        if not self.cfg.cache_path:
            return
        path = Path(self.cfg.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    def translate(self, text: str) -> str:
        """Translate a single string, using cache + chunking when needed."""
        if not text or not text.strip():
            return text
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        if len(text) <= MAX_CHUNK_CHARS:
            translated = self._translate_one(text)
        else:
            chunks = _chunk_text(text, MAX_CHUNK_CHARS)
            translated_chunks = []
            for chunk in chunks:
                ckey = self._cache_key(chunk)
                if ckey in self._cache:
                    translated_chunks.append(self._cache[ckey])
                    continue
                translated_chunk = self._translate_one(chunk)
                self._cache[ckey] = translated_chunk
                translated_chunks.append(translated_chunk)
            translated = "\n\n".join(translated_chunks)
        self._cache[key] = translated
        return translated

    def translate_many(
        self,
        texts: Iterable[str],
        on_each: Optional[Callable[[int, str, str], None]] = None,
    ) -> List[str]:
        out: List[str] = []
        for idx, text in enumerate(texts):
            try:
                translated = self.translate(text)
            except Exception as exc:
                log.warning("translation failed for record %d: %s", idx, exc)
                translated = text
            out.append(translated)
            if on_each is not None:
                on_each(idx, text, translated)
            if self.cfg.delay > 0:
                time.sleep(self.cfg.delay)
            if idx % 25 == 24:
                self._flush_cache()
        self._flush_cache()
        return out

    def _translate_one(self, text: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.retries):
            try:
                result = self._engine.translate(text)
                if result is None:
                    return text
                return result
            except NotValidLength:
                # Chunk is still too long after splitting — keep original.
                log.warning(
                    "skipping oversized text (%d chars > Google's 5000 char limit)",
                    len(text),
                )
                return text
            except (TranslationNotFound, NotValidPayload):
                return text
            except RequestError as exc:
                last_exc = exc
                wait = self.cfg.backoff * (2 ** attempt) + 0.5
                log.warning("translator request error, retrying in %.1fs: %s", wait, exc)
                time.sleep(wait)
            except Exception as exc:
                last_exc = exc
                wait = self.cfg.backoff * (2 ** attempt) + 0.5
                log.warning("translator error, retrying in %.1fs: %s", wait, exc)
                time.sleep(wait)
        if last_exc is not None:
            raise last_exc
        return text


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """Split *text* into chunks <= ``max_chars`` along paragraph / sentence breaks.

    Falls back to a hard character split when a single sentence is itself
    longer than ``max_chars`` (otherwise Google rejects the request with
    ``NotValidLength``).
    """
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    for paragraph in PARAGRAPH_SPLIT.split(text):
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        for sent in SENTENCE_SPLIT.split(paragraph):
            if len(sent) > max_chars:
                # Sentence on its own exceeds the limit — hard-split it.
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i + max_chars])
                continue
            if not chunks or len(chunks[-1]) + 1 + len(sent) > max_chars:
                chunks.append(sent)
            else:
                chunks[-1] = chunks[-1] + " " + sent
    return [c for c in chunks if c.strip()]
