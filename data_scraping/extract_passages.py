"""Extract mental-health passages from previously-downloaded books.

Pipeline (per book):

1. Iterate the manifest rows in DuckDB (filtered by date / source).
2. Read the original book bytes back from MinIO (via
   :meth:`StorageBackend.load_bytes`).
3. Pick an extractor (epub / fb2 / txt / pdf) based on the book's
   ``download_format``.
4. Stream paragraphs, score each against
   :mod:`data_scraping.topics_mental_health`, keep only matches.
5. Append a record to ``tmp/data_scraping/extracted/passages-{date}.jsonl``
   and upsert into the DuckDB ``passages`` table.
6. After every book is done, upload the JSONL to MinIO at
   ``extracted/chitanka/{date}/passages.jsonl``.

The output JSONL is the input to
``data_transformation.build_dataset`` (next step).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import duckdb

from .duckdb_store import DuckDBStore
from .extractors import BookPassage, pick_extractor
from .storage_backend import StorageBackend
from .topics_mental_health import all_topics, match_passage

log = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Run-time settings for :func:`run_extraction`.

    ``books_date`` filters the manifest to only books downloaded on a
    specific day (matches the date segment in ``local_path``). Leave
    ``None`` to extract from **every** book in the manifest — that's
    the friendly default.

    ``extract_date`` is just the timestamp baked into the output JSONL
    filename and the MinIO destination prefix; defaults to today.
    """

    storage: StorageBackend
    bucket: str
    source: str = "chitanka"
    extract_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    books_date: str | None = None  # None → no manifest filter (extract from all books)
    topics: tuple[str, ...] = ()  # empty == all topics
    min_chars: int = 80
    max_chars: int = 1200
    per_book_limit: int | None = None
    overall_limit: int | None = None
    duckdb_path: Path = Path("tmp/data_scraping/books.duckdb")
    tmp_dir: Path = Path("tmp/data_scraping/extracted")
    remote_prefix: str = "extracted"
    output_filename: str = "passages"  # base filename (no extension, no date)
    book_id_filter: tuple[str, ...] = ()
    format_filter: tuple[str, ...] = ()
    verbose: bool = True


def _log(cfg: ExtractionConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg, flush=True)


def _select_manifest(cfg: ExtractionConfig) -> list[dict]:
    """Pull manifest rows matching the run's source/date/filters."""
    conn = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        where = ["source = ?"]
        params: list = [cfg.source]
        # Optional date filter — only applied when explicitly requested.
        if cfg.books_date:
            where.append("local_path LIKE ?")
            params.append(f"%/{cfg.books_date}/%")
        if cfg.book_id_filter:
            placeholders = ",".join("?" for _ in cfg.book_id_filter)
            where.append(f"book_id IN ({placeholders})")
            params.extend(cfg.book_id_filter)
        if cfg.format_filter:
            placeholders = ",".join("?" for _ in cfg.format_filter)
            where.append(f"download_format IN ({placeholders})")
            params.extend(cfg.format_filter)
        # Discover which columns the manifest actually has — older DBs
        # may not have minio_bucket/minio_key (those came in a later
        # schema). We pick the SELECT list dynamically based on what's
        # present so the same code works against any manifest version.
        available = {r[0] for r in conn.execute("PRAGMA table_info('manifest')").fetchall()}
        base_cols = [
            "source", "book_id", "title", "authors", "download_format",
            "download_mime_type", "local_path", "language",
        ]
        optional_cols = [c for c in ("minio_bucket", "minio_key") if c in available]
        select_cols = ", ".join(base_cols + optional_cols)
        sql = (
            f"SELECT {select_cols} FROM manifest "
            f"WHERE {' AND '.join(where)} ORDER BY book_id"
        )
        rows = conn.execute(sql, params).fetchall()
        cols = [d[0] for d in conn.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def _book_minio_key(row: dict) -> str:
    """Return the MinIO key for a manifest row (prefer ``minio_key``)."""
    return row.get("minio_key") or row.get("local_path") or ""


def _build_passage_record(
    passage: BookPassage,
    *,
    book_row: dict,
    topic: str,
    keywords: Iterable[str],
    extraction_date: str,
    source: str,
) -> dict:
    book_id = book_row.get("book_id", "?")
    passage_id = f"{source}-{book_id}-p{passage.paragraph_index:06d}"
    authors = book_row.get("authors")
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except json.JSONDecodeError:
            authors = [authors]
    return {
        "passage_id": passage_id,
        "source": source,
        "book_id": book_id,
        "book_title": book_row.get("title"),
        "authors": authors,
        "topic": topic,
        "keywords_matched": list(keywords),
        "text": passage.text,
        "language": book_row.get("language") or "bg",
        "paragraph_index": passage.paragraph_index,
        "char_offset": passage.char_offset,
        "length_chars": passage.length_chars,
        "chapter_title": passage.chapter_title,
        "book_minio_key": _book_minio_key(book_row),
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "extraction_date": extraction_date,
    }


def _extract_one_book(book_row: dict, cfg: ExtractionConfig) -> Iterator[dict]:
    """Yield passage records for a single book that match the run's topics."""
    fmt = (book_row.get("download_format") or "").lower()
    if not fmt:
        _log(cfg, f"  [skip] {book_row.get('book_id')}: no download_format")
        return
    try:
        extractor = pick_extractor(fmt)
    except ValueError as exc:
        _log(cfg, f"  [skip] {book_row.get('book_id')}: {exc}")
        return

    key = _book_minio_key(book_row)
    if not key:
        _log(cfg, f"  [skip] {book_row.get('book_id')}: no minio key in manifest")
        return
    try:
        raw_bytes = cfg.storage.load_bytes(key)
    except Exception as exc:
        _log(cfg, f"  [fail] {book_row.get('book_id')}: load_bytes failed ({exc})")
        return

    topic_filter = cfg.topics or all_topics()
    kept = 0
    try:
        for passage in extractor.extract(raw_bytes):
            if passage.length_chars < cfg.min_chars or passage.length_chars > cfg.max_chars:
                continue
            match = match_passage(passage.text, topics=topic_filter)
            if not match:
                continue
            topic, keywords = match
            yield _build_passage_record(
                passage,
                book_row=book_row,
                topic=topic,
                keywords=keywords,
                extraction_date=cfg.extract_date,
                source=cfg.source,
            )
            kept += 1
            if cfg.per_book_limit and kept >= cfg.per_book_limit:
                break
    except Exception as exc:
        _log(cfg, f"  [error] {book_row.get('book_id')}: extractor failed ({exc})")


def run_extraction(cfg: ExtractionConfig) -> tuple[Path, int]:
    """Drive the full pipeline and return ``(local_jsonl_path, count)``."""
    cfg.storage.ensure_ready()
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = cfg.tmp_dir / f"{cfg.output_filename}-{cfg.extract_date}.jsonl"

    books = _select_manifest(cfg)
    _log(cfg, f"[extract] manifest matched {len(books)} books "
              f"(source={cfg.source}, date={cfg.extract_date}, formats={cfg.format_filter or 'any'})")

    duck = None
    if cfg.duckdb_path:
        try:
            duck = DuckDBStore(cfg.duckdb_path)
        except Exception as exc:
            _log(cfg, f"[warn] DuckDB unavailable ({exc}); continuing with JSONL only")

    total = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in books:
            if cfg.overall_limit and total >= cfg.overall_limit:
                break
            _log(cfg, f"[book] {row.get('book_id')} '{(row.get('title') or '')[:60]}' "
                      f"({row.get('download_format')})")
            n_book = 0
            for record in _extract_one_book(row, cfg):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if duck is not None:
                    try:
                        duck.upsert_passage(record)
                    except Exception as exc:
                        _log(cfg, f"  [warn] duckdb upsert failed: {exc}")
                total += 1
                n_book += 1
                if cfg.overall_limit and total >= cfg.overall_limit:
                    break
            _log(cfg, f"  kept {n_book} passages")
    if duck is not None:
        duck.close()

    if total == 0:
        _log(cfg, "[extract] no passages extracted — nothing to upload")
        return output_path, 0

    # MinIO path: {prefix}/{source}/{date}/{filename}.jsonl — mirrors the local name.
    remote_key = (
        f"{cfg.remote_prefix.rstrip('/')}/{cfg.source}/{cfg.extract_date}/"
        f"{cfg.output_filename}.jsonl"
    )
    uri = cfg.storage.save_file(output_path, remote_key)
    _log(cfg, f"[extract] uploaded {total} passages -> {uri}")
    return output_path, total
