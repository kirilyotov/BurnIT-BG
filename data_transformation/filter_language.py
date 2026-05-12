"""Drop books that don't match a target language.

Gutenberg's search results are not always English — psychology queries
return a small number of German / French / Dutch / Greek titles. This
module:

1. Reads the DuckDB ``manifest`` rows for the chosen source.
2. For each row, streams the file out of MinIO into memory, extracts a
   short sample of text (EPUB → ebooklib, plain text → first ~32 KB).
3. Runs :mod:`langdetect` on the sample.
4. Records the detection in a result list. If ``apply=True``, deletes
   non-matching books from MinIO and removes their rows from DuckDB.
5. Optionally re-uploads the refreshed ``books.duckdb`` + parquet
   snapshot so the published metadata stays in sync.
"""

from __future__ import annotations

import io
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import duckdb
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # make detection deterministic for identical inputs

log = logging.getLogger(__name__)
SAMPLE_BYTES = 64 * 1024  # max text we extract per book before language detection


@dataclass
class FilterResult:
    """Outcome of inspecting one manifest row.

    ``storage_uri`` is exactly the value the pipeline wrote to the DuckDB
    ``local_path`` column — an ``s3://bucket/key`` URI for the MinIO backend
    or an absolute filesystem path for the local backend.
    """

    book_id: str
    title: str
    storage_uri: str
    download_format: str
    detected_language: Optional[str]
    keep: bool
    note: str = ""


def _extract_epub_text(blob: bytes, max_chars: int = SAMPLE_BYTES) -> str:
    """Pull a chunk of plain text from an EPUB without writing to disk."""
    chunks: List[str] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            try:
                raw = zf.read(name)
            except KeyError:
                continue
            text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
            if text:
                chunks.append(text)
            if sum(len(c) for c in chunks) >= max_chars:
                break
    return " ".join(chunks)[:max_chars]


def _extract_text(blob: bytes, fmt: str) -> str:
    fmt = (fmt or "").lower()
    if fmt == "epub":
        try:
            return _extract_epub_text(blob)
        except zipfile.BadZipFile:
            return ""
    if fmt in {"txt", "html"}:
        try:
            text = blob.decode("utf-8", errors="replace")
        except Exception:
            return ""
        if fmt == "html":
            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        return text[:SAMPLE_BYTES]
    return ""


def _detect_language(text: str) -> Optional[str]:
    text = (text or "").strip()
    if len(text) < 64:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def _open_duck_for_write(duckdb_path: Path) -> duckdb.DuckDBPyConnection:
    """Open the manifest for modification, raising a helpful error on lock."""
    try:
        return duckdb.connect(str(duckdb_path))
    except duckdb.IOException as exc:
        raise RuntimeError(
            f"Cannot open DuckDB for write at {duckdb_path}: {exc}. "
            "If you have `duckdb -ui` open on this file, close it first."
        ) from exc


def evaluate(
    storage,
    duckdb_path: Path,
    source_name: str,
    keep_lang: str = "en",
    limit: Optional[int] = None,
) -> List[FilterResult]:
    """Detect language for every book of *source_name* in the manifest.

    Returns one :class:`FilterResult` per book (no mutations).
    """
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB store not found at {duckdb_path}")

    # Read-only from a copy so we don't fight any UI process holding the file.
    tmp_copy = duckdb_path.with_suffix(duckdb_path.suffix + ".read")
    shutil.copy2(duckdb_path, tmp_copy)
    con = duckdb.connect(str(tmp_copy), read_only=True)
    try:
        sql = (
            "SELECT book_id, title, local_path, download_format "
            "FROM manifest "
            "WHERE source = ? "
            "ORDER BY book_id"
        )
        rows = con.execute(sql, [source_name]).fetchall()
    finally:
        con.close()
        tmp_copy.unlink(missing_ok=True)

    if limit:
        rows = rows[:limit]
    results: List[FilterResult] = []
    for book_id, title, local_path, fmt in rows:
        if not local_path:
            results.append(FilterResult(book_id, title, "", fmt or "", None, True,
                                        "row has no storage uri; keeping"))
            continue
        try:
            blob = storage.load_bytes(local_path)
        except Exception as exc:
            results.append(FilterResult(book_id, title, local_path, fmt or "", None, True,
                                        f"could not fetch ({exc}); keeping"))
            continue
        text = _extract_text(blob, fmt)
        lang = _detect_language(text)
        keep = lang == keep_lang if lang is not None else True
        note = "" if lang is not None else "language could not be determined; keeping"
        results.append(FilterResult(book_id, title, local_path, fmt or "", lang, keep, note))
    return results


def apply_removals(
    storage,
    duckdb_path: Path,
    source_name: str,
    to_remove: Iterable[FilterResult],
) -> Tuple[int, int]:
    """Delete chosen books from the storage backend and DuckDB.

    Returns ``(storage_deletes, duckdb_deletes)``.
    """
    to_remove_list = [r for r in to_remove if r.book_id]
    storage_deletes = 0
    duckdb_deletes = 0

    for r in to_remove_list:
        if not r.storage_uri:
            continue
        try:
            storage.remove_object(r.storage_uri)
            storage_deletes += 1
        except Exception as exc:
            log.warning("failed to delete '%s': %s", r.storage_uri, exc)

    book_ids = [r.book_id for r in to_remove_list]
    if book_ids:
        con = _open_duck_for_write(duckdb_path)
        try:
            for chunk_start in range(0, len(book_ids), 500):
                chunk = book_ids[chunk_start:chunk_start + 500]
                placeholders = ",".join(["?"] * len(chunk))
                params = [source_name] + list(chunk)
                con.execute(
                    f"DELETE FROM manifest WHERE source = ? AND book_id IN ({placeholders})",
                    params,
                )
                duckdb_deletes += len(chunk)
        finally:
            con.close()
    return storage_deletes, duckdb_deletes


def reupload_manifest(
    storage,
    duckdb_path: Path,
    tmp_dir: Path,
    remote_prefix: str,
    date: str,
) -> None:
    """Refresh ``raw/metadata/books.{duckdb,parquet}`` after a mutation."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    read_copy = tmp_dir / "books.read.duckdb"
    shutil.copy2(duckdb_path, read_copy)
    parquet_path = tmp_dir / "books.parquet"
    con = duckdb.connect(str(read_copy), read_only=True)
    try:
        con.execute(
            "COPY manifest TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(parquet_path)],
        )
    finally:
        con.close()

    prefix = remote_prefix.rstrip("/")
    targets = [
        (duckdb_path, f"{prefix}/metadata/books.duckdb"),
        (duckdb_path, f"{prefix}/metadata/books-{date}.duckdb"),
        (parquet_path, f"{prefix}/metadata/books.parquet"),
        (parquet_path, f"{prefix}/metadata/books-{date}.parquet"),
    ]
    for local, remote in targets:
        uri = storage.save_file(local, remote)
        log.info("uploaded %s -> %s", local.name, uri)
