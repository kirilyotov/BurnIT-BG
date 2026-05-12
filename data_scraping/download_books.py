"""Source-agnostic book download pipeline.

Given a :class:`BookSource`, a category/topic, a list of preferred formats
and a :class:`PipelineConfig`, :func:`run_pipeline` will:

1. Iterate :class:`BookEntry` objects from the source.
2. Pick the best acquisition link in the requested format priority.
3. Stream the file into a temp directory.
4. Upload it to the configured backend at
   ``{remote_prefix}/{source}/{date}/{category}/{format}/{book_id}{ext}``.
5. Record the result in DuckDB and append to an in-memory manifest list.
6. Upload the per-source ``manifest.jsonl`` to
   ``{remote_prefix}/{source}/{date}/metadata/manifest.jsonl``.

The actual CLI lives in :mod:`data_scraping.cli`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from .duckdb_store import DuckDBStore
from .http import sha256_of, stream_to_file
from .sources import pick_acquisition
from .sources.base import BookEntry, BookSource
from .storage_backend import StorageBackend


@dataclass
class PipelineConfig:
    """Parameters that control where files land and how aggressive we are."""

    storage: StorageBackend
    bucket: str
    remote_prefix: str = "raw"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    formats: Sequence[str] = ("epub", "fb2", "txt")
    delay: float = 1.0
    per_category: int = 5
    limit: int = 25
    tmp_dir: Path = field(default_factory=lambda: Path("Local/_tmp"))
    duckdb_path: Optional[Path] = None
    verbose: bool = True


@dataclass
class DownloadResult:
    book: BookEntry
    remote_key: str
    remote_uri: str
    sha256: str
    size_bytes: int
    format_label: str
    mime: str


def _log(cfg: PipelineConfig, msg: str) -> None:
    if cfg.verbose:
        print(msg, flush=True)


def _open_duck(cfg: PipelineConfig):
    """Open the DuckDB manifest store, or return None if the file is locked.

    The pipeline can still complete using the in-memory manifest + JSONL upload;
    the user just won't get an updated DuckDB row this run. Common cause is
    another process (often ``duckdb -ui``) holding an exclusive lock.
    """
    if cfg.duckdb_path is None:
        return None
    try:
        return DuckDBStore(cfg.duckdb_path)
    except Exception as exc:
        _log(cfg, f"[warn] DuckDB '{cfg.duckdb_path}' unavailable ({exc}); continuing with JSONL only")
        return None


def _record(result: DownloadResult, cfg: PipelineConfig) -> dict:
    return {
        "source": result.book.source,
        "entry_id": result.book.entry_id,
        "book_id": result.book.book_id,
        "title": result.book.title,
        "authors": result.book.authors,
        "language": result.book.language,
        "summary": result.book.summary,
        "categories": result.book.categories,
        "catalog_url": result.book.catalog_url,
        "book_page_url": result.book.page_url,
        "download_url": next(
            (l.url for l in result.book.acquisition_links if l.format_label == result.format_label),
            "",
        ),
        "download_format": result.format_label,
        "download_mime_type": result.mime,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "local_path": result.remote_uri,
        "sha256": result.sha256,
        "file_size_bytes": result.size_bytes,
        "published": result.book.published,
        "minio_bucket": cfg.bucket,
        "minio_key": result.remote_key,
    }


def download_one(
    book: BookEntry,
    category: str,
    cfg: PipelineConfig,
) -> Optional[DownloadResult]:
    """Download a single book in the best matching format. Returns None on skip."""
    chosen = pick_acquisition(book.acquisition_links, cfg.formats)
    if not chosen:
        _log(cfg, f"  [skip] {book.book_id} '{book.title}': no acquisition link for formats={cfg.formats}")
        return None
    local_path = cfg.tmp_dir / book.source / category / chosen.format_label / f"{book.book_id}{chosen.ext}"
    if local_path.exists():
        sha = sha256_of(local_path)
        size = local_path.stat().st_size
    else:
        try:
            sha, size = stream_to_file(chosen.url, local_path, delay=cfg.delay)
        except Exception as exc:
            _log(cfg, f"  [fail] {book.book_id} '{book.title}': {exc}")
            return None
    remote_key = (
        f"{cfg.remote_prefix.rstrip('/')}/"
        f"{book.source}/{cfg.date}/{category}/{chosen.format_label}/{book.book_id}{chosen.ext}"
    )
    remote_uri = cfg.storage.save_file(local_path, remote_key)
    _log(cfg, f"  [ok] {book.book_id} '{book.title[:60]}' -> {remote_uri} ({size} B)")
    return DownloadResult(
        book=book,
        remote_key=remote_key,
        remote_uri=remote_uri,
        sha256=sha,
        size_bytes=size,
        format_label=chosen.format_label,
        mime=chosen.mime,
    )


def run_pipeline(
    source: BookSource,
    categories: Sequence[str],
    cfg: PipelineConfig,
) -> List[dict]:
    """Drive the source through *categories* and return manifest records."""
    cfg.storage.ensure_ready()
    cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
    duck = _open_duck(cfg)
    manifest: List[dict] = []
    total = 0
    try:
        for category in categories:
            if total >= cfg.limit:
                break
            remaining = cfg.limit - total
            take = min(cfg.per_category, remaining)
            _log(cfg, f"\n=== {source.name} :: {category} (take={take}, delay={cfg.delay}s) ===")
            try:
                for book in source.iter_books(category=category, limit=take, delay=cfg.delay):
                    try:
                        result = download_one(book, category, cfg)
                    except Exception as exc:
                        _log(cfg, f"  [error] {book.book_id} '{book.title}': {exc}")
                        continue
                    if result is None:
                        continue
                    record = _record(result, cfg)
                    manifest.append(record)
                    if duck is not None:
                        try:
                            duck.upsert_manifest_record(record)
                        except Exception as exc:
                            _log(cfg, f"  [warn] duckdb upsert failed for {book.book_id}: {exc}")
                    total += 1
                    if total >= cfg.limit:
                        break
            except Exception as exc:
                _log(cfg, f"[error] {source.name} :: {category} aborted ({exc}); moving on")
                continue
    finally:
        if duck is not None:
            duck.close()

    if manifest:
        _write_and_upload_manifest(manifest, source.name, cfg)
    _log(cfg, f"\n{source.name} done. Downloaded {total} books on {cfg.date}.")
    return manifest


def _write_and_upload_manifest(records: List[dict], source_name: str, cfg: PipelineConfig) -> None:
    manifest_path = cfg.tmp_dir / source_name / "metadata" / f"manifest-{cfg.date}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    remote_key = (
        f"{cfg.remote_prefix.rstrip('/')}/"
        f"{source_name}/{cfg.date}/metadata/manifest.jsonl"
    )
    uri = cfg.storage.save_file(manifest_path, remote_key)
    _log(cfg, f"Uploaded manifest -> {uri}")
