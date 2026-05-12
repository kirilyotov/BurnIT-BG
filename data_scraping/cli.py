"""Command-line interface for the book ingestion pipeline.

Usage:
    python -m data_scraping chitanka [options]
    python -m data_scraping gutenberg [options]
    python -m data_scraping upload-metadata [options]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb

from .download_books import PipelineConfig, run_pipeline
from .sources import SOURCES, ChitankaSource, GutenbergSource
from .storage_backend import StorageBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUCKDB = REPO_ROOT / "tmp" / "data_scraping" / "books.duckdb"
DEFAULT_TMP = REPO_ROOT / "tmp" / "data_scraping" / "_tmp"


def _load_env() -> None:
    """Load the repo-level .env so MinIO credentials are available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def _build_storage(args: argparse.Namespace) -> StorageBackend:
    if args.backend == "minio":
        endpoint = args.minio_endpoint or os.getenv("MINIO_ENDPOINT")
        access_key = args.minio_access_key or os.getenv("MINIO_ACCESS_KEY")
        secret_key = args.minio_secret_key or os.getenv("MINIO_SECRET_KEY")
        bucket = args.bucket or os.getenv("MINIO_BUCKET", "data")
        if not (endpoint and access_key and secret_key):
            raise SystemExit("MinIO credentials missing. Set MINIO_* in .env or pass flags.")
        secure_env = os.getenv("MINIO_FORCE_SECURE", "false").lower() == "true"
        secure = args.secure if args.secure is not None else secure_env
        return StorageBackend.minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            secure=secure,
        )
    return StorageBackend(backend=args.backend, bucket=args.bucket)


def _common_args(p: argparse.ArgumentParser, default_delay: float, default_formats: str) -> None:
    p.add_argument("--limit", type=int, default=25, help="Total cap across all categories")
    p.add_argument("--per-category", type=int, default=5, help="Books per category/topic")
    p.add_argument("--delay", type=float, default=default_delay, help="Inter-request delay (s)")
    p.add_argument(
        "--formats",
        default=default_formats,
        #before specifing a format check the source's avaiable formats
        help="Comma-separated preferred formats, in priority order (e.g. epub,fb2,txt)",
    )
    p.add_argument("--backend", default="minio", choices=["minio", "local", "huggingface"])
    p.add_argument("--bucket", default=None, help="Bucket name (defaults to env MINIO_BUCKET=data)")
    p.add_argument("--remote-prefix", default="raw", help="Object key prefix (default: 'raw')")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--duckdb", default=str(DEFAULT_DUCKDB))
    p.add_argument("--tmp-dir", default=str(DEFAULT_TMP))
    p.add_argument("--minio-endpoint", default=None)
    p.add_argument("--minio-access-key", default=None)
    p.add_argument("--minio-secret-key", default=None)
    p.add_argument(
        "--secure",
        default=None,
        type=lambda v: v.lower() in ("1", "true", "yes"),
        help="Force TLS on/off for MinIO (defaults to MINIO_FORCE_SECURE env var)",
    )


def _pipeline_config(args: argparse.Namespace, storage: StorageBackend) -> PipelineConfig:
    return PipelineConfig(
        storage=storage,
        bucket=args.bucket or os.getenv("MINIO_BUCKET", "data"),
        remote_prefix=args.remote_prefix,
        date=args.date,
        formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
        delay=args.delay,
        per_category=args.per_category,
        limit=args.limit,
        tmp_dir=Path(args.tmp_dir),
        duckdb_path=Path(args.duckdb) if args.duckdb else None,
    )


def _resolve_categories(source_name: str, requested: Optional[list[str]]) -> list[str]:
    source_cls = SOURCES[source_name]
    available = source_cls().category_choices()
    if not requested:
        return available
    bad = [c for c in requested if c not in available]
    if bad:
        raise SystemExit(
            f"Unknown {source_name} categories: {bad}. Available: {available}"
        )
    return requested


def _cmd_source(args: argparse.Namespace, source_name: str) -> int:
    _load_env()
    storage = _build_storage(args)
    cfg = _pipeline_config(args, storage)
    source = SOURCES[source_name]()
    categories = _resolve_categories(source_name, args.categories)
    run_pipeline(source, categories=categories, cfg=cfg)
    return 0


def _cmd_upload_metadata(args: argparse.Namespace) -> int:
    _load_env()
    duckdb_path = Path(args.duckdb)
    if not duckdb_path.exists():
        raise SystemExit(f"DuckDB file not found: {duckdb_path}")
    storage = _build_storage(args)
    storage.ensure_ready()

    # Work from a temp copy so we don't fight a running `duckdb -ui` process
    # that has an exclusive lock on the live file.
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    read_copy = tmp_dir / "books.read.duckdb"
    import shutil
    shutil.copy2(duckdb_path, read_copy)

    parquet_path = tmp_dir / "books.parquet"
    con = duckdb.connect(str(read_copy), read_only=True)
    try:
        n = con.execute("SELECT count(*) FROM manifest").fetchone()[0]
        print(f"manifest rows: {n}")
        con.execute(
            "COPY manifest TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(parquet_path)],
        )
    finally:
        con.close()

    prefix = args.remote_prefix.rstrip("/")
    date = args.date
    targets = [
        (duckdb_path, f"{prefix}/metadata/books.duckdb"),
        (duckdb_path, f"{prefix}/metadata/books-{date}.duckdb"),
        (parquet_path, f"{prefix}/metadata/books.parquet"),
        (parquet_path, f"{prefix}/metadata/books-{date}.parquet"),
    ]
    for local, remote in targets:
        uri = storage.save_file(local, remote)
        print(f"uploaded {local.name} -> {uri}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data_scraping", description="Book ingestion pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ch = sub.add_parser("chitanka", help="Download books from chitanka.info OPDS catalog")
    _common_args(p_ch, default_delay=3.0, default_formats="epub,fb2,txt")
    p_ch.add_argument(
        "--categories",
        nargs="*",
        choices=ChitankaSource().category_choices(),
        default=None,
    )

    p_gu = sub.add_parser("gutenberg", help="Download books from Project Gutenberg OPDS catalog")
    _common_args(p_gu, default_delay=2.0, default_formats="epub,txt")
    p_gu.add_argument(
        "--categories",
        nargs="*",
        choices=GutenbergSource().category_choices(),
        default=None,
        help="Topics (Gutenberg search queries) to walk",
    )

    p_um = sub.add_parser("upload-metadata", help="Upload books.duckdb (+ parquet) to MinIO")
    p_um.add_argument("--duckdb", default=str(DEFAULT_DUCKDB))
    p_um.add_argument("--tmp-dir", default=str(DEFAULT_TMP))
    p_um.add_argument("--backend", default="minio", choices=["minio", "local", "huggingface"])
    p_um.add_argument("--bucket", default=None)
    p_um.add_argument("--remote-prefix", default="raw")
    p_um.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    p_um.add_argument("--minio-endpoint", default=None)
    p_um.add_argument("--minio-access-key", default=None)
    p_um.add_argument("--minio-secret-key", default=None)
    p_um.add_argument(
        "--secure",
        default=None,
        type=lambda v: v.lower() in ("1", "true", "yes"),
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "chitanka":
        return _cmd_source(args, "chitanka")
    if args.command == "gutenberg":
        return _cmd_source(args, "project_gutenberg")
    if args.command == "upload-metadata":
        return _cmd_upload_metadata(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
