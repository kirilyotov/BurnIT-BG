"""Command-line interface for the data transformation pipeline.

Usage:
    python -m data_transformation translate [options]
    python -m data_transformation filter-language [options]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi

from .filter_language import apply_removals, evaluate, reupload_manifest
from .hf_dataset import (
    dataset_slug,
    download_repo,
    translate_dataset,
    upload_to_hf_bucket,
    upload_to_minio,
)
from .kaggle_dataset import (
    dataset_slug as kaggle_dataset_slug,
    download_repo as kaggle_download_repo,
)
from .io_utils import RecordWriter, get_nested, read_records, set_nested
from .translate import Translator, TranslatorConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUCKDB = REPO_ROOT / "tmp" / "data_scraping" / "books.duckdb"
DEFAULT_TMP = REPO_ROOT / "tmp" / "data_transformation"
DEFAULT_CACHE = DEFAULT_TMP / "translate_cache.json"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def _build_storage(
    backend: str,
    bucket: Optional[str] = None,
    secure: Optional[bool] = None,
):
    """Build a StorageBackend. Mirrors ``data_scraping.cli._build_storage``.

    ``backend`` is one of ``"minio"``, ``"local"``, ``"huggingface"``. For
    MinIO, credentials come from CLI flags or the matching ``MINIO_*``
    environment variables.
    """
    from data_scraping.storage_backend import StorageBackend
    if backend == "minio":
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        bucket = bucket or os.getenv("MINIO_BUCKET", "data")
        if not (endpoint and access_key and secret_key):
            raise SystemExit("MinIO credentials missing. Set MINIO_* in .env.")
        if secure is None:
            secure = os.getenv("MINIO_FORCE_SECURE", "false").lower() == "true"
        return StorageBackend.minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            secure=secure,
        )
    return StorageBackend(backend=backend, bucket=bucket)


# ----- translate ------------------------------------------------------------


def _cmd_translate(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"input file not found: {in_path}")

    fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else []
    cfg = TranslatorConfig(
        source=args.source_lang,
        target=args.target_lang,
        cache_path=Path(args.cache),
        delay=args.delay,
    )
    translator = Translator(cfg)

    total = 0
    translated_strings = 0
    with RecordWriter(out_path) as writer:
        for record in read_records(in_path):
            total += 1
            target_paths: List[str] = fields if fields else _detect_string_keys(record)
            for path in target_paths:
                value = get_nested(record, path)
                if isinstance(value, str) and value.strip():
                    translated_value = translator.translate(value)
                    set_nested(record, path, translated_value)
                    translated_strings += 1
                elif isinstance(value, list):
                    new_list = []
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            new_list.append(translator.translate(item))
                            translated_strings += 1
                        else:
                            new_list.append(item)
                    set_nested(record, path, new_list)
            writer.write(record)
            if total % 25 == 0:
                print(f"  processed {total} records, {translated_strings} fields translated", flush=True)

    print(f"\ntranslate done: {total} records, {translated_strings} fields translated → {out_path}")
    return 0


def _detect_string_keys(record: dict) -> List[str]:
    """If no --fields specified, translate top-level string values."""
    return [k for k, v in record.items() if isinstance(v, str)]


# ----- filter-language ------------------------------------------------------


def _cmd_filter_language(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()
    storage = _build_storage(
        backend=args.backend,
        bucket=args.bucket,
        secure=args.secure,
    )
    duckdb_path = Path(args.duckdb)
    tmp_dir = Path(args.tmp_dir)

    print(f"\nEvaluating {args.source} books in DuckDB {duckdb_path}")
    results = evaluate(
        storage=storage,
        duckdb_path=duckdb_path,
        source_name=args.source,
        keep_lang=args.keep_lang,
        limit=args.limit,
    )

    keep = [r for r in results if r.keep]
    drop = [r for r in results if not r.keep]
    print(f"  total inspected: {len(results)}")
    print(f"  matched '{args.keep_lang}': {len(keep)}")
    print(f"  to remove (not '{args.keep_lang}'): {len(drop)}")
    print()
    if drop:
        print("Books that would be removed:")
        for r in drop:
            print(f"  {r.book_id:>8}  lang={r.detected_language or '?':<4}  fmt={r.download_format:<5}  {r.title[:70]}")
        print()

    if not drop:
        print("Nothing to remove.")
        return 0

    if not args.apply:
        print("Dry run — pass --apply to delete from the backend and DuckDB.")
        return 0

    print("Applying removals…")
    minio_n, duckdb_n = apply_removals(
        storage=storage,
        duckdb_path=duckdb_path,
        source_name=args.source,
        to_remove=drop,
    )
    print(f"  removed from storage: {minio_n}")
    print(f"  removed from DuckDB:  {duckdb_n}")

    if args.skip_metadata_upload:
        print("Skipping metadata re-upload (per --skip-metadata-upload).")
        return 0
    print("\nRe-uploading manifest snapshot to MinIO…")
    reupload_manifest(
        storage=storage,
        duckdb_path=duckdb_path,
        tmp_dir=tmp_dir,
        remote_prefix=args.remote_prefix,
        date=args.date,
    )
    print("Done.")
    return 0


# ----- hf-dataset -----------------------------------------------------------


def _cmd_hf_dataset(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()

    slug = dataset_slug(args.repo)
    work_dir = Path(args.tmp_dir) / "datasets" / slug
    raw_dir = work_dir / "raw"
    bg_dir = work_dir / args.target_lang
    fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else []

    if not args.skip_download:
        print(f"\n[1/4] Downloading {args.repo} → {raw_dir}")
        token = os.getenv("HF_TOKEN")
        download_repo(args.repo, work_dir, token=token)
    else:
        print(f"\n[1/4] Skipping download (raw expected at {raw_dir})")

    if not args.skip_translate:
        print(f"\n[2/4] Translating {args.source_lang} → {args.target_lang} (fields={fields or 'all string fields'})")
        cache_path = Path(args.cache)
        total = translate_dataset(
            raw_dir=raw_dir,
            out_dir=bg_dir,
            fields=fields,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            cache_path=cache_path,
            delay=args.delay,
        )
        print(f"  total records processed: {total}")
    else:
        print("\n[2/4] Skipping translation")

    minio_prefix_full = f"{args.minio_prefix.rstrip('/')}/{slug}"
    if not args.skip_minio:
        print(f"\n[3/4] Uploading to MinIO at s3://{args.bucket or os.getenv('MINIO_BUCKET', 'data')}/{minio_prefix_full}")
        storage = _build_storage(backend="minio", bucket=args.bucket, secure=args.secure)
        n_raw = upload_to_minio(storage, raw_dir, f"{minio_prefix_full}/raw") if raw_dir.exists() else 0
        n_bg = upload_to_minio(storage, bg_dir, f"{minio_prefix_full}/{args.target_lang}") if bg_dir.exists() else 0
        print(f"  uploaded {n_raw} raw + {n_bg} translated files")
    else:
        print("\n[3/4] Skipping MinIO upload")

    hf_prefix_full = f"{args.hf_prefix.rstrip('/')}/{slug}"
    if not args.skip_hf:
        api = HfApi(token=os.getenv("HF_TOKEN"))
        hf_bucket = args.hf_bucket
        # Auto-qualify bare bucket names ("data" → "kiplayo/data") using
        # the token's account, so the bash wrapper stays portable.
        if "/" not in hf_bucket:
            try:
                hf_bucket = f"{api.whoami()['name']}/{hf_bucket}"
            except Exception:
                pass
        print(f"\n[4/4] Uploading to HF bucket '{hf_bucket}' at '{hf_prefix_full}'")
        n_raw = upload_to_hf_bucket(api, hf_bucket, raw_dir, f"{hf_prefix_full}/raw") if raw_dir.exists() else 0
        n_bg = upload_to_hf_bucket(api, hf_bucket, bg_dir, f"{hf_prefix_full}/{args.target_lang}") if bg_dir.exists() else 0
        print(f"  uploaded {n_raw} raw + {n_bg} translated files")
    else:
        print("\n[4/4] Skipping HF bucket upload")

    print("\nDone.")
    return 0


# ----- kaggle-dataset -------------------------------------------------------


def _cmd_kaggle_dataset(args: argparse.Namespace) -> int:
    """End-to-end pipeline for a Kaggle dataset.

    Mirrors :func:`_cmd_hf_dataset` step-for-step:

    1. ``kagglehub`` snapshot → ``tmp/datasets/{slug}/raw/``
    2. translate selected fields → ``.../{target_lang}/``
    3. push raw + translated to MinIO
    4. push raw + translated to a HuggingFace bucket

    The handle is the standard Kaggle ``owner/dataset-name`` string.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()

    slug = kaggle_dataset_slug(args.handle)
    work_dir = Path(args.tmp_dir) / "datasets" / slug
    raw_dir = work_dir / "raw"
    bg_dir = work_dir / args.target_lang
    fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else []

    if not args.skip_download:
        print(f"\n[1/4] Downloading kaggle://{args.handle} → {raw_dir}")
        kaggle_download_repo(
            args.handle,
            work_dir,
            version=args.version,
            force=args.force_download,
        )
    else:
        print(f"\n[1/4] Skipping download (raw expected at {raw_dir})")

    if not args.skip_translate:
        print(
            f"\n[2/4] Translating {args.source_lang} → {args.target_lang} "
            f"(fields={fields or 'all string fields'})"
        )
        cache_path = Path(args.cache)
        total = translate_dataset(
            raw_dir=raw_dir,
            out_dir=bg_dir,
            fields=fields,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            cache_path=cache_path,
            delay=args.delay,
        )
        print(f"  total records processed: {total}")
    else:
        print("\n[2/4] Skipping translation")

    minio_prefix_full = f"{args.minio_prefix.rstrip('/')}/{slug}"
    if not args.skip_minio:
        bucket_label = args.bucket or os.getenv("MINIO_BUCKET", "data")
        print(f"\n[3/4] Uploading to MinIO at s3://{bucket_label}/{minio_prefix_full}")
        storage = _build_storage(backend="minio", bucket=args.bucket, secure=args.secure)
        n_raw = upload_to_minio(storage, raw_dir, f"{minio_prefix_full}/raw") if raw_dir.exists() else 0
        n_bg = upload_to_minio(storage, bg_dir, f"{minio_prefix_full}/{args.target_lang}") if bg_dir.exists() else 0
        print(f"  uploaded {n_raw} raw + {n_bg} translated files")
    else:
        print("\n[3/4] Skipping MinIO upload")

    hf_prefix_full = f"{args.hf_prefix.rstrip('/')}/{slug}"
    if not args.skip_hf:
        api = HfApi(token=os.getenv("HF_TOKEN"))
        hf_bucket = args.hf_bucket
        if "/" not in hf_bucket:
            try:
                hf_bucket = f"{api.whoami()['name']}/{hf_bucket}"
            except Exception:
                pass
        print(f"\n[4/4] Uploading to HF bucket '{hf_bucket}' at '{hf_prefix_full}'")
        n_raw = upload_to_hf_bucket(api, hf_bucket, raw_dir, f"{hf_prefix_full}/raw") if raw_dir.exists() else 0
        n_bg = upload_to_hf_bucket(api, hf_bucket, bg_dir, f"{hf_prefix_full}/{args.target_lang}") if bg_dir.exists() else 0
        print(f"  uploaded {n_raw} raw + {n_bg} translated files")
    else:
        print("\n[4/4] Skipping HF bucket upload")

    print("\nDone.")
    return 0


# ----- argument parser ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_transformation",
        description="Translate datasets and filter downloaded books by language.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pt = sub.add_parser(
        "translate",
        help="Translate strings in a dataset (jsonl / json / csv / parquet / txt)",
    )
    pt.add_argument("--input", required=True, help="Input file path")
    pt.add_argument("--output", required=True, help="Output file path")
    pt.add_argument(
        "--fields",
        default=None,
        help="Comma-separated dotted field paths to translate (e.g. 'title,summary'). "
             "Omit to translate every top-level string field.",
    )
    pt.add_argument("--source-lang", default="en", help="Source language code (default: en)")
    pt.add_argument("--target-lang", default="bg", help="Target language code (default: bg)")
    pt.add_argument("--cache", default=str(DEFAULT_CACHE), help="Translation cache JSON file")
    pt.add_argument("--delay", type=float, default=0.0, help="Sleep seconds between API calls")

    pf = sub.add_parser(
        "filter-language",
        help="Drop books that don't match the expected language (default: gutenberg → en)",
    )
    pf.add_argument(
        "--source",
        default="project_gutenberg",
        help="Manifest source name to filter (default: project_gutenberg). ",
    )
    pf.add_argument("--keep-lang", default="en", help="Language code to keep (default: en)")
    pf.add_argument("--duckdb", default=str(DEFAULT_DUCKDB))
    pf.add_argument("--tmp-dir", default=str(DEFAULT_TMP))
    pf.add_argument("--remote-prefix", default="raw")
    pf.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    pf.add_argument(
        "--backend",
        default="minio",
        choices=["minio", "local", "huggingface"],
        help="Storage backend the books were saved to (default: minio)",
    )
    pf.add_argument("--bucket", default=None, help="Bucket name (MinIO; defaults to env MINIO_BUCKET=data)")
    pf.add_argument(
        "--secure",
        default=None,
        type=lambda v: v.lower() in ("1", "true", "yes"),
        help="Force MinIO TLS on/off (defaults to MINIO_FORCE_SECURE env var)",
    )
    pf.add_argument("--limit", type=int, default=None, help="Stop after inspecting N books (debug)")
    pf.add_argument(
        "--apply",
        action="store_true",
        help="Without this flag the command is a dry run.",
    )
    pf.add_argument(
        "--skip-metadata-upload",
        action="store_true",
        help="Don't re-upload books.duckdb / books.parquet after removals",
    )

    pd = sub.add_parser(
        "hf-dataset",
        help="Download a HuggingFace dataset, translate fields, push to MinIO + HF bucket",
    )
    pd.add_argument("--repo", required=True, help="HuggingFace dataset repo id (e.g. Amod/mental_health_counseling_conversations)")
    pd.add_argument(
        "--fields",
        default=None,
        help="Comma-separated field paths to translate. Omit to translate every top-level string field.",
    )
    pd.add_argument("--source-lang", default="en")
    pd.add_argument("--target-lang", default="bg")
    pd.add_argument("--cache", default=str(DEFAULT_CACHE))
    pd.add_argument("--delay", type=float, default=0.0)
    pd.add_argument("--tmp-dir", default=str(DEFAULT_TMP))
    pd.add_argument(
        "--minio-prefix",
        default="datasets/huggingface",
        help="MinIO key prefix under the bucket (default: datasets/huggingface)",
    )
    pd.add_argument("--bucket", default=None, help="MinIO bucket; defaults to env MINIO_BUCKET=data")
    pd.add_argument(
        "--secure",
        default=None,
        type=lambda v: v.lower() in ("1", "true", "yes"),
    )
    pd.add_argument(
        "--hf-bucket",
        default="data",
        help="HuggingFace bucket id to mirror into (default: data)",
    )
    pd.add_argument(
        "--hf-prefix",
        default="datasets/huggingface",
        help="Path prefix inside the HF bucket (default: datasets/huggingface)",
    )
    pd.add_argument("--skip-download", action="store_true")
    pd.add_argument("--skip-translate", action="store_true")
    pd.add_argument("--skip-minio", action="store_true")
    pd.add_argument("--skip-hf", action="store_true")

    pk = sub.add_parser(
        "kaggle-dataset",
        help="Download a Kaggle dataset, translate fields, push to MinIO + HF bucket",
    )
    pk.add_argument(
        "--handle",
        required=True,
        help="Kaggle dataset handle (e.g. nguyenletruongthien/mental-health)",
    )
    pk.add_argument(
        "--version",
        default=None,
        help="Pin a specific Kaggle dataset version (default: latest).",
    )
    pk.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if kagglehub already has a cached copy.",
    )
    pk.add_argument(
        "--fields",
        default=None,
        help="Comma-separated field paths to translate. Omit to translate every top-level string field.",
    )
    pk.add_argument("--source-lang", default="en")
    pk.add_argument("--target-lang", default="bg")
    pk.add_argument("--cache", default=str(DEFAULT_CACHE))
    pk.add_argument("--delay", type=float, default=0.0)
    pk.add_argument("--tmp-dir", default=str(DEFAULT_TMP))
    pk.add_argument(
        "--minio-prefix",
        default="datasets/kaggle",
        help="MinIO key prefix under the bucket (default: datasets/kaggle)",
    )
    pk.add_argument("--bucket", default=None, help="MinIO bucket; defaults to env MINIO_BUCKET=data")
    pk.add_argument(
        "--secure",
        default=None,
        type=lambda v: v.lower() in ("1", "true", "yes"),
    )
    pk.add_argument(
        "--hf-bucket",
        default="data",
        help="HuggingFace bucket id to mirror into (default: data)",
    )
    pk.add_argument(
        "--hf-prefix",
        default="datasets/kaggle",
        help="Path prefix inside the HF bucket (default: datasets/kaggle)",
    )
    pk.add_argument("--skip-download", action="store_true")
    pk.add_argument("--skip-translate", action="store_true")
    pk.add_argument("--skip-minio", action="store_true")
    pk.add_argument("--skip-hf", action="store_true")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "translate":
        return _cmd_translate(args)
    if args.command == "filter-language":
        return _cmd_filter_language(args)
    if args.command == "hf-dataset":
        return _cmd_hf_dataset(args)
    if args.command == "kaggle-dataset":
        return _cmd_kaggle_dataset(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
