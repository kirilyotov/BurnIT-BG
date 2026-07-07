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
from .build_dataset import BuildConfig, build_dataset, maybe_upload, split_by_topic
from .build_qa_dataset import QABuildConfig, build_qa_dataset
from .publish_dataset import PublishConfig, publish_dataset, stage_repo
from .rewrite_instructions import RewriteConfig, rewrite_instructions
from .rtuning.build_rtuning import BuildRTuningConfig, REFUSAL_TEMPLATES_BG, build_rtuning
from .rtuning.combine import CombineConfig, combine_datasets
from .rtuning.download_raw import SOURCE_SPECS, DownloadRawConfig, download_raw
from .rtuning.publish_dataset import PublishRTuningConfig, publish_rtuning
from .rtuning.translate import TranslateRTuningConfig, translate_rtuning
from .rtuning.upload_raw import UploadRawConfig, upload_raw
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


# ###### translate ######


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


# ###### filter-language ######


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


# ###### hf-dataset ######


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


# ###### kaggle-dataset ######


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


# ###### from-passages ######


def _cmd_from_passages(args: argparse.Namespace) -> int:
    """Build an Alpaca-style mental-health JSONL from extracted passages."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()

    cfg = BuildConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        seed=args.seed,
        min_words=args.min_words,
        max_chars=args.max_chars,
        drop_dangerous=not args.allow_dangerous,
        extraction_date=args.date,
        quality_score=args.quality_score,
    )
    total_in, total_out = build_dataset(cfg)
    print(f"\nbuild-from-passages: read {total_in} passages, wrote {total_out} records "
          f"-> {cfg.output_path}")

    if not args.skip_minio:
        try:
            uri = maybe_upload(cfg.output_path, source=args.source, date=args.date, bucket=args.bucket)
            if uri:
                print(f"uploaded -> {uri}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] MinIO upload failed: {exc}", file=sys.stderr)
    return 0


# ###### rewrite-instructions ######


def _cmd_rewrite_instructions(args: argparse.Namespace) -> int:
    """Rewrite each record's instruction with Mistral so it actually fits its passage."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()
    cache_path = Path(args.cache) if args.cache else None
    cfg = RewriteConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model=args.model,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        max_passage_chars=args.max_passage_chars,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_path=cache_path,
        extraction_date=args.date,
    )
    counts = rewrite_instructions(cfg)
    print(
        f"\nrewrite-instructions: read {counts['total_in']}, "
        f"rewrote {counts['rewritten_live']}, cached {counts['cache_hits']}, "
        f"fallback (kept original) {counts['fallbacks']} -> {cfg.output_path}"
    )
    if not args.skip_minio:
        try:
            uri = maybe_upload(cfg.output_path, source=args.source, date=args.date, bucket=args.bucket)
            if uri:
                print(f"uploaded -> {uri}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] MinIO upload failed: {exc}", file=sys.stderr)
    return 0


# ###### qa-from-passages ######


def _cmd_qa_from_passages(args: argparse.Namespace) -> int:
    """Generate a real Bulgarian Q->A dataset from passages via an LLM."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()

    cache_path = Path(args.cache) if args.cache else None
    cfg = QABuildConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model=args.model,
        limit=args.limit,
        min_words=args.min_words,
        max_passage_chars=args.max_passage_chars,
        delay=args.delay,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_path=cache_path,
        quality_score=args.quality_score,
        extraction_date=args.date,
    )
    total_in, total_out = build_qa_dataset(cfg)
    print(f"\nqa-from-passages: read {total_in} passages, wrote {total_out} Q->A records "
          f"-> {cfg.output_path}")

    if not args.skip_minio:
        try:
            uri = maybe_upload(cfg.output_path, source=args.source, date=args.date, bucket=args.bucket)
            if uri:
                print(f"uploaded -> {uri}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] MinIO upload failed: {exc}", file=sys.stderr)
    return 0


# ###### publish-dataset ######


def _cmd_publish_dataset(args: argparse.Namespace) -> int:
    """Stage (and optionally upload) the mental-health dataset to HuggingFace."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()

    cfg = PublishConfig(
        repo_id=args.repo,
        staging_dir=Path(args.staging_dir),
        style_path=Path(args.style) if args.style else None,
        qa_path=Path(args.qa) if args.qa else None,
        splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        private=args.private,
        license=args.license,
    )

    if args.stage_only:
        staging = stage_repo(cfg)
        print(f"\nstaged dataset repo at: {staging}")
        print("(review it, then re-run without --stage-only to upload)")
        return 0

    uri = publish_dataset(cfg)
    print(f"\npublished -> {uri}  (private={cfg.private})")
    return 0


# ###### split-by-topic ######


def _cmd_split_by_topic(args: argparse.Namespace) -> int:
    """Split a built Alpaca dataset into per-topic + combined train/eval sets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = split_by_topic(
        input_path=Path(args.input),
        out_dir=Path(args.out_dir),
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_per_topic=args.min_per_topic,
        stratify_by=args.stratify_by,
    )

    print(f"\nsplit-by-topic -> {args.out_dir}")
    print(f"{'split':<24}{'train':>8}{'eval':>8}")
    for name in ["all"] + sorted(n for n in results if n != "all"):
        n_train, n_eval = results[name]
        print(f"{name:<24}{n_train:>8}{n_eval:>8}")
    print(f"\n{len(results) - 1} topic splits + 1 combined ('all').")
    return 0


# ###### R-Tuning pipeline ######


def _cmd_rtuning_download(args: argparse.Namespace) -> int:
    """Download a slimmed raw TriviaQA / SQuAD v2 to local jsonl."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()
    cfg = DownloadRawConfig(
        source=args.source,
        output_dir=Path(args.output_dir),
        split=args.split,
        limit=args.limit,
        revision=args.revision,
    )
    out_path = download_raw(cfg)
    print(f"\nrtuning-download: {args.source} -> {out_path}")
    return 0


def _cmd_rtuning_upload_raw(args: argparse.Namespace) -> int:
    """Push slimmed raw to MinIO (and optionally HF)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()
    cfg = UploadRawConfig(
        source=args.source,
        raw_jsonl=Path(args.input),
        minio_prefix=args.minio_prefix,
        hf_repo_id=args.hf_repo,
        push_minio=not args.skip_minio,
        push_hf=args.push_hf,
        private=args.private,
        minio_bucket=args.bucket,
    )
    res = upload_raw(cfg)
    print(f"\nrtuning-upload-raw: minio={res['minio']}  hf={res['hf']}")
    return 0


def _cmd_rtuning_translate(args: argparse.Namespace) -> int:
    """BG-translate the question + answer columns of a raw R-Tuning jsonl."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()
    cfg = TranslateRTuningConfig(
        input_jsonl=Path(args.input),
        output_jsonl=Path(args.output),
        cache_path=Path(args.cache) if args.cache else None,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        fields=tuple(f.strip() for f in args.fields.split(",") if f.strip()),
        concurrency=args.concurrency,
        chunk_size=args.chunk_size,
        delay=args.delay,
        flush_every=args.flush_every,
        limit=args.limit,
        backend=args.backend,
        backend_url=args.backend_url,
    )
    counts = translate_rtuning(cfg)
    print(f"\nrtuning-translate: input={counts['total_in']} translated={counts['translated']} "
          f"resumed_from={counts.get('resumed_from', 0)} "
          f"cache={counts.get('cache_size', 0)} -> {cfg.output_jsonl}")
    return 0


def _cmd_rtuning_build(args: argparse.Namespace) -> int:
    """Turn translated rows into Alpaca-style R-Tuning records."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()
    cfg = BuildRTuningConfig(
        input_jsonl=Path(args.input),
        output_jsonl=Path(args.output),
        refusal_templates=REFUSAL_TEMPLATES_BG,
        rotate=args.rotate,
        seed=args.seed,
    )
    counts = build_rtuning(cfg)
    print(f"\nrtuning-build: total={counts['total_in']} written={counts['written']} "
          f"skipped={counts['skipped_no_question']} -> {cfg.output_jsonl}")
    return 0


def _cmd_rtuning_combine(args: argparse.Namespace) -> int:
    """Concatenate + shuffle multiple R-Tuning jsonls into one combined file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = CombineConfig(
        inputs=[Path(p) for p in args.inputs],
        output_jsonl=Path(args.output),
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    res = combine_datasets(cfg)
    print(f"\nrtuning-combine: total={res['total']}  per_source={res['per_source']} "
          f"-> {cfg.output_jsonl}")
    return 0


def _cmd_rtuning_publish(args: argparse.Namespace) -> int:
    """Push a curated R-Tuning jsonl to MinIO (and optionally HF) with a data card."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _load_env()
    cfg = PublishRTuningConfig(
        dataset_jsonl=Path(args.input),
        minio_prefix=args.minio_prefix,
        hf_repo_id=args.hf_repo,
        sources=[s.strip() for s in args.sources.split(",") if s.strip()],
        push_minio=not args.skip_minio,
        push_hf=args.push_hf,
        private=args.private,
        minio_bucket=args.bucket,
    )
    res = publish_rtuning(cfg)
    print(f"\nrtuning-publish: minio={res['minio']}  hf={res['hf']}")
    return 0


# ###### argument parser ######


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

    pfp = sub.add_parser(
        "from-passages",
        help="Build a Bulgarian Alpaca dataset from extracted Chitanka passages.",
    )
    pfp.add_argument("--input", required=True,
                     help="Path to passages JSONL produced by data_scraping extract-passages.")
    pfp.add_argument("--output", required=True,
                     help="Where to write the Alpaca JSONL.")
    pfp.add_argument("--source", default="chitanka",
                     help="Source label baked into each record's metadata + MinIO path.")
    pfp.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                     help="Date segment used in the MinIO destination key.")
    pfp.add_argument("--seed", type=int, default=42,
                     help="Random seed for the instruction-template sampler.")
    pfp.add_argument("--min-words", type=int, default=12,
                     help="Drop records whose output has fewer than N whitespace tokens. "
                          "Book passages are short, so the default is 12 (was 30, which "
                          "dropped a large share of otherwise-valid passages).")
    pfp.add_argument("--max-chars", type=int, default=1500,
                     help="Trim outputs longer than this many characters.")
    pfp.add_argument("--quality-score", type=float, default=0.80,
                     help="Default quality_score baked into every record (0..1).")
    pfp.add_argument("--allow-dangerous", action="store_true",
                     help="Disable the dangerous-phrase filter (NOT recommended).")
    pfp.add_argument("--bucket", default=None,
                     help="MinIO bucket override (defaults to env MINIO_BUCKET).")
    pfp.add_argument("--skip-minio", action="store_true",
                     help="Don't upload the built JSONL to MinIO; write only locally.")

    pri = sub.add_parser(
        "rewrite-instructions",
        help="Rewrite each record's instruction with Mistral so it actually matches its passage.",
    )
    pri.add_argument("--input", required=True,
                     help="Path to the existing Alpaca dataset JSONL (output of from-passages).")
    pri.add_argument("--output", required=True,
                     help="Where to write the rewritten JSONL (e.g. .../dataset_ai_improved.jsonl).")
    pri.add_argument("--model", default="mistral-large-3",
                     help="NVIDIA model handle (default: mistral-large-3).")
    pri.add_argument("--concurrency", type=int, default=8,
                     help="Number of parallel Mistral calls per batch (default: 8).")
    pri.add_argument("--batch-size", type=int, default=50,
                     help="Records per concurrent batch (default: 50).")
    pri.add_argument("--max-passage-chars", type=int, default=1000,
                     help="Trim each passage to this many chars before sending.")
    pri.add_argument("--temperature", type=float, default=0.3,
                     help="Sampling temperature for question generation.")
    pri.add_argument("--max-tokens", type=int, default=200,
                     help="Max tokens per generated question.")
    pri.add_argument("--cache", default=str(DEFAULT_TMP / "rewrite_cache.json"),
                     help="JSON cache keyed by passage_id so re-runs resume.")
    pri.add_argument("--source", default="chitanka",
                     help="Source label for the MinIO destination key.")
    pri.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                     help="Date segment used in the MinIO destination key.")
    pri.add_argument("--bucket", default=None,
                     help="MinIO bucket override (defaults to env MINIO_BUCKET).")
    pri.add_argument("--skip-minio", action="store_true",
                     help="Don't upload the output to MinIO; write only locally.")

    pqa = sub.add_parser(
        "qa-from-passages",
        help="Generate a real Bulgarian question->answer dataset from passages via an LLM.",
    )
    pqa.add_argument("--input", required=True,
                     help="Path to passages JSONL produced by data_scraping extract-passages.")
    pqa.add_argument("--output", required=True,
                     help="Where to write the generated Q->A Alpaca JSONL.")
    pqa.add_argument("--model", default="mistral-large-3",
                     help="NVIDIA model handle to generate with (default: mistral-large-3).")
    pqa.add_argument("--limit", type=int, default=None,
                     help="Only process the first N passages (use a small value to test cost first).")
    pqa.add_argument("--min-words", type=int, default=12,
                     help="Drop generated answers shorter than N words.")
    pqa.add_argument("--max-passage-chars", type=int, default=1200,
                     help="Trim each source passage to this many characters before sending.")
    pqa.add_argument("--delay", type=float, default=0.0,
                     help="Sleep seconds between API calls (rate limiting).")
    pqa.add_argument("--temperature", type=float, default=0.6,
                     help="Sampling temperature for generation (default: 0.6).")
    pqa.add_argument("--max-tokens", type=int, default=1024,
                     help="Max tokens per generated Q->A pair.")
    pqa.add_argument("--cache", default=str(DEFAULT_TMP / "qa_cache.json"),
                     help="JSON cache keyed by passage_id so re-runs skip generated passages.")
    pqa.add_argument("--quality-score", type=float, default=0.85,
                     help="Default quality_score baked into every record (0..1).")
    pqa.add_argument("--source", default="chitanka",
                     help="Source label for the MinIO destination path.")
    pqa.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                     help="Date segment used in the MinIO destination key.")
    pqa.add_argument("--bucket", default=None,
                     help="MinIO bucket override (defaults to env MINIO_BUCKET).")
    pqa.add_argument("--skip-minio", action="store_true",
                     help="Don't upload the built JSONL to MinIO; write only locally.")

    ppub = sub.add_parser(
        "publish-dataset",
        help="Stage + upload the mental-health dataset(s) to a HuggingFace dataset repo.",
    )
    ppub.add_argument("--repo", required=True,
                      help="HuggingFace dataset repo id (e.g. username/burnit-bg-mental-health).")
    ppub.add_argument("--style", default=None,
                      help="Path to the style dataset JSONL (from `from-passages`). "
                           "NOTE: contains verbatim book excerpts — mind copyright.")
    ppub.add_argument("--qa", default=None,
                      help="Path to the synthetic Q->A dataset JSONL (from `qa-from-passages`).")
    ppub.add_argument("--splits-dir", default=None,
                      help="Path to the topic-splits directory (from `split-by-topic`).")
    ppub.add_argument("--staging-dir", default=str(DEFAULT_TMP / "hf_dataset_staging"),
                      help="Local directory to assemble the repo in before upload.")
    ppub.add_argument("--license", default="cc-by-nc-4.0",
                      help="License string for the data card (default: cc-by-nc-4.0).")
    ppub.add_argument("--private", action="store_true",
                      help="Create the repo as private (default: public).")
    ppub.add_argument("--stage-only", action="store_true",
                      help="Assemble the repo locally but do NOT upload (review first).")

    pst = sub.add_parser(
        "split-by-topic",
        help="Split a built Alpaca dataset into per-topic + combined train/eval sets.",
    )
    pst.add_argument("--input", required=True,
                     help="Path to a built Alpaca dataset JSONL (from `from-passages`).")
    pst.add_argument("--out-dir", required=True,
                     help="Directory to write all/ and by-topic/{topic}/ splits into.")
    pst.add_argument("--eval-ratio", type=float, default=0.1,
                     help="Fraction of each split held out for eval (default: 0.1).")
    pst.add_argument("--seed", type=int, default=42,
                     help="Shuffle/split seed (default: 42).")
    pst.add_argument("--min-per-topic", type=int, default=10,
                     help="Topics with fewer records are folded into a _misc bucket.")
    pst.add_argument("--stratify-by", default="category",
                     help="Record field to stratify each split by (default: category).")

    # ── R-Tuning subcommands ────────────────────────────────────────────────
    SOURCE_CHOICES = sorted(SOURCE_SPECS.keys())

    prtd = sub.add_parser(
        "rtuning-download",
        help="Download a slimmed raw R-Tuning source (TriviaQA / SQuAD v2).",
    )
    prtd.add_argument("--source", required=True, choices=SOURCE_CHOICES,
                      help="Upstream dataset to fetch.")
    prtd.add_argument("--output-dir", required=True,
                      help="Local directory to write the raw jsonl into.")
    prtd.add_argument("--split", default="train", help="Upstream split (default: train).")
    prtd.add_argument("--limit", type=int, default=None,
                      help="Only fetch the first N rows (smoke-test mode).")
    prtd.add_argument("--revision", default=None, help="Pin a specific HF dataset revision.")

    prtu = sub.add_parser(
        "rtuning-upload-raw",
        help="Push slimmed raw R-Tuning jsonl to MinIO (and optionally HF Hub).",
    )
    prtu.add_argument("--source", required=True, choices=SOURCE_CHOICES)
    prtu.add_argument("--input", required=True, help="Raw jsonl produced by rtuning-download.")
    prtu.add_argument("--minio-prefix", required=True,
                      help="Destination prefix in the MinIO bucket "
                           "(e.g. datasets/rtuning/triviaqa/2026-06-02/raw).")
    prtu.add_argument("--hf-repo", default=None,
                      help="HF dataset repo id (e.g. kiplayo/rtuning-triviaqa-raw).")
    prtu.add_argument("--push-hf", action="store_true",
                      help="Actually push to HF (default: stage to MinIO only).")
    prtu.add_argument("--private", action="store_true", help="Create the HF repo as private.")
    prtu.add_argument("--bucket", default=None, help="MinIO bucket override.")
    prtu.add_argument("--skip-minio", action="store_true", help="Don't upload to MinIO.")

    prtt = sub.add_parser(
        "rtuning-translate",
        help="BG-translate the question + answer columns of a raw R-Tuning jsonl.",
    )
    prtt.add_argument("--input", required=True)
    prtt.add_argument("--output", required=True)
    prtt.add_argument("--cache",
                      default=str(DEFAULT_TMP / "rtuning" / "translate_cache.json"))
    prtt.add_argument("--source-lang", default="en")
    prtt.add_argument("--target-lang", default="bg")
    prtt.add_argument("--fields", default="question,answer",
                      help="Comma-separated columns to translate (default: question,answer).")
    prtt.add_argument("--backend", choices=["google", "libretranslate"], default="google",
                      help="Translation backend (default: google). Use libretranslate "
                           "for self-hosted unlimited throughput.")
    prtt.add_argument("--backend-url", default=None,
                      help="URL for the chosen backend (libretranslate only). "
                           "Defaults to $LIBRETRANSLATE_URL or http://localhost:5000.")
    prtt.add_argument("--concurrency", type=int, default=8,
                      help="Number of parallel translator threads (default: 8 for google, "
                           "raise to 32-64 for libretranslate).")
    prtt.add_argument("--chunk-size", type=int, default=500,
                      help="Records buffered per chunk (default: 500). Lower = "
                           "more frequent flushes / less memory; higher = less overhead.")
    prtt.add_argument("--delay", type=float, default=0.0,
                      help="Sleep seconds between API calls.")
    prtt.add_argument("--flush-every", type=int, default=100,
                      help="Flush output every N records.")
    prtt.add_argument("--limit", type=int, default=None,
                      help="Only process the first N rows (smoke-test mode).")

    prtb = sub.add_parser(
        "rtuning-build",
        help="Assemble translated rows into Alpaca-style R-Tuning records.",
    )
    prtb.add_argument("--input", required=True, help="Translated jsonl from rtuning-translate.")
    prtb.add_argument("--output", required=True)
    prtb.add_argument("--rotate", choices=["round-robin", "random"], default="round-robin",
                      help="How to rotate among the 3 BG refusal templates.")
    prtb.add_argument("--seed", type=int, default=42)

    prtc = sub.add_parser(
        "rtuning-combine",
        help="Concatenate + shuffle several R-Tuning jsonls into one combined file.",
    )
    prtc.add_argument("--inputs", nargs="+", required=True,
                      help="Per-source curated jsonls (e.g. triviaqa-bg.jsonl squadv2-bg.jsonl).")
    prtc.add_argument("--output", required=True)
    prtc.add_argument("--no-shuffle", action="store_true",
                      help="Concatenate in input order without shuffling.")
    prtc.add_argument("--seed", type=int, default=42)

    prtp = sub.add_parser(
        "rtuning-publish",
        help="Stage a curated R-Tuning jsonl + data card and push to MinIO + HF.",
    )
    prtp.add_argument("--input", required=True, help="Curated jsonl to publish.")
    prtp.add_argument("--minio-prefix", required=True,
                      help="Destination prefix in MinIO "
                           "(e.g. datasets/rtuning/combined/2026-06-02/bg).")
    prtp.add_argument("--hf-repo", default=None,
                      help="HF dataset repo id (e.g. kiplayo/burnit-bg-rtuning-combined-bg).")
    prtp.add_argument("--sources", default="",
                      help="Comma-separated source tags for the data card "
                           "(e.g. 'triviaqa,squadv2').")
    prtp.add_argument("--push-hf", action="store_true",
                      help="Actually push to HF (default: stage to MinIO only).")
    prtp.add_argument("--private", action="store_true")
    prtp.add_argument("--bucket", default=None, help="MinIO bucket override.")
    prtp.add_argument("--skip-minio", action="store_true", help="Don't upload to MinIO.")

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
    if args.command == "from-passages":
        return _cmd_from_passages(args)
    if args.command == "qa-from-passages":
        return _cmd_qa_from_passages(args)
    if args.command == "rewrite-instructions":
        return _cmd_rewrite_instructions(args)
    if args.command == "split-by-topic":
        return _cmd_split_by_topic(args)
    if args.command == "publish-dataset":
        return _cmd_publish_dataset(args)
    if args.command == "rtuning-download":
        return _cmd_rtuning_download(args)
    if args.command == "rtuning-upload-raw":
        return _cmd_rtuning_upload_raw(args)
    if args.command == "rtuning-translate":
        return _cmd_rtuning_translate(args)
    if args.command == "rtuning-build":
        return _cmd_rtuning_build(args)
    if args.command == "rtuning-combine":
        return _cmd_rtuning_combine(args)
    if args.command == "rtuning-publish":
        return _cmd_rtuning_publish(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
