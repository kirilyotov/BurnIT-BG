"""Upload a local directory or single file to MinIO.

Used after ``data_prep/prepare_mental_health.py`` to push the processed
``train.jsonl`` / ``eval.jsonl`` to a bucket for use in experiments.

Usage::

    python -m data_prep.upload_minio \
        --source_dir data_prep/processed \
        --prefix data_prep/processed/mental-health
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_platform.common import set_env
from data_platform.storage import MinioStorage

from ._common import file_size_mb, setup_tracking, timestamp_run_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a local path to MinIO.")
    p.add_argument("--source_dir", required=True, help="Local file or directory to upload.")
    p.add_argument("--bucket", default=None, help="MinIO bucket (defaults to $MINIO_BUCKET).")
    p.add_argument("--prefix", required=True, help="Remote key prefix.")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_env(quiet=True)
    storage = MinioStorage.from_env()

    src = Path(args.source_dir)
    if not src.exists():
        print(f"error: source not found: {src}", file=sys.stderr)
        return 1

    if src.is_dir():
        print(f"[minio] uploading dir {src} -> s3://{args.bucket or storage.bucket}/{args.prefix}")
        uri = storage.save_directory(src, args.prefix, bucket=args.bucket)
        files = [p for p in src.rglob("*") if p.is_file()]
    else:
        target = f"{args.prefix.rstrip('/')}/{src.name}" if args.prefix and not args.prefix.endswith(src.name) else args.prefix
        print(f"[minio] uploading file {src} -> s3://{args.bucket or storage.bucket}/{target}")
        uri = storage.save_file(src, target, bucket=args.bucket)
        files = [src]

    total_mb = sum(file_size_mb(p) for p in files)
    print(f"[minio] uploaded {len(files)} files ({total_mb:.2f} MB) -> {uri}")

    if args.no_mlflow:
        return 0

    try:
        tracking = setup_tracking("dataset_upload")
        with tracking.run(
            run_name=timestamp_run_name(f"upload-{args.prefix.replace('/', '-')}"),
            tags={
                "stage": "dataset_upload",
                "target": "minio",
                "prefix": args.prefix,
            },
            with_hardware=False,
            log_system_metrics=False,
        ):
            tracking.log_params({
                "bucket": args.bucket or storage.bucket,
                "prefix": args.prefix,
                "source": str(src),
            })
            tracking.log_metrics({
                "num_files": float(len(files)),
                "size_mb_total": float(total_mb),
            })
            for p in files[:50]:  # cap manifest size
                tracking.log_source_uri(f"file:{p.name}", str(p.relative_to(src.parent)))
    except Exception as exc:  # noqa: BLE001
        print(f"[minio] mlflow logging failed (continuing): {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
