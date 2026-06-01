"""Download a dataset prefix from MinIO into a local directory.

Thin wrapper around :class:`data_platform.storage.MinioStorage`. Used to
pull processed datasets that were previously uploaded with
``data_prep/upload_minio.py`` or the ``data_transformation`` pipeline.

Usage::

    python -m data_prep.download_minio \
        --bucket data \
        --prefix datasets/huggingface/mental_health_counseling_conversations \
        --output_dir data_prep/raw/mental_health_counseling_conversations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_platform.common import set_env
from data_platform.storage import MinioStorage

from ._common import file_size_mb, setup_tracking, timestamp_run_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a MinIO prefix to local.")
    p.add_argument("--bucket", default=None, help="MinIO bucket (defaults to $MINIO_BUCKET).")
    p.add_argument("--prefix", required=True, help="Remote key prefix (object path).")
    p.add_argument("--output_dir", required=True, help="Local destination directory.")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_env(quiet=True)
    storage = MinioStorage.from_env()

    out_dir = Path(args.output_dir)
    print(f"[minio] downloading s3://{args.bucket or storage.bucket}/{args.prefix} -> {out_dir}")
    local_root = storage.load_directory(args.prefix, out_dir, bucket=args.bucket)

    files = [p for p in local_root.rglob("*") if p.is_file()]
    total_mb = sum(file_size_mb(p) for p in files)
    print(f"[minio] downloaded {len(files)} files ({total_mb:.2f} MB)")

    if args.no_mlflow:
        return 0

    try:
        tracking = setup_tracking("dataset_download")
        with tracking.run(
            run_name=timestamp_run_name(f"minio-{args.prefix.replace('/', '-')}"),
            tags={
                "stage": "dataset_download",
                "source": "minio",
                "prefix": args.prefix,
            },
            with_hardware=False,
            log_system_metrics=False,
        ):
            tracking.log_params({
                "bucket": args.bucket or storage.bucket,
                "prefix": args.prefix,
            })
            tracking.log_metrics({
                "num_files": float(len(files)),
                "size_mb_total": float(total_mb),
            })
    except Exception as exc:  # noqa: BLE001
        print(f"[minio] mlflow logging failed (continuing): {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
