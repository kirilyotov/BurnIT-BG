"""Sync files from MinIO to a HuggingFace Bucket.

Usage:

    python -m run_scripts.sync_minio_to_hf_bucket

Edit the ``TRANSFERS`` table at the top to add / remove mappings. Each row is
``(minio_bucket, minio_object_key, hf_bucket_id, hf_remote_path)``.

The default ``HF_BUCKET`` is ``kiplayo/data`` — change if you want a different
target. ``.env`` at the repo root is auto-loaded for ``MINIO_*`` and ``HF_TOKEN``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from data_platform.storage.minio import MinioStorage
from data_platform.storage.hugging_face import HuggingFaceStorage


HF_BUCKET = "kiplayo/data"

# (minio_bucket, minio_object_key, hf_bucket_id, hf_remote_path)
TRANSFERS: list[tuple[str, str, str, str]] = [
    (
        "data",
        "datasets/chitanka/final/bg/dataset.jsonl",
        HF_BUCKET,
        "datasets/chitanka/bg/dataset.jsonl",
    ),
    (
        "data",
        "datasets/huggingface/mental_health_counseling_conversations/bg/combined_dataset.jsonl",
        HF_BUCKET,
        "datasets/huggingface/mental_health/combined_dataset.jsonl",
    ),
]


def _human(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"


def main() -> int:
    minio = MinioStorage.from_env()
    hf = HuggingFaceStorage.from_env()

    work = Path(tempfile.mkdtemp(prefix="minio_to_hf_"))
    print(f"[sync] staging dir: {work}\n")

    failures: list[str] = []
    for src_bucket, src_key, hf_bucket, hf_path in TRANSFERS:
        print(f"=== s3://{src_bucket}/{src_key}")
        print(f"  ↳ hf://buckets/{hf_bucket}/{hf_path}")
        local = work / Path(src_key).name
        try:
            minio.load_file(src_key, local, bucket=src_bucket)
            size = local.stat().st_size
            print(f"  [minio] downloaded {_human(size)} -> {local}")
        except Exception as exc:
            msg = f"download failed: {type(exc).__name__}: {exc}"
            print(f"  [error] {msg}")
            failures.append(f"{src_bucket}/{src_key}: {msg}")
            continue

        try:
            hf.api.batch_bucket_files(hf_bucket, add=[(str(local), hf_path)])
            print(f"  [hf] uploaded -> {hf_bucket}/{hf_path}\n")
        except Exception as exc:
            msg = f"upload failed: {type(exc).__name__}: {exc}"
            print(f"  [error] {msg}\n")
            failures.append(f"{hf_bucket}/{hf_path}: {msg}")

    print("=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"All {len(TRANSFERS)} transfers completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
