"""End-to-end Kaggle dataset pipeline.

Downloads a Kaggle dataset via :mod:`kagglehub` and reuses the same
translate / upload machinery as :mod:`data_transformation.hf_dataset`, so
running the pipeline on a Kaggle source produces the identical
``raw/`` and ``{target_lang}/`` layout on MinIO and the HuggingFace
bucket.

Authentication
--------------
``kagglehub.dataset_download`` reads credentials from any of:

* ``~/.kaggle/kaggle.json`` (the standard file)
* ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables

Set them once and the downloader is silent; the rest of this module is
just plumbing.

Layout produced under ``{tmp_dir}/datasets/{slug}``::

    raw/   ← exact files from the Kaggle dataset
    bg/    ← translated copies of data files only (.jsonl, .csv, .parquet)

These two folders are then mirrored to:

* ``s3://{bucket}/{minio_prefix}/{slug}/{raw,bg}/...``
* ``{hf_bucket}/{hf_prefix}/{slug}/{raw,bg}/...`` (HuggingFace bucket)
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from .hf_dataset import translate_dataset, upload_to_hf_bucket, upload_to_minio

log = logging.getLogger(__name__)


def dataset_slug(handle: str) -> str:
    """Turn a Kaggle handle (``owner/dataset``) into a filesystem-safe slug.

    For Kaggle the convention is ``owner/dataset-name`` — we keep only the
    dataset part, matching what :func:`hf_dataset.dataset_slug` does.
    """
    name = handle.split("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def download_repo(
    handle: str,
    dest_dir: Path,
    *,
    version: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download a Kaggle dataset into ``dest_dir/raw``.

    ``kagglehub.dataset_download`` returns a path inside its own cache; we
    copy the contents into our project's ``raw/`` directory so the rest of
    the pipeline (translation, upload) is filesystem-agnostic and
    inspectable.

    Args:
        handle: ``"owner/dataset"`` — same string used in
            ``kagglehub.dataset_download``.
        dest_dir: Working directory; ``raw/`` is created inside.
        version: Optional dataset version pin (defaults to latest).
        force: Re-download even if kagglehub already cached the snapshot.
    """
    import kagglehub  # local import — only required for this command

    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    qualified = f"{handle}/versions/{version}" if version else handle
    log.info("downloading kaggle dataset %s -> %s", qualified, raw_dir)
    cache_path = Path(kagglehub.dataset_download(qualified, force_download=force))

    if not cache_path.exists():
        raise FileNotFoundError(
            f"kagglehub returned a non-existent path: {cache_path}"
        )

    # Copy from kagglehub's cache directory into our raw/ tree so the
    # downstream pipeline works against a stable, project-local layout.
    _copy_tree(cache_path, raw_dir)
    return raw_dir


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy file or directory ``src`` into ``dst`` (created if missing).

    Works whether kagglehub returns a directory (most datasets) or a single
    file (rare, but possible for `.zip`-only datasets that have been
    auto-extracted).
    """
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst / src.name)
        return
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
