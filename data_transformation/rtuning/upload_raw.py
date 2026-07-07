"""Push the raw EN R-Tuning datasets to MinIO and (optionally) HF Hub.

Raw means the slim JSONL produced by ``download_raw`` — just enough columns to
translate. Upstream attribution + license ride along in a small README so the
HF dataset card is legally clean even before we add the BG split.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .download_raw import SOURCE_SPECS

log = logging.getLogger(__name__)


@dataclass
class UploadRawConfig:
    """Settings for :func:`upload_raw`."""

    source: str
    raw_jsonl: Path
    minio_prefix: str
    hf_repo_id: str | None = None
    push_minio: bool = True
    push_hf: bool = False
    private: bool = False
    minio_bucket: str | None = None


def upload_raw(cfg: UploadRawConfig) -> dict[str, str | None]:
    """Stage ``raw_jsonl`` + a tiny README, then push to MinIO + HF.

    Returns ``{"minio": <prefix or None>, "hf": <hf:// uri or None>}``.
    """
    spec = SOURCE_SPECS[cfg.source]
    staging = cfg.raw_jsonl.parent / f"_stage-{cfg.source}-raw"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    shutil.copy2(cfg.raw_jsonl, staging / cfg.raw_jsonl.name)
    (staging / "LICENSE.txt").write_text(
        f"Source: {spec['repo_id']} (config: {spec['config']})\n"
        f"Homepage: {spec['homepage']}\n"
        f"License: {spec['license']}\n"
        f"Citation: {spec['citation']}\n",
        encoding="utf-8",
    )
    (staging / "README.md").write_text(_raw_readme(cfg.source, spec), encoding="utf-8")

    out: dict[str, str | None] = {"minio": None, "hf": None}

    if cfg.push_minio:
        from data_platform.storage.minio import MinioStorage
        storage = MinioStorage.from_env()
        bucket = cfg.minio_bucket or storage.bucket
        storage.save_directory(staging, cfg.minio_prefix, bucket=bucket)
        out["minio"] = f"s3://{bucket}/{cfg.minio_prefix}"
        log.info("uploaded raw %s -> %s", cfg.source, out["minio"])

    if cfg.push_hf and cfg.hf_repo_id:
        from huggingface_hub import DatasetCardData

        from data_platform.storage.hugging_face import HuggingFaceStorage

        storage = HuggingFaceStorage.from_env()
        card_data = DatasetCardData(
            license=spec["license"],
            language=["en"],
            task_categories=["question-answering"],
            tags=[cfg.source, "raw", "english"],
            pretty_name=f"R-Tuning raw — {cfg.source.upper()}",
        )
        out["hf"] = storage.save_dataset(
            local_dir=staging,
            dataset_id=cfg.hf_repo_id,
            private=cfg.private,
            commit_message=f"Upload {cfg.source} raw for R-Tuning",
            card_data=card_data,
            card_content=_raw_readme(cfg.source, spec),
        )
        log.info("uploaded raw %s -> %s", cfg.source, out["hf"])

    return out


def _raw_readme(source: str, spec: dict) -> str:
    return f"""# R-Tuning raw mirror — {source}

This is a slimmed, redistributable mirror of [`{spec['repo_id']}`](https://huggingface.co/datasets/{spec['repo_id']})
({spec['config']}), kept for the BurnIT-BG R-Tuning out-of-domain refusal training pipeline.

Only the columns needed for refusal training are kept (`question`, `answer`,
plus IDs / metadata flags). The original text is unchanged.

## Upstream

- Repo: [`{spec['repo_id']}`](https://huggingface.co/datasets/{spec['repo_id']})
- Homepage: {spec['homepage']}
- License: `{spec['license']}`
- Citation: {spec['citation']}

## Why this exists

The BurnIT-BG model is a Bulgarian mental-health peer-support assistant. We
use this raw EN dataset (paired with its translated BG counterpart) to teach
the model to refuse out-of-domain trivia questions, so it stays focused on
its actual domain.
"""


__all__ = ["UploadRawConfig", "upload_raw"]
