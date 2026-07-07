"""Assemble and publish the mental-health datasets to a HuggingFace repo.

Stages a single dataset repository directory out of the artifacts produced
earlier in the pipeline::

    {staging}/
      style/dataset.jsonl     ← from `from-passages` (raw book passages)
      qa/dataset.jsonl        ← from `qa-from-passages` (synthetic Q->A)
      splits/...              ← from `split-by-topic`
      README.md               ← data card (Bulgarian + English) + disclaimer

then uploads it via :class:`data_platform.storage.hugging_face.HuggingFaceStorage`.

IMPORTANT: the ``style`` split contains verbatim excerpts from copyrighted
books. Think carefully before publishing it publicly — the synthetic ``qa``
split is the safer thing to share. ``select`` lets you choose what to include.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


DATA_CARD = """---
license: {license}
language:
- bg
task_categories:
- text-generation
tags:
- mental-health
- bulgarian
- emotional-support
- peer-support
pretty_name: BurnIT-BG Mental-Health Support (Bulgarian)
---

# BurnIT-BG — Mental-Health Support Dataset (Bulgarian)

Обучаващи данни на български език за модел за **емоционална
връстническа подкрепа** (не клинична, не медицинска).

## ⚠️ Отказ от отговорност / Disclaimer

Този набор от данни и всеки модел, обучен върху него, са предназначени
**само за изследователски цели**. Те **не са медицински съвет** и не
заместват професионална психологическа или психиатрична помощ. Ако вие
или някой друг е в криза, потърсете незабавно професионална помощ.

This dataset and any model trained on it are for **research purposes only**,
are **not medical advice**, and do not replace professional care.

## Структура / Structure

- `qa/` — синтетични двойки въпрос→подкрепящ отговор, генерирани от LLM
  (Mistral Large 3) по теми за психично здраве. *(synthetic Q→A pairs)*
- `style/` — извадки от книги, сдвоени с шаблонни въпроси. *(style /
  continued-pretraining data — raw book excerpts)*
- `splits/` — разделяне по тема + комбиниран `all/` (train/eval).

Всеки запис е в каноничен Alpaca формат:
`{{"instruction", "input", "output", "category", "language": "bg", ...}}`.

## Източници / Sources

Извадките произхождат от книги от chitanka. **Внимание:** `style`
разделът може да съдържа защитен с авторско право текст — преценете
правните аспекти преди публична употреба.

## Категории / Categories

anxiety · depression · stress · grief · relationships · self-esteem · out_of_domain
"""


@dataclass
class PublishConfig:
    """Settings for staging + uploading the HF dataset repo."""

    repo_id: str
    staging_dir: Path
    style_path: Optional[Path] = None
    qa_path: Optional[Path] = None
    splits_dir: Optional[Path] = None
    private: bool = False
    license: str = "cc-by-nc-4.0"
    commit_message: str = "Publish BurnIT-BG mental-health dataset"


def stage_repo(cfg: PublishConfig) -> Path:
    """Assemble the staging directory and return its path."""
    staging = cfg.staging_dir
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    if cfg.style_path and cfg.style_path.exists():
        (staging / "style").mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.style_path, staging / "style" / "dataset.jsonl")
        log.info("staged style dataset")
    if cfg.qa_path and cfg.qa_path.exists():
        (staging / "qa").mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.qa_path, staging / "qa" / "dataset.jsonl")
        log.info("staged qa dataset")
    if cfg.splits_dir and cfg.splits_dir.exists():
        shutil.copytree(cfg.splits_dir, staging / "splits")
        log.info("staged topic splits")

    (staging / "README.md").write_text(
        DATA_CARD.format(license=cfg.license), encoding="utf-8",
    )
    return staging


def publish_dataset(cfg: PublishConfig) -> str:
    """Stage the repo and upload it to HuggingFace. Returns the hf:// URI."""
    from data_platform.storage.hugging_face import HuggingFaceStorage

    staging = stage_repo(cfg)
    storage = HuggingFaceStorage.from_env()
    return storage.save_dataset(
        local_dir=staging,
        dataset_id=cfg.repo_id,
        private=cfg.private,
        create_repo_if_missing=True,
        commit_message=cfg.commit_message,
    )


__all__ = ["PublishConfig", "publish_dataset", "stage_repo", "DATA_CARD"]
