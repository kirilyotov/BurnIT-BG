"""Download raw TriviaQA / SQuAD v2 to local jsonl for the R-Tuning pipeline.

Pulls just the columns we need (``question`` + ``answer.value`` for TriviaQA,
``question`` + ``answers.text`` for SQuAD v2). The full datasets are heavy
because of context fields — we drop those on the way in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)


SourceName = Literal["triviaqa", "squadv2"]


# Upstream HF repos + the lightest config for each.
SOURCE_SPECS: dict[str, dict[str, str]] = {
    "triviaqa": {
        "repo_id": "mandarjoshi/trivia_qa",
        "config": "unfiltered.nocontext",
        "license": "apache-2.0",
        "homepage": "https://nlp.cs.washington.edu/triviaqa/",
        "citation": "Joshi et al. 2017 — TriviaQA: A Large Scale Distantly Supervised "
                    "Challenge Dataset for Reading Comprehension. ACL.",
    },
    "squadv2": {
        "repo_id": "rajpurkar/squad_v2",
        "config": "squad_v2",
        "license": "cc-by-sa-4.0",
        "homepage": "https://rajpurkar.github.io/SQuAD-explorer/",
        "citation": "Rajpurkar et al. 2018 — Know What You Don't Know: Unanswerable "
                    "Questions for SQuAD. ACL.",
    },
}


@dataclass
class DownloadRawConfig:
    """Settings for :func:`download_raw`."""

    source: SourceName
    output_dir: Path
    split: str = "train"
    limit: int | None = None
    revision: str | None = None


def download_raw(cfg: DownloadRawConfig) -> Path:
    """Stream the chosen split into ``{output_dir}/{source}-{split}.jsonl``.

    Only the columns we care about end up on disk — everything else is dropped
    to keep the raw artifact small.
    """
    from datasets import load_dataset

    spec = SOURCE_SPECS[cfg.source]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"{cfg.source}-{cfg.split}.jsonl"

    log.info("downloading %s/%s split=%s -> %s",
             spec["repo_id"], spec["config"], cfg.split, out_path)

    ds = load_dataset(
        spec["repo_id"],
        name=spec["config"],
        split=cfg.split,
        revision=cfg.revision,
    )

    if cfg.limit is not None:
        ds = ds.select(range(min(cfg.limit, len(ds))))

    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            record = _slim_row(cfg.source, row)
            if record is None:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    log.info("%s: wrote %d records (%.1f MB)",
             cfg.source, n_written, out_path.stat().st_size / 1e6)

    _write_license_file(cfg.output_dir, cfg.source, spec)
    return out_path


def _slim_row(source: str, row: dict) -> dict | None:
    """Keep only the columns the R-Tuning pipeline needs."""
    if source == "triviaqa":
        question = (row.get("question") or "").strip()
        answer_obj = row.get("answer") or {}
        answer = (answer_obj.get("value") or "").strip()
        if not question:
            return None
        return {
            "source": "triviaqa",
            "source_id": row.get("question_id"),
            "question": question,
            "answer": answer,
            "answer_aliases": answer_obj.get("aliases", []),
        }

    if source == "squadv2":
        question = (row.get("question") or "").strip()
        answers_obj = row.get("answers") or {}
        texts = answers_obj.get("text") or []
        answer = (texts[0].strip() if texts else "")
        if not question:
            return None
        return {
            "source": "squadv2",
            "source_id": row.get("id"),
            "question": question,
            "answer": answer,
            "unanswerable": len(texts) == 0,
        }

    raise ValueError(f"unknown source: {source}")


def _write_license_file(output_dir: Path, source: str, spec: dict) -> None:
    """Drop a small ``LICENSE.txt`` next to the raw jsonl with upstream attribution."""
    text = (
        f"Source: {spec['repo_id']} (config: {spec['config']})\n"
        f"Homepage: {spec['homepage']}\n"
        f"License: {spec['license']}\n"
        f"Citation: {spec['citation']}\n"
    )
    (output_dir / "LICENSE.txt").write_text(text, encoding="utf-8")


__all__ = ["DownloadRawConfig", "SOURCE_SPECS", "download_raw"]
