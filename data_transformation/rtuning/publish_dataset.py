"""Push a curated BG R-Tuning dataset to MinIO + (optionally) HF Hub.

Each curated dataset (``triviaqa-bg``, ``squadv2-bg``, ``combined-bg``) gets
its own staging dir with the jsonl + a Bulgarian/English README data card +
license info reflecting both upstream attribution AND the BurnIT-BG context
(out-of-domain refusal training).

Push is gated by env / config — defaults match the BurnIT-BG standing rule:
stage to MinIO, do NOT push to HF without explicit "go".
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .download_raw import SOURCE_SPECS

log = logging.getLogger(__name__)


@dataclass
class PublishRTuningConfig:
    """Settings for :func:`publish_rtuning`."""

    dataset_jsonl: Path
    minio_prefix: str
    hf_repo_id: str | None = None
    sources: list[str] = field(default_factory=list)   # e.g. ["triviaqa"] or ["triviaqa","squadv2"]
    push_minio: bool = True
    push_hf: bool = False
    private: bool = False
    minio_bucket: str | None = None


def publish_rtuning(cfg: PublishRTuningConfig) -> dict[str, str | None]:
    """Stage the curated jsonl + data card, push to MinIO + HF.

    Returns ``{"minio": <s3 uri or None>, "hf": <hf:// uri or None>}``.
    """
    if not cfg.dataset_jsonl.exists():
        raise FileNotFoundError(f"curated jsonl missing: {cfg.dataset_jsonl}")

    staging = cfg.dataset_jsonl.parent / f"_stage-{cfg.dataset_jsonl.stem}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    shutil.copy2(cfg.dataset_jsonl, staging / "dataset.jsonl")
    (staging / "README.md").write_text(_curated_readme(cfg), encoding="utf-8")

    out: dict[str, str | None] = {"minio": None, "hf": None}

    if cfg.push_minio:
        from data_platform.storage.minio import MinioStorage
        storage = MinioStorage.from_env()
        bucket = cfg.minio_bucket or storage.bucket
        storage.save_directory(staging, cfg.minio_prefix, bucket=bucket)
        out["minio"] = f"s3://{bucket}/{cfg.minio_prefix}"
        log.info("uploaded curated -> %s", out["minio"])

    if cfg.push_hf and cfg.hf_repo_id:
        from huggingface_hub import DatasetCardData
        from data_platform.storage.hugging_face import HuggingFaceStorage

        storage = HuggingFaceStorage.from_env()
        # Combine upstream licenses — most permissive of the inputs; for the
        # combined dataset we surface "mixed (see README)" so consumers check.
        license_tag = _license_for(cfg.sources)
        card_data = DatasetCardData(
            license=license_tag,
            language=["bg"],
            task_categories=["text-generation"],
            tags=["mental-health", "bulgarian", "r-tuning", "out-of-domain", "refusal"],
            pretty_name=f"BurnIT-BG R-Tuning ({', '.join(cfg.sources) or 'combined'}) — BG",
        )
        out["hf"] = storage.save_dataset(
            local_dir=staging,
            dataset_id=cfg.hf_repo_id,
            private=cfg.private,
            commit_message=f"Publish BurnIT-BG R-Tuning curated ({', '.join(cfg.sources) or 'combined'})",
            card_data=card_data,
            card_content=_curated_readme(cfg),
        )
        log.info("uploaded curated -> %s", out["hf"])

    return out


def _license_for(sources: list[str]) -> str:
    licenses = {SOURCE_SPECS[s]["license"] for s in sources if s in SOURCE_SPECS}
    if len(licenses) == 1:
        return licenses.pop()
    return "other"


def _curated_readme(cfg: PublishRTuningConfig) -> str:
    sources_md = ""
    for s in cfg.sources:
        if s in SOURCE_SPECS:
            spec = SOURCE_SPECS[s]
            sources_md += (
                f"- **{s}** — translated from "
                f"[`{spec['repo_id']}`](https://huggingface.co/datasets/{spec['repo_id']}) "
                f"({spec['config']}, `{spec['license']}`). {spec['citation']}\n"
            )
    if not sources_md:
        sources_md = "- (combined dataset — see source-specific records)\n"

    return f"""# BurnIT-BG — R-Tuning (out-of-domain refusal) — Bulgarian

Тренировъчен набор от данни на български език за **обучение в отказ**:
учи моделa BurnIT-BG да каже *"Не знам"* на въпроси извън неговия обхват
(психично здраве и емоционална подкрепа).

*Bulgarian training set for **out-of-domain refusal training**: teaches
the BurnIT-BG model to say "I don't know" on questions outside its scope
(mental health and emotional support).*

## ⚠️ Disclaimer

This dataset is for **research purposes only**. The BurnIT-BG model trained
on it is **not medical advice** and does not replace professional care.
If you or someone you know is in crisis, seek professional help immediately.

## Method

This implements the **R-Tuning-R** variant from
[Hou et al. 2024](https://arxiv.org/abs/2311.09677). Every record's
``output`` is a Bulgarian refusal (rotated among three templates for
diversity) instead of the upstream answer. The model learns: when the
question is out-of-domain, refuse politely.

## Schema

Alpaca format:

```json
{{
  "instruction": "<въпрос на български>",
  "input": "",
  "output": "<отказ на български>",
  "category": "out_of_domain",
  "language": "bg",
  "metadata": {{"source": "...", "ood": true, ...}}
}}
```

## Sources

{sources_md}
Translation: Google Translate (via `deep_translator`), cached and deterministic.

## Refusal templates

The ``output`` field rotates among:

1. *"Не съм сигурен. Този въпрос е извън моя обхват."*
2. *"Не знам отговора на този въпрос. Аз съм асистент за психично здраве."*
3. *"Не мога да отговоря. Моят обхват е психично здраве и емоционална подкрепа."*

## License

See `LICENSE` of each upstream source. The translated text inherits the
permissions of the original.
"""


__all__ = ["PublishRTuningConfig", "publish_rtuning"]
