"""Assemble translated rows into Alpaca-style R-Tuning records.

We're implementing the R-Tuning-R variant from Hou et al. 2024
(https://arxiv.org/abs/2311.09677): for every out-of-domain question (all
TriviaQA + SQuAD v2 rows from our perspective, since BurnIT-BG's domain is
mental health), the target output is a Bulgarian refusal — teaching the
model to say *"I don't know"* when asked something outside its scope.

We rotate among three refusal templates so the model doesn't lock onto a
single fixed phrase. The original EN answer is kept in metadata for
post-hoc analysis but never used as the training target.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


# Three Bulgarian refusal templates — rotated round-robin per row.
# Picked by user 2026-06-02. Order kept stable so behavior is reproducible.
REFUSAL_TEMPLATES_BG: tuple[str, ...] = (
    "Не съм сигурен. Този въпрос е извън моя обхват.",
    "Не знам отговора на този въпрос. Аз съм асистент за психично здраве.",
    "Не мога да отговоря. Моят обхват е психично здраве и емоционална подкрепа.",
)


@dataclass
class BuildRTuningConfig:
    """Settings for :func:`build_rtuning`."""

    input_jsonl: Path
    output_jsonl: Path
    refusal_templates: tuple[str, ...] = REFUSAL_TEMPLATES_BG
    rotate: str = "round-robin"  # "round-robin" | "random"
    seed: int = 42


def build_rtuning(cfg: BuildRTuningConfig) -> dict[str, int]:
    """Stream translated input -> Alpaca-style R-Tuning output.

    Each output record:

    .. code-block:: json

        {
          "instruction": "<question in BG>",
          "input": "",
          "output": "<rotated BG refusal>",
          "category": "out_of_domain",
          "language": "bg",
          "metadata": {
            "source": "triviaqa",
            "source_id": "...",
            "english_question": "...",
            "english_answer": "...",
            "ood": true,
            "unanswerable": false,
            "refusal_template_idx": 0
          }
        }

    Returns ``{"total_in": N, "written": K, "skipped_no_question": S}``.
    """
    if not cfg.input_jsonl.exists():
        raise FileNotFoundError(f"input not found: {cfg.input_jsonl}")

    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    rotate_idx = 0
    total_in = written = skipped = 0

    with cfg.output_jsonl.open("w", encoding="utf-8") as out_fh:
        for record in _stream_records(cfg.input_jsonl):
            total_in += 1
            q_bg = (record.get("question_bg") or "").strip()
            if not q_bg:
                skipped += 1
                continue

            if cfg.rotate == "random":
                template_idx = rng.randrange(len(cfg.refusal_templates))
            else:
                template_idx = rotate_idx % len(cfg.refusal_templates)
                rotate_idx += 1
            refusal = cfg.refusal_templates[template_idx]

            out_record = {
                "instruction": q_bg,
                "input": "",
                "output": refusal,
                "category": "out_of_domain",
                "language": "bg",
                "metadata": {
                    "source": record.get("source"),
                    "source_id": record.get("source_id"),
                    "english_question": record.get("question"),
                    "english_answer": record.get("answer"),
                    "english_answer_bg": record.get("answer_bg"),
                    "ood": True,
                    "unanswerable": bool(record.get("unanswerable", False)),
                    "refusal_template_idx": template_idx,
                },
            }
            out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            written += 1

    log.info("build_rtuning: total=%d written=%d skipped_no_question=%d",
             total_in, written, skipped)
    return {"total_in": total_in, "written": written, "skipped_no_question": skipped}


def _stream_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


__all__ = ["BuildRTuningConfig", "REFUSAL_TEMPLATES_BG", "build_rtuning"]
