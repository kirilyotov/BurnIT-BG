"""Generate a *real* Bulgarian question->supportive-answer dataset.

The plain ``from-passages`` builder pairs a generic question template with a
raw book passage, so the instruction and output aren't truly aligned — it's
style / continued-pretraining data. This builder instead asks an LLM
(Mistral Large 3, hosted on NVIDIA) to read each passage and produce a
genuine peer-support exchange in Bulgarian:

* ``instruction`` — a realistic first-person message from someone seeking
  support on the passage's topic;
* ``output`` — a warm, non-clinical, supportive reply grounded in the
  passage's ideas.

Output records use the same canonical Alpaca schema as
:mod:`data_transformation.build_dataset`, so both datasets are
interchangeable downstream. Results are cached by ``passage_id`` so a
re-run (or a crash) doesn't re-pay for passages already generated.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from data_platform.llm import NvidiaChatClient, NvidiaChatError

from .build_dataset import TOPIC_TO_CATEGORY, _iter_passages

log = logging.getLogger(__name__)


# Peer-support, Bulgarian, non-clinical. The model returns strict JSON.
QA_SYSTEM_PROMPT = (
    "Ти създаваш висококачествени обучаващи двойки въпрос–отговор на "
    "български език за AI асистент за емоционална ВРЪСТНИЧЕСКА подкрепа "
    "(не клинична, не медицинска).\n\n"
    "Получаваш тема и кратък откъс от книга. Създай:\n"
    "1. \"question\": реалистично, естествено звучащо съобщение от първо "
    "лице от човек, който търси подкрепа по тази тема (на български).\n"
    "2. \"answer\": топъл, съпричастен и подкрепящ отговор (на български), "
    "вдъхновен от идеите в откъса. Отговорът трябва да валидира чувствата, "
    "да предлага меки практични насоки, да НЕ поставя диагнози и да НЕ дава "
    "медицински съвети. При сериозен риск (напр. мисли за самонараняване) "
    "насърчи човека да потърси професионална помощ или кризисна линия.\n\n"
    "Върни САМО валиден JSON обект с точно тези два ключа: "
    '{"question": "...", "answer": "..."}'
)


@dataclass
class QABuildConfig:
    """Run-time settings for :func:`build_qa_dataset`."""

    input_path: Path
    output_path: Path
    model: str = "mistral-large-3"
    limit: Optional[int] = None
    min_words: int = 12
    max_passage_chars: int = 1200
    delay: float = 0.0
    temperature: float = 0.6
    max_tokens: int = 1024
    cache_path: Optional[Path] = None
    quality_score: float = 0.85
    extraction_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))


def _user_message(topic: str, passage_text: str) -> str:
    return (
        f"Тема: {topic or 'обща емоционална подкрепа'}\n\n"
        f"Откъс:\n{passage_text}\n\n"
        "Създай двойката въпрос–отговор като JSON."
    )


def _load_cache(path: Optional[Path]) -> dict[str, dict]:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("could not read QA cache at %s; starting fresh", path)
    return {}


def _save_cache(path: Optional[Path], cache: dict[str, dict]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _generate_one(
    client: NvidiaChatClient, passage: dict, cfg: QABuildConfig,
) -> Optional[dict]:
    """Return a canonical Alpaca record for one passage, or None on failure."""
    from experiments.shared.dataset_utils import (
        estimate_token_count,
        make_record,
        quality_filter,
    )

    text = (passage.get("text") or "").strip()
    if not text:
        return None
    if len(text) > cfg.max_passage_chars:
        text = text[: cfg.max_passage_chars].rsplit(" ", 1)[0]

    topic = passage.get("topic", "")
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(topic, text)},
    ]
    try:
        obj = client.chat_json(
            messages, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        )
    except NvidiaChatError as exc:
        log.warning("generation failed for %s: %s", passage.get("passage_id"), exc)
        return None

    question = (obj.get("question") or "").strip() if isinstance(obj, dict) else ""
    answer = (obj.get("answer") or "").strip() if isinstance(obj, dict) else ""
    if not question or not answer:
        return None

    category = TOPIC_TO_CATEGORY.get(topic, "out_of_domain")
    if category not in (
        "anxiety", "depression", "stress", "grief",
        "relationships", "self-esteem", "out_of_domain",
    ):
        category = "out_of_domain"

    try:
        rec = make_record(
            instruction=question,
            output=answer,
            source=f"chitanka-qa:{passage.get('book_id', '?')}",
            category=category,
            difficulty="moderate",
            quality_score=cfg.quality_score,
            is_refusal=False,
            language="bg",
        )
    except ValueError:
        return None

    if not quality_filter(rec, min_words=cfg.min_words, drop_dangerous=True):
        return None

    rec["metadata"] = {
        "passage_id": passage.get("passage_id"),
        "book_title": passage.get("book_title"),
        "authors": passage.get("authors"),
        "topic": topic,
        "generator": client.model.model_id,
        "synthetic": True,
    }
    rec["token_count"] = estimate_token_count(rec["instruction"]) + estimate_token_count(rec["output"])
    return rec


def _iter_limited(path: Path, limit: Optional[int]) -> Iterator[dict]:
    for i, passage in enumerate(_iter_passages(path)):
        if limit is not None and i >= limit:
            return
        yield passage


def build_qa_dataset(cfg: QABuildConfig) -> tuple[int, int]:
    """Generate the Q->A JSONL. Returns ``(total_in, total_out)``."""
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"input passages JSONL not found: {cfg.input_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    client = NvidiaChatClient(cfg.model)
    cache = _load_cache(cfg.cache_path)

    total_in = 0
    total_out = 0
    cache_dirty = 0
    with cfg.output_path.open("w", encoding="utf-8") as out:
        for passage in _iter_limited(cfg.input_path, cfg.limit):
            total_in += 1
            pid = passage.get("passage_id") or f"_idx{total_in}"

            rec = cache.get(pid)
            if rec is None:
                rec = _generate_one(client, passage, cfg)
                if rec is not None:
                    cache[pid] = rec
                    cache_dirty += 1
                if cfg.delay:
                    time.sleep(cfg.delay)
                if cache_dirty and cache_dirty % 25 == 0:
                    _save_cache(cfg.cache_path, cache)

            if rec is None:
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_out += 1
            if total_out % 25 == 0:
                log.info("  generated %d / %d passages", total_out, total_in)

    _save_cache(cfg.cache_path, cache)
    return total_in, total_out


__all__ = ["QABuildConfig", "build_qa_dataset", "QA_SYSTEM_PROMPT"]
