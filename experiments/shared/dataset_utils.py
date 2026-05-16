"""Dataset helpers for the mental-health experiments.

The canonical record schema used across all experiments::

    {
      "instruction": str,        # user's question / message
      "input":       str,        # extra context, typically ""
      "output":      str,        # counselor / model response
      "category":    str,        # anxiety|depression|stress|grief|relationships|self-esteem|out_of_domain
      "difficulty":  str,        # mild|moderate|severe
      "source":      str,        # which dataset this record came from
      "quality_score": float,    # 0.0 - 1.0
      "is_refusal":  bool,       # True for out-of-domain / "I don't know" examples
      "language":    str,        # "en" or "bg"
      "token_count": int         # rough token estimate (~chars / 4)
    }

Every record produced by ``datasets/prepare_mental_health.py`` adheres
to this schema; downstream training code can rely on it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


CATEGORIES = (
    "anxiety", "depression", "stress", "grief",
    "relationships", "self-esteem", "out_of_domain",
)
DIFFICULTIES = ("mild", "moderate", "severe")
LANGUAGES = ("en", "bg")


# Common red-flag phrases that should never appear in training outputs.
# Crude string-level filter — paired with a slower LLM-side review where
# higher precision is needed.
DANGEROUS_PHRASES = (
    "kill yourself", "you should die", "harm yourself",
    "убий се", "наранявай се",
)


def estimate_token_count(text: str) -> int:
    """Cheap token estimate: chars / 4 (rough Llama-tokenizer heuristic)."""
    return max(1, len(text) // 4)


def is_dangerous(text: str) -> bool:
    """Return True if ``text`` contains a known-harmful phrase."""
    lc = text.lower()
    return any(phrase in lc for phrase in DANGEROUS_PHRASES)


def is_short(text: str, min_words: int = 30) -> bool:
    """Return True if ``text`` has fewer than ``min_words`` whitespace tokens."""
    return len(text.split()) < min_words


def make_record(
    *,
    instruction: str,
    output: str,
    source: str,
    category: str = "out_of_domain",
    difficulty: str = "moderate",
    input_: str = "",
    quality_score: float = 0.7,
    is_refusal: bool = False,
    language: str = "en",
) -> dict[str, Any]:
    """Build a single canonical record. Defaults are tuned for safety."""
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
    return {
        "instruction": instruction,
        "input": input_,
        "output": output,
        "category": category,
        "difficulty": difficulty,
        "source": source,
        "quality_score": float(max(0.0, min(1.0, quality_score))),
        "is_refusal": bool(is_refusal),
        "language": language,
        "token_count": estimate_token_count(instruction) + estimate_token_count(output),
    }


def quality_filter(
    record: dict[str, Any],
    *,
    min_words: int = 30,
    drop_dangerous: bool = True,
) -> bool:
    """Return True when the record should be **kept**."""
    output = record.get("output") or ""
    if is_short(output, min_words=min_words):
        return False
    if drop_dangerous and is_dangerous(output):
        return False
    if drop_dangerous and is_dangerous(record.get("instruction") or ""):
        return False
    return True


def load_alpaca_dataset(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file (canonical Alpaca-style)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    with file_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_alpaca_dataset(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write records as JSONL. Returns the number written."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with file_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def stratified_split(
    records: list[dict[str, Any]],
    *,
    eval_ratio: float = 0.1,
    stratify_by: str = "category",
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified train/eval split keyed by ``stratify_by`` (default: category).

    Falls back to a random split when the stratify field is missing.
    """
    import random

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        key = r.get(stratify_by, "_unknown")
        buckets.setdefault(key, []).append(r)

    train: list[dict[str, Any]] = []
    eval_: list[dict[str, Any]] = []
    for items in buckets.values():
        rng.shuffle(items)
        n_eval = max(1, int(round(len(items) * eval_ratio))) if len(items) > 1 else 0
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def dataset_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary stats useful as MLflow params/metrics."""
    if not records:
        return {"count": 0}
    cat = Counter(r.get("category", "?") for r in records)
    diff = Counter(r.get("difficulty", "?") for r in records)
    src = Counter(r.get("source", "?") for r in records)
    lang = Counter(r.get("language", "?") for r in records)
    refusal = sum(1 for r in records if r.get("is_refusal"))
    tokens = [r.get("token_count", 0) for r in records]
    qs = [r.get("quality_score", 0.0) for r in records]
    return {
        "count": len(records),
        "refusal_count": refusal,
        "refusal_ratio": round(refusal / len(records), 4),
        "by_category": dict(cat),
        "by_difficulty": dict(diff),
        "by_source": dict(src),
        "by_language": dict(lang),
        "token_count_mean": round(sum(tokens) / len(tokens), 2),
        "token_count_max": max(tokens) if tokens else 0,
        "quality_score_mean": round(sum(qs) / len(qs), 4),
    }


def alpaca_to_prompt(record: dict[str, Any], *, eos_token: str | None = None) -> str:
    """Render an Alpaca-style record into a single training string.

    The canonical Alpaca template — used at fine-tuning time:

        Below is an instruction that describes a task. Write a response
        that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}{eos}
    """
    parts = [
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.",
        "",
        "### Instruction:",
        record["instruction"].strip(),
    ]
    if record.get("input"):
        parts += ["", "### Input:", record["input"].strip()]
    parts += [
        "",
        "### Response:",
        record["output"].strip() + (eos_token or ""),
    ]
    return "\n".join(parts)
