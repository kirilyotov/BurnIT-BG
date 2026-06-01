"""Merge downloaded mental-health datasets into a unified Alpaca-format set.

What this script does (high level):

1. Walks every ``*.jsonl`` under ``--input_dir`` (default ``data_prep/raw/``).
2. Coerces each record into the canonical schema declared in
   :mod:`experiments.shared.dataset_utils`.
3. Applies quality filters: short outputs dropped, dangerous content
   dropped, low-quality flagged.
4. Adds N% explicit refusal examples (out-of-domain / "I don't know"
   responses) so R-Tuning has data to learn from.
5. Stratified 90/10 train/eval split by ``category``.
6. Writes ``train.jsonl`` + ``eval.jsonl`` under ``--output_dir`` and
   logs the dataset statistics to MLflow.

Usage::

    python -m data_prep.prepare_mental_health \
        --input_dir data_prep/raw \
        --output_dir data_prep/processed \
        --refusal_ratio 0.18
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

from experiments.shared.dataset_utils import (
    CATEGORIES,
    dataset_statistics,
    estimate_token_count,
    make_record,
    quality_filter,
    stratified_split,
    write_alpaca_dataset,
)

from ._common import iter_jsonl, setup_tracking, timestamp_run_name


# Sources where we already have an Alpaca-ish schema we can map fields from.
FIELD_ALIASES = {
    "instruction": ("instruction", "Context", "context", "question", "prompt", "input", "user"),
    "output":      ("output", "Response", "response", "answer", "completion", "assistant", "reply"),
    "input":       ("input", "additional_context", "extra"),
    "category":    ("category", "topic", "label"),
    "difficulty":  ("difficulty", "severity"),
    "language":    ("language", "lang"),
}


def _first_field(record: dict[str, Any], names: tuple[str, ...]) -> str:
    for n in names:
        if record.get(n):
            return str(record[n]).strip()
    return ""


def _guess_category(text: str) -> str:
    """Best-effort keyword classification — better than always ``out_of_domain``."""
    t = text.lower()
    if any(k in t for k in ("anxiety", "panic", "тревож", "паник")):
        return "anxiety"
    if any(k in t for k in ("depress", "тъга", "тъжен", "депрес")):
        return "depression"
    if any(k in t for k in ("stress", "стрес", "burnout", "изгар")):
        return "stress"
    if any(k in t for k in ("grief", "loss", "загуба", "скърб")):
        return "grief"
    if any(k in t for k in ("relationship", "приятел", "партньор", "семей")):
        return "relationships"
    if any(k in t for k in ("self-esteem", "confidence", "самочувств", "увер")):
        return "self-esteem"
    return "out_of_domain"


def _normalize(record: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    """Coerce a free-form record into the canonical schema, or None to drop."""
    instruction = _first_field(record, FIELD_ALIASES["instruction"])
    output = _first_field(record, FIELD_ALIASES["output"])
    if not instruction or not output:
        return None

    input_ = _first_field(record, FIELD_ALIASES["input"])
    category = _first_field(record, FIELD_ALIASES["category"]) or _guess_category(instruction)
    if category not in CATEGORIES:
        category = _guess_category(instruction)
    difficulty = _first_field(record, FIELD_ALIASES["difficulty"]) or "moderate"
    if difficulty not in ("mild", "moderate", "severe"):
        difficulty = "moderate"
    language = _first_field(record, FIELD_ALIASES["language"]) or (
        "bg" if any(ord(c) > 127 for c in instruction[:80]) else "en"
    )
    if language not in ("en", "bg"):
        language = "en"

    try:
        return make_record(
            instruction=instruction,
            output=output,
            source=source,
            category=category,
            difficulty=difficulty,
            input_=input_,
            language=language,
        )
    except ValueError:
        return None


REFUSAL_PROMPTS_OOD = [
    ("Кой ще спечели следващите световни първенства?",
     "Не съм запознат с прогнози за спортни събития — фокусирам се върху ментално здраве. "
     "Ако имаш нещо, което те притеснява емоционално, тук съм да чуя."),
    ("Напиши ми Python код за уеб скрейпинг.",
     "Това е извън моята област. Аз помагам с разговори за ментално здраве. "
     "За кодиране препоръчвам да потърсиш специализиран ресурс."),
    ("Каква е рецептата за баница?",
     "Кулинарията не е моята област. Аз съм тук за подкрепа на ментално здраве — "
     "ако нещо те тревожи, можем да поговорим."),
    ("Обясни ми блокчейн.",
     "Технологичните теми са извън моя обхват. Ако нещо те натоварва емоционално, "
     "разкажи ми."),
    ("Кой е президентът на САЩ?",
     "Не следя политически новини. Тук съм да помогна с теми за ментално здраве — "
     "как се чувстваш днес?"),
    ("Колко тежи Луната?",
     "Това е извън моята експертиза. Аз помагам с разговори за ментално здраве."),
]


def _build_refusal_examples(target_count: int, language: str = "bg") -> list[dict[str, Any]]:
    """Synthesize is_refusal=True records to mix into the train set."""
    records: list[dict[str, Any]] = []
    i = 0
    while len(records) < target_count:
        prompt, response = REFUSAL_PROMPTS_OOD[i % len(REFUSAL_PROMPTS_OOD)]
        records.append(make_record(
            instruction=prompt,
            output=response,
            source="synthetic_refusals",
            category="out_of_domain",
            difficulty="mild",
            quality_score=0.85,
            is_refusal=True,
            language=language,
        ))
        i += 1
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the unified mental-health dataset.")
    p.add_argument("--input_dir", default="data_prep/raw",
                   help="Directory holding raw JSONL files (recursively scanned).")
    p.add_argument("--output_dir", default="data_prep/processed",
                   help="Output directory for train.jsonl / eval.jsonl.")
    p.add_argument("--refusal_ratio", type=float, default=0.18,
                   help="Fraction of the FINAL training set that is_refusal=True (default 0.18).")
    p.add_argument("--eval_ratio", type=float, default=0.10,
                   help="Eval split ratio (default 0.10).")
    p.add_argument("--min_words", type=int, default=30,
                   help="Drop responses shorter than this many words (default 30).")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        print(f"error: input dir not found: {in_dir}", file=sys.stderr)
        return 1

    print(f"[prepare] scanning {in_dir} for *.jsonl")
    raw_files = sorted(in_dir.rglob("*.jsonl"))
    if not raw_files:
        print("error: no .jsonl files found.", file=sys.stderr)
        return 1

    normalized: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    for path in raw_files:
        source = str(path.relative_to(in_dir).parent) if path.parent != in_dir else path.stem
        kept = 0
        seen = 0
        for record in iter_jsonl(path):
            seen += 1
            r = _normalize(record, source=source)
            if r is None:
                continue
            if not quality_filter(r, min_words=args.min_words):
                continue
            normalized.append(r)
            kept += 1
        per_source[source] = kept
        print(f"  {path}: {kept}/{seen} kept")

    if not normalized:
        print("error: every record was filtered out.", file=sys.stderr)
        return 1

    # Add refusal examples — count is computed against the *final* set so
    # the ratio is correct after mixing.
    n_real = len(normalized)
    target_refusals = max(0, int(round(n_real * args.refusal_ratio / max(1 - args.refusal_ratio, 1e-6))))
    refusals = _build_refusal_examples(target_refusals)
    normalized.extend(refusals)
    print(f"[prepare] added {len(refusals)} synthetic refusal examples")

    train, eval_ = stratified_split(normalized, eval_ratio=args.eval_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = write_alpaca_dataset(train, out_dir / "train.jsonl")
    n_eval = write_alpaca_dataset(eval_, out_dir / "eval.jsonl")
    print(f"[prepare] wrote {n_train} train + {n_eval} eval rows to {out_dir}")

    stats_train = dataset_statistics(train)
    stats_eval = dataset_statistics(eval_)
    print(f"[prepare] train stats: {stats_train}")
    print(f"[prepare] eval  stats: {stats_eval}")

    if args.no_mlflow:
        return 0

    try:
        tracking = setup_tracking("dataset_prepare")
        with tracking.run(
            run_name=timestamp_run_name("prepare-mental-health"),
            tags={"stage": "dataset_prepare"},
            with_hardware=False,
            log_system_metrics=False,
        ):
            tracking.log_params({
                "refusal_ratio_target": args.refusal_ratio,
                "eval_ratio": args.eval_ratio,
                "min_words": args.min_words,
                "input_dir": str(in_dir),
            })
            tracking.log_metrics({
                "n_train": float(n_train),
                "n_eval": float(n_eval),
                "n_refusals_added": float(len(refusals)),
                "n_raw_records": float(sum(per_source.values())),
                "train_refusal_ratio": float(stats_train.get("refusal_ratio", 0.0)),
                "train_token_count_mean": float(stats_train.get("token_count_mean", 0.0)),
                "train_quality_score_mean": float(stats_train.get("quality_score_mean", 0.0)),
            })
            tracking.save_data(out_dir / "train.jsonl", artifact_path="processed")
            tracking.save_data(out_dir / "eval.jsonl", artifact_path="processed")
    except Exception as exc:
        print(f"[prepare] mlflow logging failed (continuing): {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
