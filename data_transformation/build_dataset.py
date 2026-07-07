"""Turn extracted Chitanka passages into an Alpaca-style mental-health dataset.

The input is the JSONL produced by ``data_scraping extract-passages`` —
each line is one matched book passage with its topic and source book.
This step pairs every passage with a Bulgarian instruction template
chosen from the topic-specific bank below, producing canonical Alpaca
records (compatible with :mod:`experiments.shared.dataset_utils`)::

    {
      "instruction": "<topic-appropriate question>",
      "input": "",
      "output": "<the book passage>",
      "category": "<our 7-category mental-health label>",
      "difficulty": "moderate",
      "source": "chitanka:<book_id>",
      "quality_score": 0.8,
      "is_refusal": false,
      "language": "bg",
      "token_count": <int>,
      "metadata": {
        "passage_id": "...",
        "book_title": "...",
        "authors": [...],
        "topic": "<fine-grained topic>",
        "keywords_matched": [...]
      }
    }

The resulting JSONL is the input to
``data_prep.prepare_mental_health`` (which mixes it with other sources,
filters and stratified-splits into train/eval).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)


# Fine-grained topic (from data_scraping.topics_mental_health) → canonical
# 7-category label used by the rest of the training pipeline.
TOPIC_TO_CATEGORY: dict[str, str] = {
    "anxiety":        "anxiety",
    "fear":           "anxiety",
    "depression":     "depression",
    "grief":          "depression",
    "stress":         "stress",
    "anger":          "stress",
    "sleep":          "stress",
    "self_esteem":    "self-esteem",
    "self_compassion": "self-esteem",
    "self_awareness": "self-esteem",
    "growth":         "self-esteem",
    "relationships":  "relationships",
    "boundaries":     "relationships",
    "communication":  "relationships",
    "loneliness":     "relationships",
    "parenting":      "relationships",
    "addiction":      "out_of_domain",   # handled separately downstream
    "trauma":         "out_of_domain",
    # everything else is general wellness — map to a sensible default
    "mindfulness":    "self-esteem",
    "resilience":     "self-esteem",
    "habits":         "self-esteem",
    "motivation":     "self-esteem",
    "happiness":      "self-esteem",
}


# Bulgarian instruction templates per fine-grained topic. Each list is a
# pool we sample from with a stable seed so re-runs are reproducible.
# Phrasing is peer-support — never clinical advice.
INSTRUCTION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "anxiety": (
        "Изпитвам силна тревожност в последно време. Какво може да помогне?",
        "Често ме обзема безпокойство. Как да се справя?",
        "Имам паник атаки. Можеш ли да ми кажеш какво да правя в момента?",
        "Как да успокоя ума си, когато съм много напрегнат/а?",
        "Страхувам се без ясна причина. Какво се случва с мен?",
    ),
    "depression": (
        "Чувствам се много тъжен/тъжна и без енергия. Какво да направя?",
        "Имам усещане, че нищо няма смисъл. Как да си върна желанието за живот?",
        "Чувствам апатия и безнадеждност. Можеш ли да ми помогнеш?",
        "Не ми се става сутрин и не виждам радост в нещата. Какво да направя?",
        "Чувствам се празен/празна отвътре. Как да изляза от това?",
    ),
    "stress": (
        "Изгарям на работа и не мога да си почина. Какво да направя?",
        "Имам твърде много задачи и не знам откъде да започна. Помогни ми.",
        "Постоянно съм в напрежение. Как да намеря баланс?",
        "Усещам, че издишам. Как да се възстановя?",
        "Тялото ми е изтощено от стрес. Какво да направя?",
    ),
    "anger": (
        "Често се ядосвам и съжалявам по-късно. Как да овладея гнева?",
        "Реагирам остро и не мога да се контролирам. Какво да правя?",
        "Близките ме изваждат от равновесие. Как да реагирам по-спокойно?",
    ),
    "grief": (
        "Загубих близък човек и не знам как да продължа. Какво да направя?",
        "Скръбя от много време. Как да приема загубата?",
        "Не мога да приема, че този, когото обичах, си отиде. Помогни ми.",
    ),
    "self_esteem": (
        "Имам ниско самочувствие. Как да го изградя?",
        "Не вярвам в себе си. От къде да започна?",
        "Постоянно се сравнявам с другите и излизам по-лош/а. Какво да направя?",
        "Как да приема себе си такъв/такава, какъвто/каквато съм?",
    ),
    "self_compassion": (
        "Винаги съм твърд/а със себе си. Как да бъда по-мил/а?",
        "Не успявам да си простя за нещо, което съм направил/а. Какво да правя?",
        "Как да практикувам грижа към себе си?",
    ),
    "self_awareness": (
        "Искам да познавам себе си по-добре. От къде да започна?",
        "Имам много мисли и не мога да ги подредя. Как да си изясня?",
        "Как да разпозная истинските си нужди?",
    ),
    "growth": (
        "Искам да се развивам като личност. С какво да започна?",
        "Чувствам, че съм заседнал/а. Как да продължа напред?",
        "Какво помага за личностен растеж?",
    ),
    "relationships": (
        "Имам трудности във връзката си. Какво да направя?",
        "Отдалечих се от близък човек. Как да възстановим връзката?",
        "Не мога да се доверя на хората. Как да изградя доверие?",
        "Как да поддържам здрави отношения със семейството?",
    ),
    "boundaries": (
        "Не мога да казвам 'не' на хората. Как да поставя граници?",
        "Хората се възползват от мен. Как да защитя личното си пространство?",
        "Как да поставя граници, без да обидя другия?",
    ),
    "communication": (
        "Не умея да изразявам чувствата си. Какво да направя?",
        "Имаме чести конфликти. Как да общуваме по-добре?",
        "Как да слушам активно?",
    ),
    "loneliness": (
        "Чувствам се сам/а, дори когато съм с хора. Какво да направя?",
        "Изолирах се от приятели. Как да възстановя връзките?",
        "Самотата ме потиска. Помогни ми.",
    ),
    "parenting": (
        "Не знам как да говоря с детето си. Какво да направя?",
        "Тийнейджърът ми не ме чува. Как да подходя?",
        "Как да възпитавам, без да наранявам?",
    ),
    "mindfulness": (
        "Постоянно съм в мислите си. Как да живея в настоящето?",
        "Как да започна да медитирам?",
        "Кои дихателни упражнения помагат при стрес?",
    ),
    "resilience": (
        "Как да изградя психическа устойчивост?",
        "Преживявам труден период. Откъде да намеря сили?",
        "Какво помага да се справиш с трудностите?",
    ),
    "habits": (
        "Как да формирам здравословни навици?",
        "Започвам и спирам всеки път. Как да задържа промяната?",
        "Кои малки промени правят голяма разлика?",
    ),
    "sleep": (
        "Не мога да спя нощем. Какво да направя?",
        "Как да подобря качеството на съня си?",
        "Будя се изтощен/а. Защо?",
    ),
    "addiction": (
        "Имам зависимост, която ме контролира. От къде да започна?",
        "Как да се справя с пристрастяване?",
        "Близък човек има зависимост. Как да му помогна?",
    ),
    "motivation": (
        "Загубих мотивация. Как да я върна?",
        "Имам цели, но не действам. Какво да направя?",
        "Как да си поставя постижими цели?",
    ),
    "happiness": (
        "Как да съм по-щастлив/а в ежедневието?",
        "Не помня кога за последно бях наистина доволен/доволна. Помогни ми.",
        "Какво помага за повече благополучие?",
    ),
    "fear": (
        "Имам страх, който ме спира. Как да го преодолея?",
        "Страхувам се да опитам нови неща. Какво да направя?",
        "Как да се справя с фобия?",
    ),
    "trauma": (
        "Преживял/а съм нещо много тежко в миналото. Как да продължа?",
        "Травмата от детството ми още ме преследва. Какво да направя?",
        "Как да преработя травматичен спомен?",
    ),
}

# Generic fallback for any topic without explicit templates.
GENERIC_TEMPLATES = (
    "Можеш ли да ми кажеш повече за тази тема?",
    "Как мога да подходя към това?",
    "Какво съветваш по този въпрос?",
)


# ##########################################################################
# Config + helpers
# ##########################################################################


@dataclass
class BuildConfig:
    """Run-time settings for :func:`build_dataset`."""
    input_path: Path
    output_path: Path
    seed: int = 42
    min_words: int = 12
    max_chars: int = 1500
    drop_dangerous: bool = True
    extraction_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    quality_score: float = 0.80


def _iter_passages(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _instruction_for(topic: str, rng: random.Random) -> str:
    pool = INSTRUCTION_TEMPLATES.get(topic) or GENERIC_TEMPLATES
    return rng.choice(pool)


def _make_alpaca_record(passage: dict, instruction: str, cfg: BuildConfig) -> Optional[dict]:
    """Build one Alpaca record from a passage. None when filtered out."""
    # Local import to avoid pulling experiments/shared into data_transformation
    # at module-import time (CLI startup stays fast).
    from experiments.shared.dataset_utils import (
        make_record, quality_filter, estimate_token_count,
    )

    topic = passage.get("topic", "")
    category = TOPIC_TO_CATEGORY.get(topic, "out_of_domain")
    output_text = (passage.get("text") or "").strip()
    if not output_text:
        return None
    if len(output_text) > cfg.max_chars:
        output_text = output_text[: cfg.max_chars].rsplit(" ", 1)[0] + "…"

    try:
        rec = make_record(
            instruction=instruction,
            output=output_text,
            source=f"chitanka:{passage.get('book_id', '?')}",
            category=category if category in (
                "anxiety", "depression", "stress", "grief", "relationships",
                "self-esteem", "out_of_domain",
            ) else "out_of_domain",
            difficulty="moderate",
            quality_score=cfg.quality_score,
            is_refusal=False,
            language=passage.get("language") or "bg",
        )
    except ValueError:
        return None
    if not quality_filter(rec, min_words=cfg.min_words, drop_dangerous=cfg.drop_dangerous):
        return None
    # Stash a metadata blob so we can trace back to the source book.
    rec["metadata"] = {
        "passage_id": passage.get("passage_id"),
        "book_title": passage.get("book_title"),
        "authors": passage.get("authors"),
        "topic": topic,
        "keywords_matched": passage.get("keywords_matched"),
        "paragraph_index": passage.get("paragraph_index"),
    }
    rec["token_count"] = estimate_token_count(rec["instruction"]) + estimate_token_count(rec["output"])
    return rec


def build_dataset(cfg: BuildConfig) -> tuple[int, int]:
    """Materialize the Alpaca JSONL. Returns ``(total_in, total_out)``."""
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"input passages JSONL not found: {cfg.input_path}")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)

    total_in = 0
    total_out = 0
    with cfg.output_path.open("w", encoding="utf-8") as out:
        for passage in _iter_passages(cfg.input_path):
            total_in += 1
            topic = passage.get("topic", "")
            instruction = _instruction_for(topic, rng)
            rec = _make_alpaca_record(passage, instruction, cfg)
            if rec is None:
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_out += 1
    return total_in, total_out


def maybe_upload(
    local_path: Path,
    *,
    source: str,
    date: str,
    bucket: Optional[str] = None,
    remote_prefix: str = "datasets",
    remote_key: Optional[str] = None,
) -> Optional[str]:
    """Optionally push the built JSONL to MinIO. Returns the s3 URI or None.

    The remote key defaults to
    ``{remote_prefix}/{source}/{date}/{local_filename}`` — the MinIO
    filename always mirrors the local one. Pass ``remote_key`` to
    override completely.
    """
    try:
        from data_platform.common import set_env
        from data_platform.storage import MinioStorage
    except ImportError:
        return None
    set_env(quiet=True)
    storage = MinioStorage.from_env()
    if remote_key is None:
        remote_key = f"{remote_prefix.rstrip('/')}/{source}/{date}/{local_path.name}"
    return storage.save_file(local_path, remote_key, bucket=bucket)


def _topic_of(record: dict) -> str:
    """Fine-grained topic for a record, falling back to category / unknown."""
    topic = (record.get("metadata") or {}).get("topic")
    if topic:
        return str(topic)
    return record.get("category") or "_unknown"


def split_by_topic(
    input_path: Path,
    out_dir: Path,
    *,
    eval_ratio: float = 0.1,
    seed: int = 42,
    min_per_topic: int = 10,
    stratify_by: str = "category",
) -> dict[str, tuple[int, int]]:
    """Split a built Alpaca dataset into per-topic + combined train/eval sets.

    Reads ``input_path`` (the JSONL produced by :func:`build_dataset`),
    groups records by their fine-grained ``metadata.topic`` and writes::

        {out_dir}/all/train.jsonl          {out_dir}/all/eval.jsonl
        {out_dir}/by-topic/{topic}/train.jsonl   .../eval.jsonl

    Topics with fewer than ``min_per_topic`` records are folded into a
    ``_misc`` bucket so we don't emit single-example splits. Each split is
    stratified by ``stratify_by`` (default ``category``) for balance.

    Returns ``{topic: (n_train, n_eval)}`` including the ``"all"`` key.
    """
    from experiments.shared.dataset_utils import (
        load_alpaca_dataset,
        stratified_split,
        write_alpaca_dataset,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"input dataset JSONL not found: {input_path}")

    by_topic: dict[str, list[dict]] = {}
    all_records: list[dict] = []
    for rec in load_alpaca_dataset(input_path):
        all_records.append(rec)
        by_topic.setdefault(_topic_of(rec), []).append(rec)

    # Fold tiny topics into a shared _misc bucket.
    misc: list[dict] = []
    for topic in list(by_topic):
        if len(by_topic[topic]) < min_per_topic:
            misc.extend(by_topic.pop(topic))
    if misc:
        by_topic.setdefault("_misc", []).extend(misc)

    results: dict[str, tuple[int, int]] = {}

    def _emit(name: str, records: list[dict], dest: Path) -> None:
        train, eval_ = stratified_split(
            records, eval_ratio=eval_ratio, stratify_by=stratify_by, seed=seed,
        )
        dest.mkdir(parents=True, exist_ok=True)
        n_train = write_alpaca_dataset(train, dest / "train.jsonl")
        n_eval = write_alpaca_dataset(eval_, dest / "eval.jsonl")
        results[name] = (n_train, n_eval)

    _emit("all", all_records, out_dir / "all")
    for topic, records in sorted(by_topic.items()):
        _emit(topic, records, out_dir / "by-topic" / topic)

    return results


__all__ = [
    "BuildConfig",
    "INSTRUCTION_TEMPLATES",
    "TOPIC_TO_CATEGORY",
    "build_dataset",
    "maybe_upload",
    "split_by_topic",
]
