"""Rewrite each record's instruction using Mistral so it actually matches its passage.

The chitanka style dataset pairs a generic topic-template question with a
random passage — instruction and output are weakly related. This module
streams the dataset, sends each passage to Mistral with a prompt asking for
a realistic first-person Bulgarian question the passage would answer, and
writes the updated records to a new JSONL. Only ``instruction`` changes;
``input`` stays "" (Alpaca convention) and ``output`` (the passage) is kept.

Resumable: caches by ``passage_id`` (JSON file). Concurrent: uses a
ThreadPoolExecutor for N parallel Mistral calls per batch. Streams the
input file so 34K records don't load into memory; writes output as it goes.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from data_platform.llm import NvidiaChatClient, NvidiaChatError

log = logging.getLogger(__name__)


REWRITE_SYSTEM_PROMPT = (
    "Прочети откъса от книга на български. Създай реалистичен въпрос или "
    "съобщение от ПЪРВО ЛИЦЕ (на български) — нещо, което човек би написал "
    "в чат за емоционална подкрепа — на което този откъс е смислен и "
    "подкрепящ отговор.\n\n"
    "Изисквания:\n"
    "- ПЪРВО ЛИЦЕ, емоционален, естествено звучащ.\n"
    "- Не повтаряй буквално фрази от откъса; формулирай въпроса от "
    "усещанията и предизвикателствата, които откъсът адресира.\n"
    "- Темата трябва да съвпада със СМИСЪЛА на откъса (не само с ключова дума).\n"
    "- Максимум 200 знака.\n\n"
    "Върни САМО валиден JSON с един ключ:\n"
    '{"question": "..."}'
)


@dataclass
class RewriteConfig:
    """Run-time settings for :func:`rewrite_instructions`."""

    input_path: Path
    output_path: Path
    model: str = "mistral-large-3"
    concurrency: int = 8
    batch_size: int = 50
    max_passage_chars: int = 1000
    temperature: float = 0.3
    max_tokens: int = 200
    cache_path: Optional[Path] = None
    extraction_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))


def _user_msg(topic: str, passage: str) -> str:
    return (
        f"Тема: {topic or 'обща'}\n\n"
        f"Откъс:\n{passage}\n\n"
        "Какъв е въпросът или съобщението от първо лице на български, "
        "на което този откъс е разумен и подкрепящ отговор?"
    )


def _stream_records(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file one at a time (no in-memory load)."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_cache(path: Optional[Path]) -> dict[str, str]:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("could not read cache at %s; starting fresh", path)
    return {}


def _save_cache(path: Optional[Path], cache: dict[str, str]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _rewrite_one(client: NvidiaChatClient, record: dict, cfg: RewriteConfig) -> Optional[str]:
    """Call Mistral for one record; return the new question or None on failure."""
    passage = (record.get("output") or "").strip()
    if not passage:
        return None
    if len(passage) > cfg.max_passage_chars:
        passage = passage[: cfg.max_passage_chars].rsplit(" ", 1)[0]
    topic = (record.get("metadata") or {}).get("topic") or record.get("category", "")
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": _user_msg(topic, passage)},
    ]
    try:
        obj = client.chat_json(
            messages, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
        )
    except NvidiaChatError as exc:
        log.warning("rewrite failed: %s", exc)
        return None
    if not isinstance(obj, dict):
        return None
    q = (obj.get("question") or "").strip()
    if not q or len(q) < 5:
        return None
    return q[:600]


def _apply_question(record: dict, new_q: str, source: str) -> dict:
    """Return a new record with rewritten instruction; preserves input='' and output."""
    new_rec = dict(record)
    new_rec["input"] = record.get("input", "")
    new_rec["instruction"] = new_q
    meta = dict(record.get("metadata") or {})
    meta.setdefault("original_instruction", record.get("instruction", ""))
    meta["rewritten_by"] = source
    new_rec["metadata"] = meta
    return new_rec


def rewrite_instructions(cfg: RewriteConfig) -> dict[str, int]:
    """Stream input, rewrite in concurrent batches, write output progressively.

    Returns ``{"total_in", "rewritten_live", "cache_hits", "fallbacks"}``.
    """
    if not cfg.input_path.exists():
        raise FileNotFoundError(cfg.input_path)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Long backoff + many retries so NVIDIA's 429 rate limit doesn't burn rows.
    client = NvidiaChatClient(cfg.model, max_retries=6, base_delay=15.0, timeout=120.0)
    cache = _load_cache(cfg.cache_path)

    total_in = 0
    live_writes = 0
    cache_hits = 0
    fallbacks = 0
    cache_dirty = 0
    cache_save_every = 20  # save the cache more aggressively so an interrupt loses few rows
    pending: list[tuple[str, dict]] = []
    started = time.time()

    with cfg.output_path.open("w", encoding="utf-8") as fout:
        for record in _stream_records(cfg.input_path):
            total_in += 1
            pid = (record.get("metadata") or {}).get("passage_id") or f"_idx{total_in}"

            if pid in cache:
                cache_hits += 1
                fout.write(json.dumps(_apply_question(record, cache[pid], "cached"), ensure_ascii=False) + "\n")
                continue

            pending.append((pid, record))
            if len(pending) >= cfg.batch_size:
                rw, fb = _process_batch(client, pending, cfg, cache, fout)
                live_writes += rw
                fallbacks += fb
                cache_dirty += rw
                pending.clear()
                if cache_dirty >= cache_save_every:
                    _save_cache(cfg.cache_path, cache)
                    cache_dirty = 0
                elapsed = time.time() - started
                rate = total_in / max(elapsed, 1e-6)
                log.info(
                    "  processed %d  (cached %d, rewrote %d, fallback %d) | %.2f rec/s",
                    total_in, cache_hits, live_writes, fallbacks, rate,
                )

        if pending:
            rw, fb = _process_batch(client, pending, cfg, cache, fout)
            live_writes += rw
            fallbacks += fb
            pending.clear()

    _save_cache(cfg.cache_path, cache)
    return {
        "total_in": total_in,
        "rewritten_live": live_writes,
        "cache_hits": cache_hits,
        "fallbacks": fallbacks,
    }


def _process_batch(
    client: NvidiaChatClient,
    batch: list[tuple[str, dict]],
    cfg: RewriteConfig,
    cache: dict[str, str],
    fout,
) -> tuple[int, int]:
    """Concurrent rewrite of one batch; write results in input order."""
    results: list[Optional[str]] = [None] * len(batch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futs = {ex.submit(_rewrite_one, client, rec, cfg): i for i, (_, rec) in enumerate(batch)}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("batch task failed: %s", exc)
                results[i] = None

    rewritten = 0
    fallbacks = 0
    for (pid, rec), new_q in zip(batch, results):
        if new_q:
            cache[pid] = new_q
            fout.write(json.dumps(_apply_question(rec, new_q, client.model.model_id), ensure_ascii=False) + "\n")
            rewritten += 1
        else:
            fallbacks += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fout.flush()
    return rewritten, fallbacks


__all__ = ["RewriteConfig", "rewrite_instructions", "REWRITE_SYSTEM_PROMPT"]
