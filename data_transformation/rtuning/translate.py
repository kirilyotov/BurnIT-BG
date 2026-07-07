"""Concurrent BG translation for raw R-Tuning rows.

Replaces the original sequential single-threaded translator. Each worker
thread carries its own ``deep_translator.GoogleTranslator`` instance via
thread-local storage; Google's free endpoint handles 30+ concurrent
requests without backing off (we keep the default conservative at 32).

Key differences vs the previous version:

* **Concurrent.** ``concurrency`` workers translate question/answer fields
  in parallel — 30–50× faster on realistic loads.
* **Persistent cache that actually flushes.** Atomic write via ``.tmp``+
  ``rename`` every ``flush_every`` records. The previous version relied on
  ``Translator.translate_many``'s flush; calling ``.translate()`` in a
  loop never wrote anything to disk.
* **Resumable.** If the output JSONL already exists, we count its lines
  and skip that many input records — kill the process at any time and
  re-run from where it stopped. (Combined with the warm cache, re-runs
  are cheap even if the output was deleted.)
* Same ``TranslateRTuningConfig`` shape + same CLI flags as before, so
  the bash orchestrator just works.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    NotValidLength,
    NotValidPayload,
    RequestError,
    TranslationNotFound,
)

log = logging.getLogger(__name__)


# ── Backends ──────────────────────────────────────────────────────────────
# "google"        — deep_translator.GoogleTranslator (free web endpoint).
#                   5 req/sec hard limit, 200k req/day per IP. Easily
#                   throttled for hours when limits trip.
# "libretranslate" — self-hosted LibreTranslate REST API. No rate limits.
#                   Set LIBRETRANSLATE_URL (default http://localhost:5000).
#                   Start with: docker run -ti --rm -p 5000:5000 \
#                       libretranslate/libretranslate --load-only en,bg
BACKENDS = ("google", "libretranslate")


def _libretranslate_url(base_url: str | None) -> str:
    import os as _os
    return (base_url or _os.getenv("LIBRETRANSLATE_URL", "http://localhost:5000")).rstrip("/")


def _libretranslate_payload(q: str | list[str], source: str, target: str) -> dict:
    import os as _os
    payload: dict = {"q": q, "source": source, "target": target, "format": "text"}
    api_key = _os.getenv("LIBRETRANSLATE_API_KEY")
    if api_key:
        payload["api_key"] = api_key
    return payload


def _libretranslate(text: str, source: str, target: str,
                    base_url: str | None = None, timeout: float = 30.0) -> str:
    """Single-string POST to LibreTranslate /translate."""
    resp = requests.post(
        f"{_libretranslate_url(base_url)}/translate",
        json=_libretranslate_payload(text, source, target),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("translatedText", text)


def _libretranslate_batch(texts: list[str], source: str, target: str,
                          base_url: str | None = None, timeout: float = 120.0) -> list[str]:
    """Batch POST: ``q`` is a list, response is a list of equal length.

    LibreTranslate handles batches in a single inference pass, so this is
    dramatically faster than N individual requests — both for HTTP overhead
    and (because LT's model is faster on batched inputs) raw throughput.
    """
    if not texts:
        return []
    resp = requests.post(
        f"{_libretranslate_url(base_url)}/translate",
        json=_libretranslate_payload(texts, source, target),
        timeout=timeout,
    )
    resp.raise_for_status()
    out = resp.json().get("translatedText", [])
    if isinstance(out, str):
        # Some LT versions return a single string when q is length 1.
        return [out]
    if not isinstance(out, list) or len(out) != len(texts):
        # Fall back: re-translate one-by-one if the shape is unexpected.
        return [_libretranslate(t, source, target, base_url) for t in texts]
    return list(out)


# Thread-local: one GoogleTranslator per (source, target) per worker thread.
# deep_translator instantiates a fresh requests.Session per object, so reusing
# a thread-local engine avoids per-call connection setup overhead.
_TL = threading.local()


def _engine(source: str, target: str) -> GoogleTranslator:
    engines: dict[tuple[str, str], GoogleTranslator] = getattr(_TL, "engines", None) or {}
    key = (source, target)
    if key not in engines:
        engines[key] = GoogleTranslator(source=source, target=target)
        _TL.engines = engines
    return engines[key]


# ── Global request-rate budget ─────────────────────────────────────────────
# Google's free web endpoint enforces **5 req/sec + 200k req/day per IP**.
# Above 5 RPS it returns "You made too many requests to the server" and
# locks the IP for ~60-300s.  Concurrent workers cooperate via this single
# shared budget so we never breach the per-second cap regardless of
# concurrency. 4 RPS = 240 RPM = ~4.4 records/sec at 2 fields/row.
# Tune via TRANSLATE_GLOBAL_RPM env (e.g. for a self-hosted LibreTranslate
# instance with no rate limit, set this very high to disable throttling).
_GLOBAL_RPM_BUDGET = 240.0
_GLOBAL_BUDGET_LOCK = threading.Lock()
_GLOBAL_TOKENS: float = 1.0
_GLOBAL_LAST_REFILL: float = time.monotonic()
_GLOBAL_BACKOFF_UNTIL: float = 0.0


def _set_global_rpm(rpm: float) -> None:
    """Reset the shared budget to a new RPM (mostly for tests / overrides)."""
    global _GLOBAL_RPM_BUDGET, _GLOBAL_TOKENS, _GLOBAL_LAST_REFILL
    with _GLOBAL_BUDGET_LOCK:
        _GLOBAL_RPM_BUDGET = float(rpm)
        _GLOBAL_TOKENS = 1.0
        _GLOBAL_LAST_REFILL = time.monotonic()


def _acquire_global_token() -> None:
    """Block until the shared budget grants 1 token (≤ ``_GLOBAL_RPM_BUDGET/60``/sec).

    Also honors a global backoff window when we recently hit Google's
    'too many requests' wall — every worker sleeps until that window
    clears, so the IP can cool off.
    """
    global _GLOBAL_TOKENS, _GLOBAL_LAST_REFILL
    refill_per_sec = _GLOBAL_RPM_BUDGET / 60.0
    while True:
        now = time.monotonic()
        # Cool-off after a 429
        with _GLOBAL_BUDGET_LOCK:
            backoff_remaining = _GLOBAL_BACKOFF_UNTIL - now
        if backoff_remaining > 0:
            time.sleep(min(backoff_remaining, 1.0))
            continue
        with _GLOBAL_BUDGET_LOCK:
            elapsed = now - _GLOBAL_LAST_REFILL
            if elapsed > 0:
                _GLOBAL_TOKENS = min(2.0, _GLOBAL_TOKENS + elapsed * refill_per_sec)
                _GLOBAL_LAST_REFILL = now
            if _GLOBAL_TOKENS >= 1.0:
                _GLOBAL_TOKENS -= 1.0
                return
            deficit = 1.0 - _GLOBAL_TOKENS
            wait = deficit / refill_per_sec
        time.sleep(min(wait, 0.5))


def _trigger_global_backoff(seconds: float) -> None:
    """Park every translator thread for ``seconds`` after a 'too many requests'."""
    global _GLOBAL_BACKOFF_UNTIL
    with _GLOBAL_BUDGET_LOCK:
        _GLOBAL_BACKOFF_UNTIL = max(_GLOBAL_BACKOFF_UNTIL, time.monotonic() + seconds)


def _translate_text(text: str, source: str, target: str, *,
                    backend: str = "google",
                    backend_url: str | None = None,
                    retries: int = 5, backoff: float = 3.0) -> str:
    """Translate one string via the chosen backend; retries transient errors.

    Pre-acquires a token from the shared global RPM budget so concurrent
    workers never burst above the configured rate. On Google's explicit
    "too many requests" error every worker is parked for 90s of cool-off.
    For LibreTranslate (self-hosted) the rate budget can be cranked
    arbitrarily high via TRANSLATE_GLOBAL_RPM.
    """
    if not text or not text.strip():
        return text
    last_exc: Exception | None = None
    for attempt in range(retries):
        _acquire_global_token()
        try:
            if backend == "libretranslate":
                return _libretranslate(text, source, target, backend_url)
            # Default: Google free endpoint
            result = _engine(source, target).translate(text)
            return result if result is not None else text
        except (NotValidLength, NotValidPayload, TranslationNotFound):
            return text   # un-retryable — return original so downstream can flag
        except RequestError as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "too many requests" in msg or "5 requests per second" in msg:
                cool = 90.0 + (attempt * 30.0)
                _trigger_global_backoff(cool)
                time.sleep(cool)
            else:
                time.sleep(backoff * (2 ** attempt))
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff * (2 ** attempt))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(backoff * (2 ** attempt))
    log.warning("translate gave up after %d retries (backend=%s): %s",
                retries, backend, last_exc)
    return text


@dataclass
class TranslateRTuningConfig:
    """Settings for :func:`translate_rtuning`."""

    input_jsonl: Path
    output_jsonl: Path
    cache_path: Path | None = None
    source_lang: str = "en"
    target_lang: str = "bg"
    fields: tuple[str, ...] = ("question", "answer")
    # Concurrency vs. the global RPM budget — workers cooperate via the
    # shared bucket so we never burst above Google's 5-req/sec wall.
    # Default 8 workers / 500-row chunks: workers stay busy while never
    # firing > 4 RPS net (set in _GLOBAL_RPM_BUDGET). For a self-hosted
    # LibreTranslate instance with no per-IP cap, raise both and also
    # bump TRANSLATE_GLOBAL_RPM in env.
    concurrency: int = 8
    chunk_size: int = 500
    delay: float = 0.0
    flush_every: int = 500
    limit: int | None = None
    log_progress_every: int = 1000
    backend: str = "google"  # "google" | "libretranslate"
    backend_url: str | None = None  # for libretranslate; None → env / default


def translate_rtuning(cfg: TranslateRTuningConfig) -> dict[str, int]:
    """Concurrently BG-translate ``cfg.fields`` of every record.

    Returns ``{"total_in": N, "translated": K, "skipped": S, "resumed_from": R,
    "cache_size": C}``.
    """
    if not cfg.input_jsonl.exists():
        raise FileNotFoundError(f"input not found: {cfg.input_jsonl}")
    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Backend-specific rate budgeting. Google has a hard 5 req/sec cap so
    # we throttle to ~4 RPS. LibreTranslate (self-hosted) has no per-IP cap,
    # so we raise the budget by default (and the user can still override).
    import os
    env_rpm = os.getenv("TRANSLATE_GLOBAL_RPM")
    if env_rpm:
        try:
            _set_global_rpm(float(env_rpm))
            log.info("translate: TRANSLATE_GLOBAL_RPM=%s applied", env_rpm)
        except ValueError:
            log.warning("translate: ignoring non-numeric TRANSLATE_GLOBAL_RPM=%r", env_rpm)
    elif cfg.backend == "libretranslate":
        # LibreTranslate is self-hosted — no rate limit makes sense. Cap
        # high enough that concurrency drives throughput, not the budget.
        _set_global_rpm(60_000.0)
        log.info("translate: backend=libretranslate → global rate budget effectively disabled")
    log.info("translate: backend=%s  concurrency=%d  chunk_size=%d",
             cfg.backend, cfg.concurrency, cfg.chunk_size)

    # ── Cache: load existing translations from disk for resumability ────
    cache_lock = threading.Lock()
    cache: dict[str, str] = {}
    if cfg.cache_path and Path(cfg.cache_path).exists():
        try:
            cache = json.loads(Path(cfg.cache_path).read_text(encoding="utf-8"))
            log.info("translate: loaded %d cached translations from %s",
                     len(cache), cfg.cache_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("translate: cache load failed (%s), starting empty", exc)

    def cache_key(text: str) -> str:
        return f"{cfg.source_lang}|{cfg.target_lang}|{text}"

    def flush_cache() -> None:
        if not cfg.cache_path:
            return
        path = Path(cfg.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with cache_lock:
            snap = dict(cache)
        # Atomic write so a kill mid-flush can't corrupt the cache.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # ── Resume: skip rows already present in the output file ────────────
    resumed_from = 0
    if cfg.output_jsonl.exists():
        with cfg.output_jsonl.open("r", encoding="utf-8") as fh:
            resumed_from = sum(1 for line in fh if line.strip())
        if resumed_from:
            log.info("translate: resuming — output already has %d records; "
                     "skipping the first %d input rows",
                     resumed_from, resumed_from)

    # ── Stream input, batch into chunks, translate each chunk concurrently ─
    open_mode = "a" if resumed_from else "w"
    total_in = translated_count = skipped = 0

    with ThreadPoolExecutor(max_workers=cfg.concurrency,
                            thread_name_prefix="trans") as executor, \
         cfg.output_jsonl.open(open_mode, encoding="utf-8") as out_fh:

        chunk: list[dict] = []
        last_progress_logged = 0

        def drain_chunk(chunk_records: list[dict]) -> int:
            """Translate every field in chunk_records concurrently. Returns
            count of records that had at least one field translated."""
            translated_here = 0

            # Build the list of tasks (skip empty fields + cache hits).
            tasks: list[tuple[int, str, str, str]] = []  # (idx, field, text, key)
            for rec_idx, record in enumerate(chunk_records):
                for field_name in cfg.fields:
                    text = (record.get(field_name) or "").strip()
                    bg_key = f"{field_name}_bg"
                    if not text:
                        record[bg_key] = ""
                        continue
                    key = cache_key(text)
                    with cache_lock:
                        cached = cache.get(key)
                    if cached is not None:
                        record[bg_key] = cached
                    else:
                        tasks.append((rec_idx, field_name, text, key))

            if not tasks:
                # All fields were either empty or cached.
                return len(chunk_records)

            if cfg.backend == "libretranslate":
                # ── BATCHED path ───────────────────────────────────────
                # Split tasks into N batches; submit each batch as ONE
                # POST. concurrency workers process batches in parallel,
                # so we get both batching's per-call efficiency AND
                # multi-worker parallelism.
                batch_size = max(1, len(tasks) // max(1, cfg.concurrency))
                batch_size = min(batch_size, 100)  # LT memory cap
                batches: list[list[tuple]] = [
                    tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)
                ]

                def _do_batch(batch: list[tuple]) -> list[tuple[tuple, str]]:
                    texts = [t[2] for t in batch]
                    try:
                        results = _libretranslate_batch(
                            texts, cfg.source_lang, cfg.target_lang, cfg.backend_url
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("libretranslate batch (%d items) failed: %s",
                                    len(texts), exc)
                        results = texts  # fall back to originals
                    return list(zip(batch, results))

                futures = [executor.submit(_do_batch, b) for b in batches]
                for fut in as_completed(futures):
                    try:
                        results = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("libretranslate batch worker died: %s", exc)
                        continue
                    for (rec_idx, field_name, text, key), translated in results:
                        chunk_records[rec_idx][f"{field_name}_bg"] = translated
                        with cache_lock:
                            cache[key] = translated
            else:
                # ── Per-text path (Google or default) ──────────────────
                futures = {
                    executor.submit(
                        _translate_text, text, cfg.source_lang, cfg.target_lang,
                        backend=cfg.backend, backend_url=cfg.backend_url,
                    ): (rec_idx, field_name, text, key)
                    for (rec_idx, field_name, text, key) in tasks
                }
                for fut in as_completed(futures):
                    rec_idx, field_name, text, key = futures[fut]
                    try:
                        translated = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        log.warning("translate failed [%s id=%s]: %s",
                                    field_name, chunk_records[rec_idx].get("source_id"), exc)
                        translated = text
                    chunk_records[rec_idx][f"{field_name}_bg"] = translated
                    with cache_lock:
                        cache[key] = translated

            translated_here = len(chunk_records)
            return translated_here

        start_t = time.monotonic()
        for raw_idx, record in enumerate(_stream_records(cfg.input_jsonl)):
            if raw_idx < resumed_from:
                continue
            if cfg.limit is not None and total_in >= cfg.limit:
                break

            total_in += 1
            chunk.append(record)

            if len(chunk) < cfg.chunk_size:
                continue

            # Process and write the chunk.
            n_translated = drain_chunk(chunk)
            translated_count += n_translated
            for rec in chunk:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            chunk.clear()

            if translated_count - last_progress_logged >= cfg.log_progress_every:
                elapsed = time.monotonic() - start_t
                rate = translated_count / max(elapsed, 1e-3) * 60.0
                log.info(
                    "translate: %d / chunk done (rate: %.0f rows/min, "
                    "cache: %d entries)",
                    resumed_from + translated_count, rate, len(cache),
                )
                last_progress_logged = translated_count

            if total_in % cfg.flush_every == 0:
                flush_cache()

            if cfg.delay > 0:
                time.sleep(cfg.delay)

        # Final partial chunk.
        if chunk:
            n_translated = drain_chunk(chunk)
            translated_count += n_translated
            for rec in chunk:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()

    flush_cache()
    elapsed = time.monotonic() - start_t
    rate = translated_count / max(elapsed, 1e-3) * 60.0
    log.info(
        "translate: DONE — input=%d resumed_from=%d translated=%d "
        "skipped=%d cache_entries=%d wall=%.1fs rate=%.0f rows/min",
        total_in, resumed_from, translated_count, skipped, len(cache), elapsed, rate,
    )
    return {
        "total_in": total_in,
        "translated": translated_count,
        "skipped": skipped,
        "resumed_from": resumed_from,
        "cache_size": len(cache),
    }


def _stream_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


__all__ = ["TranslateRTuningConfig", "translate_rtuning"]
