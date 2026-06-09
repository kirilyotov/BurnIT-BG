"""Thin client for NVIDIA-hosted, OpenAI-compatible chat models.

All models (Mistral Large 3, Llama Guard 4, Nemotron Content Safety) share
one endpoint with a per-model bearer token from env. Exposes ``chat`` (text),
``chat_json`` (parsed JSON), retries on 429/5xx, and a model registry.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


# ────────────────────────────────────────────────────────────────────────────
#  Rate limiting — pre-emptive token bucket
# ────────────────────────────────────────────────────────────────────────────
# NVIDIA's free tier is ~40 RPM per key. 429s have no Retry-After, so we pace
# requests up-front. One shared bucket per (model, rpm) covers all clients
# and parallel scorer threads on that key.

_BUCKETS_LOCK = threading.Lock()
_BUCKETS: dict[str, "RpmBucket"] = {}


class RpmBucket:
    """Token bucket: refills at ``rpm`` tokens per minute, with a small burst."""

    def __init__(self, rpm: float, burst: float | None = None) -> None:
        self.rpm = float(rpm)
        self.refill_per_sec = self.rpm / 60.0
        # Allow a small burst so multi-judge panels don't trickle one-by-one.
        self.capacity = float(burst if burst is not None else max(3.0, rpm / 8.0))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
            self._last = now

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until at least ``tokens`` are available."""
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.refill_per_sec
            # Sleep OUTSIDE the lock so other threads can refill too.
            time.sleep(min(wait, 0.5))


def get_bucket(key: str, rpm: float) -> RpmBucket:
    """Return the process-wide shared bucket for ``key`` (e.g. a model handle).

    Once a bucket is created the rpm is fixed — subsequent calls with the same
    key ignore the rpm argument. This is intentional: we never want two
    competing buckets racing against the same upstream rate limit.
    """
    with _BUCKETS_LOCK:
        if key not in _BUCKETS:
            _BUCKETS[key] = RpmBucket(rpm)
        return _BUCKETS[key]


@dataclass(frozen=True)
class NvidiaModel:
    """An NVIDIA-hosted model id plus the env var holding its API key."""

    model_id: str
    api_key_env: str
    label: str


# Canonical registry. Keys are short, stable handles used across the repo.
# ByteDance Seed is intentionally excluded per project policy.
MODELS: dict[str, NvidiaModel] = {
    "mistral-large-3": NvidiaModel(
        model_id="mistralai/mistral-large-3-675b-instruct-2512",
        api_key_env="MISTRAL_LARGE_3_675B_API_KEY",
        label="Mistral Large 3 (675B)",
    ),
    "llama-guard-4": NvidiaModel(
        model_id="meta/llama-guard-4-12b",
        api_key_env="LLAMA_GUARD_4_12B_API_KEY",
        label="Llama Guard 4 (12B)",
    ),
    "nemotron-content-safety": NvidiaModel(
        model_id="nvidia/nemotron-3-content-safety",
        api_key_env="NEMETRON_3_CONTENT_SAFETY_API_KEY",
        label="Nemotron-3 Content Safety",
    ),
}


class NvidiaChatError(RuntimeError):
    """Raised when a chat completion cannot be obtained."""


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_OBJECT = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from model text.

    Handles bare JSON, ```json fenced blocks, and prose with an embedded
    object. Raises ``NvidiaChatError`` when nothing parses.
    """
    text = (text or "").strip()
    candidates: list[str] = []
    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    embedded = _JSON_OBJECT.search(text)
    if embedded:
        candidates.append(embedded.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise NvidiaChatError(f"No parseable JSON in model output: {text[:200]!r}")


class NvidiaChatClient:
    """Minimal chat client for one NVIDIA-hosted model.

    ``model`` may be a registry handle (e.g. ``"mistral-large-3"``) or an
    :class:`NvidiaModel`. The API key is read from the model's
    ``api_key_env`` at construction time, so load your ``.env`` first.
    """

    def __init__(
        self,
        model: str | NvidiaModel,
        *,
        timeout: float = 90.0,
        max_retries: int = 4,
        base_delay: float = 2.0,
        rpm: float | None = None,
    ) -> None:
        if isinstance(model, str):
            if model not in MODELS:
                raise NvidiaChatError(
                    f"Unknown model handle {model!r}. Known: {sorted(MODELS)}"
                )
            model = MODELS[model]
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        raw_key = os.getenv(model.api_key_env)
        if not raw_key:
            raise NvidiaChatError(
                f"Missing API key: set {model.api_key_env} in your environment / .env"
            )
        # Strip whitespace — pasted-from-browser keys frequently carry a
        # trailing newline, which makes the Bearer header `"Bearer <key>\n"`
        # and NVIDIA rejects with 401. python-dotenv strips this for .env
        # files; we mirror that for env vars sourced from Colab Secrets.
        self.api_key = raw_key.strip()
        # Pre-emptive rate limit. NVIDIA free tier is ~40 RPM per key — pick a
        # default well below to leave room for retries. Buckets are shared
        # process-wide per model_id so parallel scorers from
        # mlflow.genai.evaluate cooperate instead of stampeding.
        if rpm is None:
            env_rpm = os.getenv("NVIDIA_DEFAULT_RPM")
            rpm = float(env_rpm) if env_rpm else 30.0
        self.rate_bucket: RpmBucket | None = (
            get_bucket(model.model_id, rpm) if rpm and rpm > 0 else None
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        top_p: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant's text for a list of chat ``messages``.

        Retries on HTTP 429 and 5xx with exponential backoff.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            # Pre-emptive throttle — wait for a token BEFORE we hit the wire.
            # Saves a failed 429 round-trip that would otherwise count against
            # the same per-minute budget.
            if self.rate_bucket is not None:
                self.rate_bucket.acquire()
            try:
                resp = requests.post(
                    NVIDIA_INVOKE_URL, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 429:
                    # NVIDIA's NIM endpoints DO NOT reliably set Retry-After,
                    # so we use exponential backoff with FULL JITTER (the AWS
                    # "Architecture Blog" pattern). Without jitter, multiple
                    # parallel scorer threads all retry at the same moment and
                    # immediately re-trigger 429s. Honor Retry-After when
                    # present, otherwise jitter our own backoff.
                    retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                    if retry_after is not None:
                        wait = retry_after + random.uniform(0.0, 0.5)
                    else:
                        cap = max(self.base_delay * (2**attempt), 15.0)
                        wait = random.uniform(0.0, cap)
                    last_exc = NvidiaChatError(
                        f"HTTP 429 Too Many Requests (retry in {wait:.1f}s, "
                        f"attempt {attempt + 1}/{self.max_retries}): {resp.text[:160]}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    last_exc = NvidiaChatError(
                        f"transient HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(self.base_delay * (2**attempt))
                    continue
                # 401: don't retry, but emit a one-line diagnostic with key
                # length + masked head/tail so the user can tell at a glance
                # whether the key was loaded at all, truncated, or mismatched.
                # The actual key value is NEVER printed in full.
                if resp.status_code == 401:
                    masked = f"{self.api_key[:4]}…{self.api_key[-4:]}" if len(self.api_key) >= 12 else "(too short)"
                    raise NvidiaChatError(
                        f"401 Unauthorized for {self.model.label}. "
                        f"Key length={len(self.api_key)}, value={masked} "
                        f"(env: {self.model.api_key_env}). "
                        f"Server response: {resp.text[:160]}. "
                        f"Likely causes: key expired/revoked on build.nvidia.com, "
                        f"key has invisible whitespace (now auto-stripped on load), "
                        f"or Colab Secret value differs from the .env value."
                    )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.base_delay * (2**attempt))
        raise NvidiaChatError(
            f"{self.model.label} failed after {self.max_retries} attempts: {last_exc}"
        )

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Parse a Retry-After header (seconds or HTTP-date). Returns float seconds or None."""
        if not value:
            return None
        value = value.strip()
        try:
            return float(value)
        except ValueError:
            pass
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            dt = parsedate_to_datetime(value)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            seconds = (dt - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, seconds)
        except Exception:
            return None

    def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Like :meth:`chat` but parse a JSON object/array from the reply."""
        return extract_json(self.chat(messages, **kwargs))
