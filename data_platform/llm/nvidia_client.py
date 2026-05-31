"""Thin client for NVIDIA-hosted, OpenAI-compatible chat models.

Every model we use (Mistral Large 3 as a generator/quality judge, Llama
Guard 4 and Nemotron Content Safety as safety classifiers) is served from
the same endpoint — ``https://integrate.api.nvidia.com/v1/chat/completions``
— with a per-model bearer token read from the environment. The raw,
copy-pasteable calls this wraps live in ``api_judges_scripts/``.

The client is deliberately small: a ``chat`` method that returns the
assistant text, a ``chat_json`` helper that parses a JSON object out of
that text, retries on rate-limit/5xx, and a registry of the models we use.

"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


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
        self.api_key = os.getenv(model.api_key_env)
        if not self.api_key:
            raise NvidiaChatError(
                f"Missing API key: set {model.api_key_env} in your environment / .env"
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
            try:
                resp = requests.post(
                    NVIDIA_INVOKE_URL, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 429:
                    # Honor Retry-After when present (seconds or HTTP-date); fall
                    # back to a longer base delay for rate limits than for 5xx.
                    retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                    wait = retry_after if retry_after is not None else max(
                        self.base_delay * (2**attempt), 15.0,
                    )
                    last_exc = NvidiaChatError(
                        f"HTTP 429 Too Many Requests (retry after {wait:.1f}s): "
                        f"{resp.text[:160]}"
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
