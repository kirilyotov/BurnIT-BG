"""HTTP helpers shared across catalog sources.

Implements retry-with-backoff for 429/5xx, honours ``Retry-After`` when the
server provides it, and exposes a streaming downloader that returns
``(sha256, size_bytes)``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

USER_AGENT = "BookIngestBot/1.0 (+https://github.com/kirilyotov/BurnIT-BG)"
RETRY_STATUSES = {429, 500, 502, 503, 504}


def _backoff(attempt: int, base: float, response: Optional[requests.Response] = None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
    return min(60.0, base * (2 ** attempt) + 1.0)


def request_with_retries(
    url: str,
    *,
    method: str = "GET",
    retries: int = 5,
    delay: float = 1.0,
    stream: bool = False,
    timeout: float = 30.0,
    headers: Optional[dict] = None,
    allow_redirects: bool = True,
) -> requests.Response:
    """Wrap ``requests.request`` with exponential backoff for transient errors."""
    final_headers = {"User-Agent": USER_AGENT}
    if headers:
        final_headers.update(headers)
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.request(
                method,
                url,
                headers=final_headers,
                stream=stream,
                timeout=timeout,
                allow_redirects=allow_redirects,
            )
            if resp.status_code in RETRY_STATUSES and attempt < retries - 1:
                wait = _backoff(attempt, delay, resp)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(_backoff(attempt, delay))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unreachable: request_with_retries exhausted for {url}")


def stream_to_file(
    url: str,
    dest: Path,
    *,
    retries: int = 5,
    delay: float = 1.0,
    chunk: int = 65536,
) -> Tuple[str, int]:
    """Download *url* to *dest* with retries, returning ``(sha256, size)``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = request_with_retries(url, stream=True, retries=retries, delay=delay, timeout=60)
    hasher = hashlib.sha256()
    size = 0
    with open(dest, "wb") as fh:
        for piece in resp.iter_content(chunk_size=chunk):
            if piece:
                fh.write(piece)
                hasher.update(piece)
                size += len(piece)
    return hasher.hexdigest(), size


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
