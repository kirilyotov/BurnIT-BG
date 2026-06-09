"""Environment loading helpers for .env files and Colab secrets.

This module exposes two helpers:

* ``load_env`` – low-level helper that loads one or more ``.env`` files into
  ``os.environ`` (thin wrapper over ``python-dotenv``).
* ``set_env`` – high-level helper that auto-detects the runtime
  (Google Colab or local) and pulls secrets from the appropriate source so
  the rest of the code can keep using ``os.getenv(...)``.

The motivation: notebooks should run unchanged on a laptop (``.env`` file)
and in Google Colab (``google.colab.userdata`` secrets). One call to
``set_env()`` at the top of a notebook is all that's needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal

Runtime = Literal["colab", "local"]


# Default keys we attempt to populate when none are explicitly given.
# These mirror the variables used by MinioStorage / HuggingFaceStorage /
# MLflowTracking so a single ``set_env()`` call configures everything.
DEFAULT_KEYS: tuple[str, ...] = (
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_SECURE",
    "MINIO_BUCKET",
    "HF_TOKEN",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_INSECURE_TLS",
    "MLFLOW_EXPERIMENT_NAME",
    "TAILSCALE_AUTHKEY",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    # ── NVIDIA judge API keys ────────────────────────────────────────────
    # Without these in os.environ the JudgePanel silently constructs every
    # client as None → all verdicts come back null. The set_env helper now
    # copies them out of Colab Secrets just like the storage / MLflow keys.
    "MISTRAL_LARGE_3_675B_API_KEY",
    "LLAMA_GUARD_4_12B_API_KEY",
    "NEMETRON_3_CONTENT_SAFETY_API_KEY",
    # NVIDIA tuning knobs (optional; safe defaults are baked in).
    "NVIDIA_DEFAULT_RPM",
    "MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS",
    # Optional notebook-publish overrides.
    "NOTEBOOK_NAME",
    "NOTEBOOK_PATH",
    # HF push gating.
    "PUSH_TO_HF",
    "HF_MODEL_REPO",
    "HF_REVISION",
)

_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "NEMETRON_3_CONTENT_SAFETY_API_KEY": ("NEMOTRON_3_CONTENT_SAFETY_API_KEY",),
}


def detect_runtime() -> Runtime:
    """Return ``"colab"`` or ``"local"`` for the current process."""
    try:
        # `google.colab` only exists inside the Colab runtime — no pip
        # package — so static checkers can't resolve it. We import for the
        # side-effect of triggering ImportError on non-Colab machines.
        import google.colab
        return "colab"
    except ImportError:
        return "local"


def load_env(*env_files: str | Path, override: bool = False) -> None:
    """Load environment variables from one or more ``.env`` files.

    - No files → no-op (``os.getenv`` keeps reading the real environment).
    - Multiple files are loaded left-to-right.
    - ``override=False`` (default) means real env vars win over file values.
    """
    if not env_files:
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required to load .env files. "
            "Install it with: pip install python-dotenv"
        ) from exc

    for path in env_files:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f".env file not found: {p}")
        load_dotenv(dotenv_path=str(p), override=override)


def _load_from_colab(keys: Iterable[str], override: bool) -> dict[str, str]:
    """Pull secrets from ``google.colab.userdata`` into ``os.environ``.

    Also accepts spelling aliases via ``_KEY_ALIASES`` — a Colab Secret
    named with the alias still ends up as the canonical env var. This is
    a one-line forgiveness for typos like ``NEMETRON`` vs ``NEMOTRON``.
    """
    from google.colab import userdata  # type: ignore[import-not-found]  # pylint: disable=import-error,no-name-in-module

    loaded: dict[str, str] = {}
    for key in keys:
        if not override and os.environ.get(key):
            continue
        value = None
        # Try the canonical name first, then any aliases.
        for candidate in (key, *_KEY_ALIASES.get(key, ())):
            try:
                value = userdata.get(candidate)
            except Exception:
                value = None
            if value is not None:
                break
        if value is None:
            continue
        # Strip whitespace — copying from a browser often picks up a trailing
        # newline. python-dotenv does this automatically for .env files; we
        # mirror that for Colab Secrets so a paste error doesn't cause silent
        # 401s ("Bearer <key>\n" gets rejected by most APIs).
        cleaned = str(value).strip()
        os.environ[key] = cleaned
        loaded[key] = cleaned
    return loaded


def _resolve_env_file(env_file: str | Path | None) -> Path | None:
    """Resolve the .env file to load locally — explicit path or auto-discovered."""
    if env_file is not None:
        p = Path(env_file)
        return p if p.exists() else None

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def set_env(
    keys: Iterable[str] | None = None,
    *,
    env_file: str | Path | None = None,
    runtime: Runtime | None = None,
    override: bool = False,
    quiet: bool = False,
) -> Runtime:
    """Auto-detect runtime and populate ``os.environ`` with required secrets.

    Behaviour by runtime:

    * **local** – loads ``env_file`` if given, otherwise walks parent
      directories looking for ``.env``. If none found it's a no-op (existing
      ``os.environ`` is used).
    * **colab** – reads each key from ``google.colab.userdata`` and sets it
      on ``os.environ``. Missing/unauthorized keys are silently skipped.

    Args:
        keys: Iterable of env var names to populate on Colab. Defaults to
            ``DEFAULT_KEYS`` (MinIO + HF + MLflow). Ignored on local.
        env_file: Optional explicit path to a ``.env`` file (local only).
        runtime: Force a specific runtime instead of auto-detecting.
        override: If True, secrets/files override values already in
            ``os.environ``. Default keeps existing env vars (CI-friendly).
        quiet: Suppress the one-line "loaded N secrets" message.

    Returns:
        The detected (or forced) runtime name.
    """
    rt: Runtime = runtime or detect_runtime()
    target_keys = tuple(keys) if keys is not None else DEFAULT_KEYS

    if rt == "local":
        path = _resolve_env_file(env_file)
        if path is not None:
            load_env(path, override=override)
            if not quiet:
                print(f"[set_env] local: loaded {path}")
        elif not quiet:
            print("[set_env] local: no .env file found, using existing os.environ")
        return rt

    if rt == "colab":
        loaded = _load_from_colab(target_keys, override=override)
        if not quiet:
            print(f"[set_env] colab: loaded {len(loaded)}/{len(target_keys)} secrets")
        return rt

    raise ValueError(f"Unknown runtime: {rt!r}")
