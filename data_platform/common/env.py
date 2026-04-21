"""Environment loading helpers for optional .env-based configuration."""

from __future__ import annotations

from pathlib import Path


def load_env(*env_files: str | Path, override: bool = False) -> None:
    """Load environment variables from one or more ``.env`` files.

    Behaviour:
    - If **no files** are given the function is a no-op; ``os.getenv()`` still
      reads from the actual process environment as usual.
    - When files *are* provided they are loaded left-to-right.
    - By default (``override=False``) variables that are **already set** in the
      process environment are *not* changed — so real env vars win over the
      file.  Pass ``override=True`` to let the file take precedence instead.

    Examples::

        from data_platform.common.env import load_env

        # Works without any .env file (just uses OS env vars)
        load_env()

        # Load a single .env file
        load_env(".env")

        # Load a base file then a local override (later file wins duplicates
        # only when override=True)
        load_env(".env", ".env.local", override=True)

        # Absolute paths work too
        load_env("/secrets/prod.env")
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
