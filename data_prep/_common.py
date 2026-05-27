"""Shared helpers for the ``data_prep/`` scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from data_platform.common import set_env
from data_platform.tracking import MLflowTracking


def setup_tracking(stage: str) -> MLflowTracking:
    """Load env + return an MLflowTracking instance configured for dataset jobs."""
    set_env(quiet=True)
    tracking = MLflowTracking.from_env(enable_system_metrics=False)
    tracking.set_experiment(__import__("os").environ.get("MLFLOW_EXPERIMENT_NAME", "datasets"))
    return tracking


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON records from a ``.jsonl`` file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    """Write records as JSONL; returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def file_size_mb(path: Path) -> float:
    """Path size in MB rounded to 2dp."""
    return round(path.stat().st_size / 1024 ** 2, 2) if path.exists() else 0.0


def timestamp_run_name(prefix: str) -> str:
    """Canonical run name: ``{prefix}-{utc-timestamp}``."""
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
