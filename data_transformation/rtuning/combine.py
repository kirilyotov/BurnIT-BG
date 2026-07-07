"""Interleave triviaqa-bg + squadv2-bg into a single combined R-Tuning jsonl.

Stable order — same inputs + same seed always produce the same combined file
(important for reproducibility of training runs).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CombineConfig:
    """Settings for :func:`combine_datasets`."""

    inputs: list[Path]
    output_jsonl: Path
    shuffle: bool = True
    seed: int = 42


def combine_datasets(cfg: CombineConfig) -> dict[str, int]:
    """Concatenate every ``inputs`` jsonl into one, with optional shuffle.

    Returns ``{"total": N, "per_source": {...}}``.
    """
    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    per_source: dict[str, int] = {}
    for in_path in cfg.inputs:
        if not in_path.exists():
            raise FileNotFoundError(f"combine input missing: {in_path}")
        n = 0
        with in_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
                n += 1
        per_source[in_path.stem] = n
        log.info("combine: loaded %d records from %s", n, in_path)

    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        rng.shuffle(records)

    with cfg.output_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("combine: wrote %d records -> %s", len(records), cfg.output_jsonl)
    return {"total": len(records), "per_source": per_source}


__all__ = ["CombineConfig", "combine_datasets"]
