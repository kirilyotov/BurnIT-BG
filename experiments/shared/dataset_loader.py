"""Unified dataset loader for experiment notebooks.

One entry point — ``get_dataset(name, source=...)`` — abstracts over the three
places experiment data can live:

* ``source="hf"``    — HuggingFace Hub dataset (e.g. ``kiplayo/burnit-bg-mental-health``)
* ``source="minio"`` — MinIO prefix (resolved via ``dataset_browser.resolve``)
* ``source="local"`` — Local directory with ``train.jsonl`` / ``eval.jsonl``

Returns a :class:`LoadedDataset` carrying the records, the resolved source URI
(useful for ``mlflow.log_input``), and ``to_mlflow_input(context=...)`` to build
an ``mlflow.data.Dataset`` for the run's Datasets tab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoadedDataset:
    """Train + eval records plus the metadata needed to register in MLflow."""

    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    name: str
    source: str
    source_uri: str
    subset: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        # Lets callers do `train, eval = get_dataset(...)`.
        yield self.train
        yield self.eval

    def to_mlflow_input(self, context: str = "train"):
        """Build an ``mlflow.data.Dataset`` for ``mlflow.log_input``.

        ``context`` selects which split is reported (``"train"`` | ``"eval"``).
        The resulting dataset shows up in the run's *Datasets* tab with the
        resolved ``source_uri`` so a reviewer can click through to where the
        data actually lives.
        """
        import mlflow.data
        import pandas as pd

        records = self.train if context == "train" else self.eval
        df = pd.DataFrame(records)
        ds_name = f"{self.name}@{context}"
        if self.subset:
            ds_name = f"{self.name}:{self.subset}@{context}"
        return mlflow.data.from_pandas(df, source=self.source_uri, name=ds_name)


_ALPACA_KEYS = ("instruction", "input", "output")
# Common alternative column names found across mental-health corpora.
_INSTRUCTION_ALIASES = (
    "instruction", "Context", "context", "question", "Question", "prompt",
    "Prompt", "user", "User", "human", "Human", "input_text",
)
_OUTPUT_ALIASES = (
    "output", "Response", "response", "answer", "Answer", "completion",
    "assistant", "Assistant", "bot", "output_text",
)


def _normalize_to_alpaca(rec: dict[str, Any]) -> dict[str, Any]:
    """Map heterogeneous QA-record column names to ``instruction/input/output``.

    Keeps the original keys around in ``metadata`` so nothing is lost — just
    ensures every record carries the canonical Alpaca shape the trainer and
    judge panel both expect.
    """
    if all(k in rec for k in _ALPACA_KEYS):
        return rec
    out_rec: dict[str, Any] = {}
    instr_val: str | None = None
    out_val: str | None = None
    for k in _INSTRUCTION_ALIASES:
        if k in rec and rec[k]:
            instr_val = str(rec[k])
            break
    for k in _OUTPUT_ALIASES:
        if k in rec and rec[k]:
            out_val = str(rec[k])
            break
    out_rec["instruction"] = instr_val or rec.get("instruction") or ""
    out_rec["input"] = rec.get("input", "")
    out_rec["output"] = out_val or rec.get("output") or ""
    # Preserve everything else as metadata for inspection.
    extras = {k: v for k, v in rec.items() if k not in {"instruction", "input", "output"}
              and k not in _INSTRUCTION_ALIASES and k not in _OUTPUT_ALIASES}
    if extras:
        out_rec["metadata"] = extras
    return out_rec


def _looks_like_registry_id(name: Any) -> bool:
    """Return True if ``name`` (string or list) refers to a registry id.

    Treats either an explicit comma-separated string of registry ids or a list
    that contains at least one known id as registry-routed. Pure single-name
    strings still need ``source="registry"`` to disambiguate from raw HF repo
    ids that happen to be alphanumeric.
    """
    try:
        from experiments.shared.datasets_registry import BY_ID
    except Exception:
        return False
    if isinstance(name, str) and "," in name:
        ids = [t.strip() for t in name.split(",") if t.strip()]
        return any(i in BY_ID for i in ids)
    if isinstance(name, list):
        return any(isinstance(t, str) and t in BY_ID for t in name)
    return False


def get_dataset(
    name: str,
    *,
    source: str = "hf",
    subset: str | None = None,
    split_train: str = "train",
    split_eval: str = "validation",
    eval_fallback_take: int = 200,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    file: str | None = None,
    eval_split_ratio: float = 0.1,
) -> LoadedDataset:
    """Load one or more datasets by name + source.

    Pass multiple names as a comma-separated string (or a list) to concatenate
    them — useful for training on mental-health + R-Tuning refusal records in
    one run. Records are interleaved with a deterministic shuffle so a small
    second dataset doesn't always sit at the tail.

    Parameters
    ----------
    name : str | list[str]
        - ``source="hf"``   : HuggingFace repo id (``"org/name"``) — or a comma-
          separated list (``"org/a,org/b"``) to concatenate multiple HF datasets.
        - ``source="minio"``: MinIO prefix (``"datasets/processed/mental-health"``);
          multi-source supported too.
        - ``source="local"``: filesystem directory containing ``train.jsonl`` and
          ``eval.jsonl``; multi-source supported.
    source : {"hf", "minio", "local"}
    subset : str, optional
        Config name for multi-config HF datasets (e.g. ``"unfiltered.nocontext"`` for TriviaQA).
        Applied to every name in a multi-dataset load.
    split_train, split_eval : str
        HF split names. Ignored for ``minio``/``local``.
    eval_fallback_take : int
        If a single HF dataset lacks an eval split, slice this many rows off the
        head of train so the Trainer's eval loop still has something.
    shuffle, shuffle_seed : bool, int
        When loading multiple datasets, interleave their records with a
        deterministic shuffle. Disable for ordered concatenation.
    """
    source = source.lower()

    # ── Registry lookup ──────────────────────────────────────────────────
    # source="registry" (or a comma-separated string of registry ids)
    # resolves to one or more DatasetSpec entries from datasets_registry.
    if source == "registry" or _looks_like_registry_id(name):
        from experiments.shared.datasets_registry import resolve_dataset_spec

        if isinstance(name, str) and "," in name:
            ids = [n.strip() for n in name.split(",") if n.strip()]
        elif isinstance(name, list):
            ids = [str(n).strip() for n in name if str(n).strip()]
        else:
            ids = [name if isinstance(name, str) else name]  # type: ignore[list-item]

        specs = [resolve_dataset_spec(t) for t in ids]
        if len(specs) == 1:
            s = specs[0]
            return get_dataset(
                s.location,
                source=s.source,
                subset=s.subset or subset,
                split_train=split_train,
                split_eval=split_eval,
                eval_fallback_take=eval_fallback_take,
                file=s.file,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
            )
        # Multi-spec: load each, concat, shuffle.
        import random
        parts = [
            get_dataset(
                s.location,
                source=s.source,
                subset=s.subset or subset,
                split_train=split_train,
                split_eval=split_eval,
                eval_fallback_take=eval_fallback_take,
                file=s.file,
                shuffle=False,
            )
            for s in specs
        ]
        train: list[dict[str, Any]] = []
        eval_: list[dict[str, Any]] = []
        per_source: dict[str, int] = {}
        for s, p in zip(specs, parts):
            train.extend(p.train)
            eval_.extend(p.eval)
            per_source[s.id] = len(p.train)
        if shuffle:
            rng = random.Random(shuffle_seed)
            rng.shuffle(train)
            rng.shuffle(eval_)
        return LoadedDataset(
            train=train,
            eval=eval_,
            name="+".join(s.id for s in specs),
            source="registry",
            source_uri=";".join(f"{s.source}://{s.location}" for s in specs),
            subset=subset,
            extras={"per_source": per_source, "specs": [s.id for s in specs]},
        )

    # Multi-dataset fan-out: comma-separated names or an explicit list.
    if isinstance(name, str) and "," in name:
        names = [n.strip() for n in name.split(",") if n.strip()]
    elif isinstance(name, list):
        names = [str(n).strip() for n in name if str(n).strip()]
    else:
        names = None

    if names and len(names) > 1:
        import random
        parts = [
            get_dataset(
                n,
                source=source,
                subset=subset,
                split_train=split_train,
                split_eval=split_eval,
                eval_fallback_take=eval_fallback_take,
                shuffle=False,
            )
            for n in names
        ]
        train: list[dict[str, Any]] = []
        eval_: list[dict[str, Any]] = []
        per_source: dict[str, int] = {}
        for p in parts:
            train.extend(p.train)
            eval_.extend(p.eval)
            per_source[p.name] = len(p.train)
        if shuffle:
            rng = random.Random(shuffle_seed)
            rng.shuffle(train)
            rng.shuffle(eval_)
        return LoadedDataset(
            train=train,
            eval=eval_,
            name="+".join(names),
            source=source,
            source_uri=";".join(p.source_uri for p in parts),
            subset=subset,
            extras={"per_source": per_source},
        )

    # Normalize single-name case.
    if names and len(names) == 1:
        name = names[0]

    if source == "hf-bucket":
        # HF *Buckets* are flat blob storage — not HF Datasets. We download
        # the specific ``file`` from bucket id ``name`` (e.g. ``"kiplayo/data"``),
        # parse the jsonl, and do a deterministic 90/10 train/eval split.
        import json
        import random
        import tempfile

        from data_platform.storage.hugging_face import HuggingFaceStorage

        if not file:
            raise ValueError(
                'source="hf-bucket" requires file=<path-in-bucket> '
                '(e.g. file="datasets/chitanka/bg/dataset.jsonl")'
            )
        storage = HuggingFaceStorage.from_env()
        tmp = Path(tempfile.mkdtemp(prefix="hf_bucket_"))
        local_path = tmp / Path(file).name
        storage.load_file_from_bucket(name, file, str(local_path))

        records: list[dict[str, Any]] = []
        with open(local_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(_normalize_to_alpaca(json.loads(line)))

        rng = random.Random(shuffle_seed)
        rng.shuffle(records)
        cut = max(1, int(len(records) * (1.0 - eval_split_ratio)))
        train, eval_ = records[:cut], records[cut:]
        return LoadedDataset(
            train=train,
            eval=eval_,
            name=f"{name}/{file}",
            source="hf-bucket",
            source_uri=f"hf-bucket://{name}/{file}",
            subset=subset,
            extras={"local_path": str(local_path), "total_records": len(records)},
        )

    if source == "hf":
        from datasets import load_dataset

        kwargs: dict[str, Any] = {}
        if subset:
            kwargs["name"] = subset
        ds = load_dataset(name, **kwargs)

        train = list(ds[split_train])
        if split_eval in ds:
            eval_ = list(ds[split_eval])
        elif "test" in ds:
            eval_ = list(ds["test"])
        else:
            eval_ = train[:eval_fallback_take]

        return LoadedDataset(
            train=train,
            eval=eval_,
            name=name,
            source="hf",
            source_uri=f"hf://datasets/{name}" + (f":{subset}" if subset else ""),
            subset=subset,
        )

    if source == "minio":
        from experiments.shared.dataset_browser import resolve
        from experiments.shared.dataset_utils import load_alpaca_dataset

        local_dir = resolve(prefix=name, auto=True)
        train = list(load_alpaca_dataset(local_dir / "train.jsonl"))
        eval_ = list(load_alpaca_dataset(local_dir / "eval.jsonl"))
        bucket = os.getenv("MINIO_BUCKET", "raw-data")
        return LoadedDataset(
            train=train,
            eval=eval_,
            name=name,
            source="minio",
            source_uri=f"s3://{bucket}/{name}",
            subset=subset,
            extras={"local_dir": str(local_dir)},
        )

    if source == "local":
        from experiments.shared.dataset_utils import load_alpaca_dataset

        root = Path(name)
        train = list(load_alpaca_dataset(root / "train.jsonl"))
        eval_path = root / "eval.jsonl"
        eval_ = list(load_alpaca_dataset(eval_path)) if eval_path.exists() else train[:eval_fallback_take]
        return LoadedDataset(
            train=train,
            eval=eval_,
            name=str(root),
            source="local",
            source_uri=f"file://{root.resolve()}",
            subset=subset,
        )

    raise ValueError(
        f"Unknown dataset source {source!r}. Expected one of: 'hf', 'minio', 'local'."
    )


__all__ = ["LoadedDataset", "get_dataset"]
