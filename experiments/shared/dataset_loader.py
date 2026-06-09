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


def get_dataset(
    name: str,
    *,
    source: str = "hf",
    subset: str | None = None,
    split_train: str = "train",
    split_eval: str = "validation",
    eval_fallback_take: int = 200,
) -> LoadedDataset:
    """Load a dataset by name + source.

    Parameters
    ----------
    name : str
        - ``source="hf"``   : HuggingFace repo id (``"org/name"``).
        - ``source="minio"``: MinIO prefix (``"datasets/processed/mental-health"``).
        - ``source="local"``: filesystem directory containing ``train.jsonl`` and ``eval.jsonl``.
    source : {"hf", "minio", "local"}
    subset : str, optional
        Config name for multi-config HF datasets (e.g. ``"unfiltered.nocontext"`` for TriviaQA).
    split_train, split_eval : str
        HF split names. Ignored for ``minio``/``local``.
    eval_fallback_take : int
        If the HF dataset lacks an eval split, slice this many rows off the head
        of the train split so the Trainer's eval loop still has something.
    """
    source = source.lower()

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
