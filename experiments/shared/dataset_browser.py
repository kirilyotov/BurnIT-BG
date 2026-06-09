"""Browse and pick datasets stored in MinIO from inside a notebook.

The notebooks need an ergonomic way to point at a previously-prepared
dataset that lives under some bucket prefix. This module gives them:

* :func:`list_datasets` – scan a bucket prefix for "dataset" folders
  (folders that contain at least one ``.jsonl`` / ``.parquet`` /
  ``.csv``).
* :func:`pick_dataset` – an ipywidgets dropdown when running in
  Jupyter/Colab, or a plain ``input()`` prompt otherwise.
* :func:`download_dataset` – pull the picked dataset locally with
  ``MinioStorage.load_directory``.

Typical notebook flow::

    from experiments.shared.dataset_browser import (
        list_datasets, pick_dataset, download_dataset,
    )

    options = list_datasets(prefix="datasets/processed")
    chosen = pick_dataset(options)                       # widget / prompt
    local_dir = download_dataset(chosen)                 # → Path
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATA_SUFFIXES = (".jsonl", ".ndjson", ".parquet", ".csv", ".json")


@dataclass(frozen=True)
class DatasetRef:
    """A pointer to a dataset folder in MinIO.

    ``prefix`` is the key prefix inside the bucket (e.g.
    ``datasets/processed/mental-health``). ``files`` lists the data
    files found under that prefix at discovery time.
    """
    bucket: str
    prefix: str
    files: tuple[str, ...]

    @property
    def name(self) -> str:
        """Last path segment — what to show in pickers."""
        return self.prefix.rstrip("/").rsplit("/", 1)[-1] or self.prefix

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix.rstrip('/')}"


def _connect():
    """Return a ready ``MinioStorage`` configured from env."""
    from data_platform.storage import MinioStorage
    return MinioStorage.from_env()


def list_datasets(
    prefix: str = "",
    *,
    bucket: str | None = None,
    suffixes: Iterable[str] = DATA_SUFFIXES,
    min_files: int = 1,
) -> list[DatasetRef]:
    """Discover dataset folders under ``prefix`` in the MinIO bucket.

    A "folder" here is the parent of any object whose suffix is in
    ``suffixes``. Returned list is sorted by prefix and deduplicated.
    """
    storage = _connect()
    target_bucket = bucket or storage.bucket
    objects = storage.list_objects(prefix=prefix, bucket=target_bucket)

    by_prefix: dict[str, list[str]] = {}
    suf_lower = tuple(s.lower() for s in suffixes)
    for obj_name in objects:
        if not obj_name.lower().endswith(suf_lower):
            continue
        # Group by the immediate parent prefix.
        parent = obj_name.rsplit("/", 1)[0] if "/" in obj_name else ""
        by_prefix.setdefault(parent, []).append(obj_name)

    refs = [
        DatasetRef(bucket=target_bucket, prefix=p, files=tuple(sorted(files)))
        for p, files in by_prefix.items()
        if len(files) >= min_files
    ]
    refs.sort(key=lambda r: r.prefix)
    return refs


def _is_notebook() -> bool:
    """Return True when called from an IPython / Jupyter kernel."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def pick_dataset(
    options: list[DatasetRef] | None = None,
    *,
    prefix: str = "",
    bucket: str | None = None,
    default: int = 0,
    auto: bool = False,
) -> DatasetRef:
    """Return one ``DatasetRef`` from a discovered list.

    If ``options`` is not provided, calls :func:`list_datasets`.

    Behaviour:
    * Inside a notebook (and ``auto=False``): renders an ipywidgets
      Dropdown when available; otherwise falls back to a numbered
      ``input()`` prompt.
    * Outside a notebook OR ``auto=True``: returns ``options[default]``.
    """
    if options is None:
        options = list_datasets(prefix=prefix, bucket=bucket)
    if not options:
        raise ValueError(f"No datasets found under prefix {prefix!r}.")
    if auto or not _is_notebook():
        return options[default]

    try:
        import ipywidgets as widgets
        from IPython.display import display

        dropdown = widgets.Dropdown(
            options=[(f"{ref.name}  ({len(ref.files)} files)", ref) for ref in options],
            value=options[default],
            description="Dataset:",
            layout=widgets.Layout(width="80%"),
        )
        confirm = widgets.Button(description="Use this dataset", button_style="primary")
        out_box = widgets.Output()
        result_box = {"ref": options[default]}

        def _on_click(_):
            result_box["ref"] = dropdown.value
            with out_box:
                out_box.clear_output()
                print(f"selected: {dropdown.value.s3_uri}")

        confirm.on_click(_on_click)
        display(widgets.VBox([dropdown, confirm, out_box]))
        # Synchronously return current selection — caller can re-run the
        # cell after picking, or use the printed URI directly.
        return dropdown.value
    except ImportError:
        pass

    # Plain stdin fallback.
    print(f"Datasets under {prefix!r}:")
    for i, ref in enumerate(options):
        marker = " (default)" if i == default else ""
        print(f"  [{i}] {ref.name}  ({len(ref.files)} files){marker}")
    try:
        raw = input(f"pick [0-{len(options) - 1}] (default {default}): ").strip()
    except EOFError:
        raw = ""
    idx = int(raw) if raw else default
    idx = max(0, min(idx, len(options) - 1))
    return options[idx]


def download_dataset(
    ref: DatasetRef,
    *,
    local_dir: str | Path | None = None,
    bucket: str | None = None,
) -> Path:
    """Download every file under ``ref.prefix`` to ``local_dir``."""
    storage = _connect()
    target_bucket = bucket or ref.bucket or storage.bucket
    if local_dir is None:
        local_dir = Path("./tmp/data") / ref.name
    local_dir = Path(local_dir)
    print(f"[browser] downloading {ref.s3_uri} → {local_dir}")
    return storage.load_directory(ref.prefix, local_dir, bucket=target_bucket)


def resolve(
    prefix: str | None = None,
    *,
    bucket: str | None = None,
    local_dir: str | Path | None = None,
    env_var: str = "DATASET_PREFIX",
    auto: bool = False,
) -> Path:
    """One-call helper: pick from the bucket and download to ``local_dir``.

    Resolution order for the prefix:

    1. ``prefix`` argument.
    2. ``$DATASET_PREFIX`` (or whatever ``env_var`` is set to).
    3. The default ``"data_prep/processed"``.

    Useful in notebooks::

        local = resolve()                    # interactive picker
        local = resolve(auto=True)           # pick default option
        local = resolve("datasets/processed/mental-health")  # explicit
    """
    prefix = prefix or os.environ.get(env_var) or "data_prep/processed"
    options = list_datasets(prefix=prefix, bucket=bucket)
    if not options:
        raise FileNotFoundError(
            f"No datasets found at s3://{bucket or '<default>'}/{prefix}. "
            "Run `python -m data_prep.upload_minio` first."
        )
    ref = pick_dataset(options, auto=auto)
    return download_dataset(ref, local_dir=local_dir, bucket=bucket)


__all__ = [
    "DatasetRef",
    "list_datasets",
    "pick_dataset",
    "download_dataset",
    "resolve",
]
