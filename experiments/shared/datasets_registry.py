"""Single source of truth for every named dataset BurnIT-BG knows about.

Each entry binds a short logical id (``chitanka-bg``) to its storage location
(HF Bucket / HF Dataset repo / MinIO prefix / local path). Notebooks reference
these ids — never raw paths — so a dataset migration is a one-line change here.

Usage::

    from experiments.shared.datasets_registry import Datasets, resolve_dataset_spec

    # In a notebook config cell:
    DATASET_NAME = Datasets.MENTAL_HEALTH_ALL.value.composite_name()
    DATASET_SOURCE = "registry"          # tell get_dataset() to look us up

    # …or as a single id:
    DATASET_NAME = "chitanka-bg"
    DATASET_SOURCE = "registry"

    # …or combine several:
    DATASET_NAME = "chitanka-bg,mh-counseling-bg,kaggle-mh-conversations-bg"
    DATASET_SOURCE = "registry"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class DatasetSpec:
    """Where a single named dataset actually lives + how to load it."""

    id: str                                # short logical name, the canonical key
    source: str                            # "hf-bucket" | "hf" | "minio" | "local"
    location: str                          # bucket_id / hf_repo_id / minio_prefix / local_dir
    file: str | None = None                # specific file under location (HF bucket / MinIO)
    description: str = ""
    language: str = "bg"
    license: str | None = None
    # For multi-config HF datasets (e.g. TriviaQA "unfiltered.nocontext").
    subset: str | None = None
    # Hint for the loader when records aren't already in Alpaca shape.
    record_format: str = "alpaca"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def composite_name(self) -> str:
        """Return ``self.id`` so a single spec can be passed as ``DATASET_NAME``."""
        return self.id


class Datasets(Enum):
    """Every named dataset the project uses, keyed by a short logical id.

    Add a new entry by appending a member here; nothing else changes.
    """

    # ── Bulgarian mental-health domain datasets ───────────────────────────
    CHITANKA_BG = DatasetSpec(
        id="chitanka-bg",
        source="hf-bucket",
        location="kiplayo/data",
        file="datasets/chitanka/bg/dataset.jsonl",
        description="Chitanka books → Bulgarian mental-health Q→A (Mistral-generated)",
        license="cc-by-nc-4.0",
        tags=("mental-health", "bulgarian", "q-and-a", "synthetic"),
    )
    MH_COUNSELING_BG = DatasetSpec(
        id="mh-counseling-bg",
        source="hf-bucket",
        location="kiplayo/data",
        file="datasets/huggingface/mental_health/combined_dataset.jsonl",
        description="HuggingFace mental-health counseling conversations, BG-translated",
        license="cc-by-nc-4.0",
        tags=("mental-health", "bulgarian", "counseling"),
    )
    KAGGLE_MH_CONVERSATIONS_BG = DatasetSpec(
        id="kaggle-mh-conversations-bg",
        source="hf-bucket",
        location="kiplayo/data",
        file="datasets/kaggle/mental-health/bg/conversations_training.jsonl",
        description="Kaggle mental-health conversations (training split), BG-translated",
        license="cc-by-nc-4.0",
        tags=("mental-health", "bulgarian", "conversations"),
    )

    # ── R-Tuning out-of-domain refusal datasets (BG) ──────────────────────
    RTUNING_TRIVIAQA_BG = DatasetSpec(
        id="rtuning-triviaqa-bg",
        source="hf",
        location="kiplayo/burnit-bg-rtuning-triviaqa-bg",
        description="R-Tuning refusal training derived from TriviaQA (BG)",
        license="apache-2.0",
        tags=("r-tuning", "refusal", "out-of-domain"),
    )
    RTUNING_SQUAD_BG = DatasetSpec(
        id="rtuning-squad-bg",
        source="hf",
        location="kiplayo/burnit-bg-rtuning-squad-bg",
        description="R-Tuning refusal training derived from SQuAD v2 (BG)",
        license="cc-by-sa-4.0",
        tags=("r-tuning", "refusal", "out-of-domain"),
    )
    RTUNING_COMBINED_BG = DatasetSpec(
        id="rtuning-combined-bg",
        source="hf",
        location="kiplayo/burnit-bg-rtuning-combined-bg",
        description="R-Tuning out-of-domain refusal (BG) — combined trivia + SQuAD",
        license="other",
        tags=("r-tuning", "refusal", "out-of-domain"),
    )


# Logical-id index: every spec by its ``.id`` string.
BY_ID: dict[str, DatasetSpec] = {m.value.id: m.value for m in Datasets}


# Convenience groupings the notebooks can reference by name.
MENTAL_HEALTH_ALL: tuple[str, ...] = (
    Datasets.CHITANKA_BG.value.id,
    Datasets.MH_COUNSELING_BG.value.id,
    Datasets.KAGGLE_MH_CONVERSATIONS_BG.value.id,
)


def resolve_dataset_spec(token: str | Datasets | DatasetSpec) -> DatasetSpec:
    """Turn anything that names a dataset into a concrete :class:`DatasetSpec`.

    Accepts a :class:`Datasets` enum member, a :class:`DatasetSpec`, or a string
    that's either a logical id (``"chitanka-bg"``) or the literal `.location`
    of an existing spec (back-compat for "the path-only world").
    """
    if isinstance(token, Datasets):
        return token.value
    if isinstance(token, DatasetSpec):
        return token
    if isinstance(token, str):
        token = token.strip()
        if token in BY_ID:
            return BY_ID[token]
        # Allow round-tripping: if someone passes the literal HF repo or bucket
        # path, find the spec that points there.
        for spec in BY_ID.values():
            if spec.location == token or (spec.file and spec.file == token):
                return spec
        raise KeyError(
            f"unknown dataset id {token!r}. Known: {sorted(BY_ID)}"
        )
    raise TypeError(f"cannot resolve dataset spec from {type(token).__name__}")


__all__ = [
    "Datasets", "DatasetSpec", "BY_ID", "MENTAL_HEALTH_ALL",
    "resolve_dataset_spec",
]
