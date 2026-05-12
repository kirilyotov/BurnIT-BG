"""End-to-end HuggingFace dataset pipeline.

Downloads a HuggingFace dataset, translates specified fields with the
shared :mod:`data_transformation.translate` helper, and uploads both the
original snapshot and the translated copy to MinIO and (optionally) a
HuggingFace bucket.

Layout produced under ``{tmp_dir}/datasets/{slug}``::

    raw/   ← snapshot of the source repo (data + README + LICENSE + …)
    bg/    ← translated copies of data files only (.jsonl, .csv, .parquet)

These two folders are then mirrored to:

* ``s3://{bucket}/{minio_prefix}/{slug}/{raw,bg}/...``
* ``{hf_bucket}/{hf_prefix}/{slug}/{raw,bg}/...`` (Hugging Face bucket)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from huggingface_hub import HfApi, snapshot_download

from .io_utils import RecordWriter, get_nested, read_records, set_nested
from .translate import Translator, TranslatorConfig

log = logging.getLogger(__name__)

# Files we consider "data" files worth translating; everything else (README,
# LICENSE, .gitattributes) is shipped under raw/ but not translated.
TRANSLATABLE_SUFFIXES = {".jsonl", ".ndjson", ".json", ".csv", ".parquet"}
SKIP_FILENAME_PREFIXES = ("license", "readme", "copying", "notice", "changelog")


def dataset_slug(repo_id: str) -> str:
    """Turn ``owner/name`` into a filesystem-safe slug (``name``)."""
    name = repo_id.split("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def download_repo(repo_id: str, dest_dir: Path, token: Optional[str] = None) -> Path:
    """Snapshot a HuggingFace dataset repo into ``dest_dir/raw``."""
    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", repo_id, raw_dir)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
        token=token,
    )
    return raw_dir


def _translatable_files(raw_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in TRANSLATABLE_SUFFIXES:
            continue
        name_lc = p.stem.lower()
        if any(name_lc.startswith(prefix) for prefix in SKIP_FILENAME_PREFIXES):
            continue
        out.append(p)
    return out


def _translate_file(
    src: Path,
    dst: Path,
    translator: Translator,
    fields: Sequence[str],
) -> Tuple[int, int]:
    """Stream-translate one file. Returns ``(records, translated_strings)``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    translated = 0
    with RecordWriter(dst) as writer:
        for record in read_records(src):
            records += 1
            target_paths = list(fields) if fields else [k for k, v in record.items() if isinstance(v, str)]
            for path in target_paths:
                value = get_nested(record, path)
                if isinstance(value, str) and value.strip():
                    set_nested(record, path, translator.translate(value))
                    translated += 1
                elif isinstance(value, list):
                    new_list = []
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            new_list.append(translator.translate(item))
                            translated += 1
                        else:
                            new_list.append(item)
                    set_nested(record, path, new_list)
            writer.write(record)
    return records, translated


def translate_dataset(
    raw_dir: Path,
    out_dir: Path,
    fields: Sequence[str],
    source_lang: str = "en",
    target_lang: str = "bg",
    cache_path: Optional[Path] = None,
    delay: float = 0.0,
    output_format: str = "jsonl",
) -> int:
    """Translate every data file in ``raw_dir`` into ``out_dir``.

    ``output_format`` is the suffix (without the dot) for files that have
    an ambiguous source extension — by default we coerce everything to
    JSONL so the downstream consumer doesn't need format-sniffing logic.
    """
    cfg = TranslatorConfig(
        source=source_lang,
        target=target_lang,
        cache_path=cache_path,
        delay=delay,
    )
    translator = Translator(cfg)
    total_records = 0
    for src in _translatable_files(raw_dir):
        rel = src.relative_to(raw_dir)
        # `.json` files are usually JSONL on Hugging Face; force `.jsonl`
        # output so downstream tooling reads the file correctly.
        if src.suffix.lower() in {".json", ".jsonl", ".ndjson", ".txt"}:
            dst = out_dir / rel.with_suffix(f".{output_format}")
        else:
            dst = out_dir / rel
        log.info("translating %s -> %s", src, dst)
        records, translated = _translate_file(src, dst, translator, fields)
        total_records += records
        log.info("  %d records, %d strings translated", records, translated)
    return total_records


# --- Upload helpers ---------------------------------------------------------


def _iter_relative_files(local_dir: Path) -> Iterable[Tuple[Path, str]]:
    """Yield (path, posix-relative-path) pairs, skipping hidden dirs.

    Hugging Face's ``snapshot_download`` leaves a ``.cache/`` next to the
    payload; we don't want to mirror it.
    """
    for p in sorted(local_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir)
        if any(part.startswith(".") and part not in {".", ".."} for part in rel.parts):
            continue
        yield p, str(rel.as_posix())


def upload_to_minio(storage, local_root: Path, remote_prefix: str) -> int:
    """Recursively upload ``local_root`` to ``remote_prefix`` in MinIO.

    The remote key is ``{remote_prefix}/{relative_path}``.
    """
    storage.ensure_ready()
    prefix = remote_prefix.rstrip("/")
    count = 0
    for path, rel in _iter_relative_files(local_root):
        remote_key = f"{prefix}/{rel}"
        storage.save_file(path, remote_key)
        count += 1
    return count


def upload_to_hf_bucket(
    api: HfApi,
    bucket_id: str,
    local_root: Path,
    remote_prefix: str,
) -> int:
    """Recursively upload ``local_root`` to a HuggingFace bucket.

    Files land at ``{bucket_id}/{remote_prefix}/{relative_path}``.
    """
    prefix = remote_prefix.rstrip("/")
    pairs: List[Tuple[str, str]] = []
    for path, rel in _iter_relative_files(local_root):
        remote_path = f"{prefix}/{rel}" if prefix else rel
        pairs.append((str(path), remote_path))
    if not pairs:
        return 0
    log.info("uploading %d files to HF bucket '%s' (prefix '%s')", len(pairs), bucket_id, prefix)
    api.batch_bucket_files(bucket_id, add=pairs)
    return len(pairs)
