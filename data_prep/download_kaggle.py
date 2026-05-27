"""Download a Kaggle dataset and normalize CSV → JSONL (Alpaca shape).

Uses :mod:`kagglehub`. Requires Kaggle credentials — either
``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY`` env vars.

Usage::

    python -m data_prep.download_kaggle \
        --dataset bhavikjikadara/mental-health-dataset \
        --output_dir data_prep/raw/bhavikjikadara_mental_health

Or use a preset::

    python -m data_prep.download_kaggle --preset bhavik
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Iterable

from ._common import file_size_mb, setup_tracking, timestamp_run_name, write_jsonl


PRESETS: dict[str, str] = {
    "bhavik": "bhavikjikadara/mental-health-dataset",
}

# Heuristic column mapping: which CSV column maps to which Alpaca field.
# We accept lowercased exact matches.
INSTRUCTION_COLUMNS = ("instruction", "prompt", "question", "context", "input", "user")
OUTPUT_COLUMNS = ("output", "response", "answer", "completion", "reply", "assistant")


def _find_column(cols: list[str], candidates: Iterable[str]) -> str | None:
    lc_to_orig = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lc_to_orig:
            return lc_to_orig[cand]
    return None


def _csv_to_alpaca_jsonl(csv_path: Path, jsonl_path: Path, source: str) -> int:
    """Convert a CSV to Alpaca-style JSONL. Returns rows written."""
    from experiments.shared.dataset_utils import make_record

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        instr_col = _find_column(cols, INSTRUCTION_COLUMNS)
        out_col = _find_column(cols, OUTPUT_COLUMNS)
        if not instr_col or not out_col:
            print(
                f"  warn: could not auto-detect instruction/output columns in {csv_path.name}; "
                f"columns were {cols!r}. Falling back to raw passthrough.",
                file=sys.stderr,
            )
            return write_jsonl((dict(row) for row in reader), jsonl_path)

        def _records() -> Iterable[dict]:
            for row in reader:
                instruction = (row.get(instr_col) or "").strip()
                output = (row.get(out_col) or "").strip()
                if not instruction or not output:
                    continue
                yield make_record(
                    instruction=instruction,
                    output=output,
                    source=source,
                )

        return write_jsonl(_records(), jsonl_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a Kaggle dataset → Alpaca JSONL.")
    p.add_argument("--dataset", help="Kaggle handle owner/dataset-name.")
    p.add_argument("--preset", choices=sorted(PRESETS), help="Built-in dataset shortcut.")
    p.add_argument("--output_dir", required=True, help="Where to write JSONL output.")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    if not args.dataset and not args.preset:
        p.error("Pass --dataset or --preset.")
    if args.preset and not args.dataset:
        args.dataset = PRESETS[args.preset]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    try:
        import kagglehub
    except ImportError as exc:
        print(f"error: {exc}\nInstall: pip install -r requirements_experiments.txt", file=sys.stderr)
        return 1

    print(f"[kaggle] downloading {args.dataset}")
    cache_path = Path(kagglehub.dataset_download(args.dataset))
    # Copy cached files into our project tree
    if cache_path.is_file():
        shutil.copy2(cache_path, raw_dir / cache_path.name)
    else:
        for src in cache_path.rglob("*"):
            if not src.is_file():
                continue
            target = raw_dir / src.relative_to(cache_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    print(f"[kaggle] raw files in {raw_dir}")

    # Convert every CSV to JSONL (Alpaca shape). Non-CSV files are left under raw/.
    total_records = 0
    written_files: list[Path] = []
    for csv_path in sorted(raw_dir.rglob("*.csv")):
        jsonl_path = out_dir / f"{csv_path.stem}.jsonl"
        n = _csv_to_alpaca_jsonl(csv_path, jsonl_path, source=args.dataset)
        total_records += n
        written_files.append(jsonl_path)
        print(f"  {csv_path.name} -> {jsonl_path.name} ({n} rows, {file_size_mb(jsonl_path)} MB)")

    if not written_files:
        print("[kaggle] no CSVs found — review raw/ manually.", file=sys.stderr)

    if args.no_mlflow:
        return 0

    try:
        tracking = setup_tracking("dataset_download")
        with tracking.run(
            run_name=timestamp_run_name(f"kaggle-{args.dataset.replace('/', '-')}"),
            tags={
                "stage": "dataset_download",
                "source": "kaggle",
                "dataset": args.dataset,
            },
            with_hardware=False,
            log_system_metrics=False,
        ):
            tracking.log_params({"dataset": args.dataset})
            tracking.log_metrics({
                "num_records": float(total_records),
                "num_files": float(len(written_files)),
                "size_mb_total": float(sum(file_size_mb(p) for p in written_files)),
            })
            for p in written_files:
                tracking.save_data(p, artifact_path="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"[kaggle] mlflow logging failed (continuing): {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
