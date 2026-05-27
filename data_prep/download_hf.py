"""Download a HuggingFace dataset and save as JSONL.

Usage::

    python -m data_prep.download_hf \
        --dataset_name Amod/mental_health_counseling_conversations \
        --split train \
        --output_dir data_prep/raw/mental_health_counseling_conversations

Built-in presets (see :data:`PRESETS`) skip needing ``--dataset_name``::

    python -m data_prep.download_hf --preset counseling

The download is logged as an MLflow run tagged ``stage=dataset_download``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ._common import file_size_mb, setup_tracking, timestamp_run_name, write_jsonl


PRESETS: dict[str, str] = {
    "counseling": "Amod/mental_health_counseling_conversations",
    "chatbot": "heliosbrahma/mental_health_chatbot_dataset",
    "mentalchat16k": "PennShenLab/MentalChat16K",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a HuggingFace dataset → JSONL.")
    p.add_argument("--dataset_name", help="HF dataset repo id (owner/name).")
    p.add_argument("--preset", choices=sorted(PRESETS), help="Built-in dataset shortcut.")
    p.add_argument("--split", default="train", help="HF split (default: train).")
    p.add_argument("--output_dir", required=True, help="Directory for the resulting JSONL.")
    p.add_argument("--token", default=None, help="HF token (else $HF_TOKEN).")
    p.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = p.parse_args(argv)
    if not args.dataset_name and not args.preset:
        p.error("Pass --dataset_name or --preset.")
    if args.preset and not args.dataset_name:
        args.dataset_name = PRESETS[args.preset]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(f"error: {exc}\nInstall: pip install -r requirements_experiments.txt", file=sys.stderr)
        return 1

    token = args.token or os.getenv("HF_TOKEN")
    print(f"[hf] loading {args.dataset_name} split={args.split}")
    ds = load_dataset(args.dataset_name, split=args.split, token=token)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.split}.jsonl"
    n_records = write_jsonl((dict(row) for row in ds), output_path)

    columns = list(ds.column_names) if hasattr(ds, "column_names") else []
    size_mb = file_size_mb(output_path)
    print(f"[hf] wrote {n_records} records ({size_mb} MB) -> {output_path}")

    if args.no_mlflow:
        return 0

    try:
        tracking = setup_tracking("dataset_download")
        with tracking.run(
            run_name=timestamp_run_name(f"hf-{args.dataset_name.replace('/', '-')}"),
            tags={
                "stage": "dataset_download",
                "source": "huggingface",
                "dataset": args.dataset_name,
                "split": args.split,
            },
            with_hardware=False,
            log_system_metrics=False,
        ):
            tracking.log_params({
                "dataset_name": args.dataset_name,
                "split": args.split,
                "columns": ",".join(columns) if columns else "",
            })
            tracking.log_metrics({
                "num_rows": float(n_records),
                "size_mb": float(size_mb),
                "num_columns": float(len(columns)),
            })
            tracking.save_data(output_path, artifact_path="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"[hf] mlflow logging failed (continuing): {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
