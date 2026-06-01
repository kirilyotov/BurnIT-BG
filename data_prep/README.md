# `data_prep/` — Dataset acquisition & preparation

Thin wrappers around `data_platform.storage` / `data_platform.datasets`
that make pulling and preparing the mental-health training data
straightforward at experiment time. Every script logs to MLflow
(experiment name: whatever `MLFLOW_EXPERIMENT_NAME` is set to in
`.env`, defaulting to `data_prep`).

> **Why not `datasets/`?** That name collides with the HuggingFace
> `datasets` library, which the notebooks need to import. We use a
> distinct package name so `from datasets import Dataset` always
> resolves to HuggingFace's, never to our local scripts.

## Scripts

| Script | What it does |
| --- | --- |
| `download_hf.py` | Pull a HuggingFace dataset → JSONL. Built-in presets for the mental-health datasets. |
| `download_kaggle.py` | Pull a Kaggle dataset via `kagglehub`, auto-convert CSV → Alpaca JSONL. |
| `download_minio.py` | Pull a prefix from MinIO into a local directory. |
| `upload_minio.py` | Push a local directory/file to MinIO. |
| `prepare_mental_health.py` | Merge all downloaded JSONLs into the canonical Alpaca schema, filter, split 90/10. |

## End-to-end example

```bash
# 1. Pull three sources (all log download stats to MLflow)
./venv/bin/python -m data_prep.download_hf --preset counseling \
    --output_dir data_prep/raw/counseling
./venv/bin/python -m data_prep.download_hf --preset chatbot \
    --output_dir data_prep/raw/chatbot
./venv/bin/python -m data_prep.download_kaggle --preset bhavik \
    --output_dir data_prep/raw/bhavik

# 2. Merge + filter + split into train/eval
./venv/bin/python -m data_prep.prepare_mental_health \
    --input_dir data_prep/raw \
    --output_dir data_prep/processed \
    --refusal_ratio 0.18

# 3. Push processed set to MinIO so notebooks can reload it
./venv/bin/python -m data_prep.upload_minio \
    --source_dir data_prep/processed \
    --prefix data_prep/processed/mental-health
```

## Canonical Alpaca record

Every record produced by `prepare_mental_health.py` matches the schema
in [`experiments/shared/dataset_utils.py`](../experiments/shared/dataset_utils.py):

```json
{
  "instruction": "...",
  "input": "",
  "output": "...",
  "category": "anxiety | depression | stress | grief | relationships | self-esteem | out_of_domain",
  "difficulty": "mild | moderate | severe",
  "source": "...",
  "quality_score": 0.85,
  "is_refusal": false,
  "language": "en | bg",
  "token_count": 142
}
```

## Credentials

| Source | How |
| --- | --- |
| HuggingFace | `HF_TOKEN` in `.env` (or pass `--token`). |
| Kaggle | `~/.kaggle/kaggle.json` *or* `KAGGLE_USERNAME` + `KAGGLE_KEY` in shell. |
| MinIO | `MINIO_*` in `.env`. See the main `README.md`. |

## Notes

- `download_kaggle.py` heuristically maps CSV columns to `instruction` / `output`
  (looks for column names like `question`, `prompt`, `answer`, `response`). When
  it can't auto-detect, it passes the raw row through unchanged and prints a warning
  — re-run `prepare_mental_health.py` after manually fixing the columns.
- `prepare_mental_health.py` mixes in synthetic refusal examples so the dataset
  is ready for R-Tuning (Notebook 04). Use `--refusal_ratio 0` to disable.
- All scripts accept `--no-mlflow` for offline use.
