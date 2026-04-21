# BurnIT-BG

BurnIT-BG is a machine-learning project focused on data and experimentation workflows for Bulgarian-language LLM use cases.

The repository provides:

- data collection and transformation areas
- reusable data/model storage and tracking package (`data_platform`)
- utility helpers for plotting and CLI/dataclass parsing (`utils`)
- experiment and test scaffolding

## Main Capabilities

- save/load models and datasets to local storage, MinIO, Hugging Face Hub, and MLflow artifacts
- run these operations via Python API and CLI
- track data/model lineage through MLflow run metadata
- generate common training and pruning plots for LLM workflows

## Quick Start

Install base dependencies:

```bash
pip install -r requirements.txt
```

For local development tools:

```bash
pip install -r requirements_local_dev.txt
```

## Configuration

Typical `.env` entries:

```dotenv
MINIO_ENDPOINT=host:9000
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
MINIO_SECURE=false
MINIO_BUCKET=raw-data

HF_TOKEN=...

MLFLOW_TRACKING_URI=https://your-mlflow-host/mlflow/
MLFLOW_EXPERIMENT_NAME=burnit-bg
MLFLOW_TRACKING_INSECURE_TLS=true
```

## Repository Organization

- `data_platform/`: reusable package for storage, tracking, and dataset/model transfer orchestration
	- `data_platform/common/`: config loaders, `.env` support, package exceptions
	- `data_platform/storage/`: backend adapters (local, MinIO, Hugging Face)
	- `data_platform/tracking/`: MLflow adapter for run and artifact operations
	- `data_platform/datasets/`: high-level `DatasetService` and CLI entrypoint
- `utils/`: reusable helper modules
	- `utils/plots.py`: plotting functions for training/pruning diagnostics
	- `utils/argparser.py`: dataclass-to-CLI helper
- `data_scraping/`: data acquisition components and scripts
- `data_transformation/`: transformation/cleaning logic for datasets
- `experiments/`: experiment scripts/notebooks and iteration workflows
- `docs/`: project documentation assets
- `test/`: integration and unit test structure
- `requirements*.txt`: dependency sets for runtime/dev/test/package
- `setup.py`: package definition and install metadata

## Package READMEs

- `data_platform/README.md`: architecture and usage examples (CLI + Python)
- `utils/README.md`: utility overview and examples

## Example CLI Usage

Save model to Hugging Face:

```bash
python -m data_platform.datasets \
	--env-file .env \
	save-model \
	--backend huggingface \
	--source ./artifacts/my_model \
	--repo-id your-user/my-model
```

Save data to MinIO bucket:

```bash
python -m data_platform.datasets \
	--env-file .env \
	save-data \
	--backend minio \
	--source ./data/processed \
	--path datasets/run_001/processed \
	--bucket raw-data
```

## Example Python Usage

```python
from data_platform.datasets.service import DatasetService
from data_platform.storage.minio import MinioStorage

service = DatasetService(minio_storage=MinioStorage.from_env())
uri = service.save_data(
		source_path="./data/train.jsonl",
		backend="minio",
		path="datasets/run_001/train.jsonl",
)
print(uri)
```
