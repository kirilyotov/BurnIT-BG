# data_platform

`data_platform` provides a modular interface for saving/loading model and dataset artifacts across:

- local filesystem
- MinIO (S3-compatible object storage)
- Hugging Face Hub
- MLflow tracking artifacts

It is designed around backend adapters + a high-level service + a CLI.

## Package Structure

- `common/`: shared config loading, env helpers, custom exceptions
- `storage/`: low-level storage adapters (`LocalStorage`, `MinioStorage`, `HuggingFaceStorage`)
- `tracking/`: MLflow adapter (`MLflowTracking`)
- `datasets/`: high-level orchestration service and CLI

## Environment Variables

Common variables used by this package:

- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET` (optional, default: `raw-data`)
- `MINIO_SECURE` (optional, `true|false`)
- `HF_TOKEN`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME` (optional)
- `MLFLOW_TRACKING_INSECURE_TLS` (optional, dev/self-signed only)

## CLI Usage

Run the package CLI via module entrypoint:

```bash
python -m data_platform.datasets --help
```

### Save model to Hugging Face

```bash
python -m data_platform.datasets \
  --env-file .env \
  save-model \
  --backend huggingface \
  --source ./artifacts/my_model \
  --repo-id your-user/my-model
```

### Save dataset to MinIO bucket

```bash
python -m data_platform.datasets \
  --env-file .env \
  save-data \
  --backend minio \
  --source ./data/processed \
  --path datasets/run_001/processed \
  --bucket raw-data
```

### Save model/data to MLflow artifacts

```bash
python -m data_platform.datasets \
  --env-file .env \
  --tracking-uri https://your-mlflow-host/mlflow/ \
  --experiment burnit-bg \
  save-model \
  --backend mlflow \
  --source ./artifacts/my_model \
  --artifact-path model
```

## Python Usage

### 1. Direct service usage

```python
import os
from data_platform.datasets.service import DatasetService
from data_platform.storage.local import LocalStorage
from data_platform.storage.minio import MinioStorage
from data_platform.storage.hugging_face import HuggingFaceStorage
from data_platform.tracking.mlflow import MLflowTracking

service = DatasetService(
    local_storage=LocalStorage(base_dir="./.local_storage"),
    minio_storage=MinioStorage.from_env(),
    hf_storage=HuggingFaceStorage(token=os.getenv("HF_TOKEN")),
    mlflow_tracking=MLflowTracking(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "default"),
        insecure_tls=os.getenv("MLFLOW_TRACKING_INSECURE_TLS", "false").lower() == "true",
    ),
)

# Save a model directory to Hugging Face
model_uri = service.save_model(
    source_dir="./artifacts/my_model",
    backend="huggingface",
    repo_id="your-user/my-model",
)
print(model_uri)

# Save a file to MinIO
data_uri = service.save_data(
    source_path="./data/train.jsonl",
    backend="minio",
    path="datasets/run_001/train.jsonl",
    bucket="raw-data",
)
print(data_uri)
```

### 2. Lower-level adapter usage

```python
from data_platform.storage.minio import MinioStorage

minio = MinioStorage.from_env()
uri = minio.save_file("./data/train.jsonl", "datasets/run_001/train.jsonl")
print(uri)
```

## Error Handling

Public operations can raise typed exceptions from `data_platform.common.exceptions`:

- `ConfigurationError`
- `StorageError`
- `TrackingError`

Catch `DataPlatformError` to handle all package-level errors in one place.
