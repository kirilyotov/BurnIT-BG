# BurnIT-BG

BurnIT-BG is a research project building a **Bulgarian-language mental-health peer-support LLM**, end-to-end: data scraping, dataset transformation, fine-tuning/compression/alignment experiments, and a 3-judge evaluation panel — all tracked in MLflow.

The repository provides:

- data scraping and transformation pipelines for Bulgarian text
- reusable data/model storage and tracking package (`data_platform`)
- utility helpers for plotting and CLI/dataclass parsing (`utils`)
- notebook experiments (fine-tune, compression, safety, alignment)
- a 3-judge LLM evaluation panel (NVIDIA-hosted: Mistral Large 3 + Llama Guard 4 + Nemotron Content Safety)

## Main Capabilities

- save/load models and datasets to local storage, MinIO, Hugging Face Hub, and MLflow artifacts
- run these operations via Python API and CLI
- track data/model lineage and prompt versions through MLflow
- generate common training and pruning plots for LLM workflows
- run reproducible experiments via shared notebook scaffolding ([_notebook_builder.py](experiments/_notebook_builder.py))

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
MLFLOW_TRACKING_URI=...
MLFLOW_EXPERIMENT_NAME=...
MLFLOW_TRACKING_INSECURE_TLS=...
TAILSCALE_AUTHKEY=...
MISTRAL_LARGE_3_675B_API_KEY=...
LLAMA_GUARD_4_12B_API_KEY=...
NEMETRON_3_CONTENT_SAFETY_API_KEY=...
BYTEDANCE_SEED_OSS_36B_INSTRUCT_API_KEY=...
```
## Huggingface resources for the project
[Data bucket](https://huggingface.co/buckets/kiplayo/data)

## Documentation

- [Used Resources](docs/guideline/resources/UsedResources.md) — papers, repos, datasets, and courses the project draws on
- [Experiments & HuggingFace guide](docs/experiments_and_huggingface.md) — 12-notebook plan, judge panel, MLflow wiring, HF publishing strategy
- Environment setup: [Python](docs/guideline/envirement/Python.md), [CUDA](docs/guideline/envirement/Cuda.md), [PyTorch](docs/guideline/envirement/Pytorch.md), [HuggingFace](docs/guideline/envirement/HuggingFace.md), [MLflow](docs/guideline/envirement/Mlflow.md)
- IDE / compute: [Google Colab](docs/guideline/ide_and_computation/GoogleColab.md), [Local](docs/guideline/ide_and_computation/Local.md), [VSCode extensions](docs/guideline/ide_and_computation/VsCodeExtentions.md)
- Run scripts: [Data scraping](docs/guideline/run_scripts/DataScraping.md), [Data transformation](docs/guideline/run_scripts/DataTransformation.md)


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
- `data_transformation/`: transformation/cleaning logic for datasets (style rewrite, Q→A, topic split, publish)
- `experiments/`: experiment notebooks (fine-tune, compression, safety, alignment) and the shared notebook scaffolding
- `api_judges_scripts/`: raw copy-pasteable calls for the NVIDIA-hosted judge models
- `spaces/`: Hugging Face Space app (`burnit-bg-demo`) for the trained model
- `run_scripts/`: shell entrypoints for data pipeline steps
- `docs/`: project documentation (see [Documentation](#documentation))
- `test/`: integration and unit test structure
- `requirements*.txt`: dependency sets for runtime/dev/test/package/experiments
- `setup.py`: package definition and install metadata

## Package READMEs

- `data_platform/README.md`: architecture and usage examples (CLI + Python)
- `utils/README.md`: utility overview and examples


