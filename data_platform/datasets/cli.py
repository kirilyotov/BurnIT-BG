"""Command-line interface for dataset and model save/load workflows."""

from __future__ import annotations

import argparse
import os
import sys

from data_platform.common.env import load_env
from data_platform.common.exceptions import DataPlatformError
from data_platform.storage.hugging_face import HuggingFaceStorage
from data_platform.storage.local import LocalStorage
from data_platform.storage.minio import MinioStorage
from data_platform.tracking.mlflow import MLflowTracking
from data_platform.datasets.service import DatasetService


def _try_minio_from_env() -> MinioStorage | None:
    """Try creating MinIO backend from environment, returning ``None`` if unavailable."""

    try:
        return MinioStorage.from_env()
    except (ValueError, DataPlatformError):
        return None


def _build_service(args: argparse.Namespace) -> DatasetService:
    """Build a DatasetService using CLI arguments and environment credentials."""

    local_storage = LocalStorage(base_dir=args.local_base_dir)
    minio_storage = _try_minio_from_env()
    hf_storage = HuggingFaceStorage(token=os.getenv("HF_TOKEN"))

    mlflow_tracking = None
    if args.tracking_uri:
        mlflow_tracking = MLflowTracking(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            insecure_tls=args.insecure_tls,
        )

    return DatasetService(
        local_storage=local_storage,
        minio_storage=minio_storage,
        hf_storage=hf_storage,
        mlflow_tracking=mlflow_tracking,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser for dataset/model operations."""

    parser = argparse.ArgumentParser(description="Dataset/model storage CLI")
    parser.add_argument(
        "--env-file",
        action="append",
        dest="env_files",
        metavar="PATH",
        default=[],
        help="Path to a .env file to load (can be repeated for multiple files; "
             "later files do NOT override earlier values unless --env-override is set)",
    )
    parser.add_argument(
        "--env-override",
        action="store_true",
        help="Allow .env files to override existing environment variables",
    )
    parser.add_argument("--local-base-dir", default="./.local_storage", help="Base directory for local backend")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")
    parser.add_argument("--experiment", default="default", help="MLflow experiment name")
    parser.add_argument("--insecure-tls", action="store_true", help="Set MLFLOW_TRACKING_INSECURE_TLS=true")

    sub = parser.add_subparsers(dest="command", required=True)

    save_model = sub.add_parser("save-model", help="Save model directory to a backend")
    save_model.add_argument("--source", required=True, help="Local source model directory")
    save_model.add_argument("--backend", choices=["local", "minio", "huggingface", "mlflow"], required=True)
    save_model.add_argument("--path", help="Destination path/prefix for local or minio")
    save_model.add_argument("--repo-id", help="HuggingFace repository id")
    save_model.add_argument("--artifact-path", default="model", help="MLflow artifact path")
    save_model.add_argument("--bucket", default=None, help="Optional MinIO bucket override")
    save_model.add_argument("--run-name", default="cli-save-model", help="MLflow run name for mlflow backend")

    load_model = sub.add_parser("load-model", help="Load model directory from a backend")
    load_model.add_argument("--target", required=True, help="Local target directory")
    load_model.add_argument("--backend", choices=["local", "minio", "huggingface", "mlflow"], required=True)
    load_model.add_argument("--path", help="Source path/prefix for local or minio")
    load_model.add_argument("--repo-id", help="HuggingFace repository id")
    load_model.add_argument("--run-id", help="MLflow run id")
    load_model.add_argument("--artifact-path", default="model", help="MLflow artifact path")
    load_model.add_argument("--bucket", default=None, help="Optional MinIO bucket override")
    load_model.add_argument("--revision", default=None, help="HF revision")

    save_data = sub.add_parser("save-data", help="Save file/directory data to a backend")
    save_data.add_argument("--source", required=True, help="Local source data file or directory")
    save_data.add_argument("--backend", choices=["local", "minio", "huggingface", "mlflow"], required=True)
    save_data.add_argument("--path", help="Destination path/prefix for local or minio")
    save_data.add_argument("--repo-id", help="HuggingFace dataset repository id")
    save_data.add_argument("--artifact-path", default="data", help="MLflow artifact path")
    save_data.add_argument("--bucket", default=None, help="Optional MinIO bucket override")
    save_data.add_argument("--run-name", default="cli-save-data", help="MLflow run name for mlflow backend")

    load_data = sub.add_parser("load-data", help="Load file/directory data from a backend")
    load_data.add_argument("--target", required=True, help="Local destination path")
    load_data.add_argument("--backend", choices=["local", "minio", "huggingface", "mlflow"], required=True)
    load_data.add_argument("--path", help="Source path/prefix for local or minio")
    load_data.add_argument("--repo-id", help="HuggingFace dataset repository id")
    load_data.add_argument("--run-id", help="MLflow run id")
    load_data.add_argument("--artifact-path", default="data", help="MLflow artifact path")
    load_data.add_argument("--bucket", default=None, help="Optional MinIO bucket override")
    load_data.add_argument("--revision", default=None, help="HF revision")
    load_data.add_argument("--is-dir", action="store_true", help="Treat local/minio path as directory")

    return parser


def main() -> None:
    """CLI entry point handling environment loading, dispatch, and user-facing errors."""

    parser = build_parser()
    args = parser.parse_args()

    # Load .env files before reading any environment variables.
    # Falls back to actual env vars automatically if no files are given.
    try:
        load_env(*args.env_files, override=args.env_override)
    except (FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        service = _build_service(args)
        _run_command(args, service)
    except DataPlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_command(args: argparse.Namespace, service: DatasetService) -> None:
    """Execute the selected CLI subcommand against the configured service."""

    if args.command == "save-model":
        if args.backend == "mlflow":
            if not service.mlflow:
                raise ValueError("MLflow backend selected but tracking is not configured. Pass --tracking-uri.")
            with service.mlflow.run(run_name=args.run_name):
                destination = service.save_model(
                    source_dir=args.source,
                    backend=args.backend,
                    path=args.path,
                    repo_id=args.repo_id,
                    artifact_path=args.artifact_path,
                    bucket=args.bucket,
                )
        else:
            destination = service.save_model(
                source_dir=args.source,
                backend=args.backend,
                path=args.path,
                repo_id=args.repo_id,
                artifact_path=args.artifact_path,
                bucket=args.bucket,
            )
        print(destination)
        return

    if args.command == "load-model":
        local_path = service.load_model(
            backend=args.backend,
            target_dir=args.target,
            path=args.path,
            repo_id=args.repo_id,
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            bucket=args.bucket,
            revision=args.revision,
        )
        print(local_path)
        return

    if args.command == "save-data":
        if args.backend == "mlflow":
            if not service.mlflow:
                raise ValueError("MLflow backend selected but tracking is not configured. Pass --tracking-uri.")
            with service.mlflow.run(run_name=args.run_name):
                destination = service.save_data(
                    source_path=args.source,
                    backend=args.backend,
                    path=args.path,
                    repo_id=args.repo_id,
                    artifact_path=args.artifact_path,
                    bucket=args.bucket,
                )
        else:
            destination = service.save_data(
                source_path=args.source,
                backend=args.backend,
                path=args.path,
                repo_id=args.repo_id,
                artifact_path=args.artifact_path,
                bucket=args.bucket,
            )
        print(destination)
        return

    if args.command == "load-data":
        local_path = service.load_data(
            backend=args.backend,
            target_path=args.target,
            path=args.path,
            repo_id=args.repo_id,
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            bucket=args.bucket,
            revision=args.revision,
            is_dir=args.is_dir,
        )
        print(local_path)
        return


if __name__ == "__main__":
    main()
