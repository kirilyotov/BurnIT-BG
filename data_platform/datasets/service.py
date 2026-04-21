"""High-level service orchestrating model and data transfer across backends."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from data_platform.storage.hugging_face import HuggingFaceStorage
from data_platform.storage.local import LocalStorage
from data_platform.storage.minio import MinioStorage
from data_platform.tracking.mlflow import MLflowTracking


class DatasetService:
    """Unified service for moving model/data artifacts between backends.

    Supported backends: local, minio, huggingface, mlflow.
    """

    def __init__(
        self,
        local_storage: LocalStorage | None = None,
        minio_storage: MinioStorage | None = None,
        hf_storage: HuggingFaceStorage | None = None,
        mlflow_tracking: MLflowTracking | None = None,
    ) -> None:
        """Initialize service with optional backend adapters."""

        self.local = local_storage
        self.minio = minio_storage
        self.hf = hf_storage
        self.mlflow = mlflow_tracking

    def _require(self, name: str, value: object) -> None:
        """Validate backend availability before performing an operation."""

        if value is None:
            raise ValueError(
                f"Backend '{name}' is not configured. Provide it explicitly or through environment variables."
            )

    def save_model(self, source_dir: str | Path, backend: str, **kwargs) -> str:
        """Save a model directory to the selected backend and return destination URI/path."""

        source = Path(source_dir)
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Model directory not found: {source}")

        if backend == "local":
            self._require("local", self.local)
            return self.local.save_directory(source, kwargs["path"])

        if backend == "minio":
            self._require("minio", self.minio)
            return self.minio.save_directory(source, kwargs["path"], bucket=kwargs.get("bucket"))

        if backend == "huggingface":
            self._require("huggingface", self.hf)
            return self.hf.save_model(
                local_dir=source,
                repo_id=kwargs["repo_id"],
                private=kwargs.get("private", False),
            )

        if backend == "mlflow":
            self._require("mlflow", self.mlflow)
            return self.mlflow.save_model(source, artifact_path=kwargs.get("artifact_path", "model"))

        raise ValueError(f"Unsupported backend: {backend}")

    def load_model(self, backend: str, target_dir: str | Path, **kwargs) -> Path:
        """Load a model directory from the selected backend into ``target_dir``."""

        target = Path(target_dir)

        if backend == "local":
            self._require("local", self.local)
            return self.local.load_directory(kwargs["path"], target)

        if backend == "minio":
            self._require("minio", self.minio)
            return self.minio.load_directory(kwargs["path"], target, bucket=kwargs.get("bucket"))

        if backend == "huggingface":
            self._require("huggingface", self.hf)
            return self.hf.load_model(model_id=kwargs["repo_id"], local_dir=target, revision=kwargs.get("revision"))

        if backend == "mlflow":
            self._require("mlflow", self.mlflow)
            return self.mlflow.load_model(
                run_id=kwargs["run_id"],
                artifact_path=kwargs.get("artifact_path", "model"),
                dst_path=target,
            )

        raise ValueError(f"Unsupported backend: {backend}")

    def save_data(self, source_path: str | Path, backend: str, **kwargs) -> str:
        """Save a data file or directory to the selected backend."""

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Data path not found: {source}")

        if backend == "local":
            self._require("local", self.local)
            if source.is_dir():
                return self.local.save_directory(source, kwargs["path"])
            return self.local.save_file(source, kwargs["path"])

        if backend == "minio":
            self._require("minio", self.minio)
            if source.is_dir():
                return self.minio.save_directory(source, kwargs["path"], bucket=kwargs.get("bucket"))
            return self.minio.save_file(source, kwargs["path"], bucket=kwargs.get("bucket"))

        if backend == "huggingface":
            self._require("huggingface", self.hf)
            if source.is_file():
                with TemporaryDirectory(prefix="hf_dataset_upload_") as temp_dir:
                    temp_path = Path(temp_dir)
                    copied = temp_path / source.name
                    copied.write_bytes(source.read_bytes())
                    return self.hf.save_dataset(local_dir=temp_path, dataset_id=kwargs["repo_id"])
            return self.hf.save_dataset(local_dir=source, dataset_id=kwargs["repo_id"])

        if backend == "mlflow":
            self._require("mlflow", self.mlflow)
            return self.mlflow.save_data(source, artifact_path=kwargs.get("artifact_path", "data"))

        raise ValueError(f"Unsupported backend: {backend}")

    def load_data(self, backend: str, target_path: str | Path, **kwargs) -> Path:
        """Load a data file or directory from selected backend into ``target_path``."""

        target = Path(target_path)

        if backend == "local":
            self._require("local", self.local)
            path = kwargs["path"]
            if kwargs.get("is_dir", True):
                return self.local.load_directory(path, target)
            return self.local.load_file(path, target)

        if backend == "minio":
            self._require("minio", self.minio)
            path = kwargs["path"]
            if kwargs.get("is_dir", True):
                return self.minio.load_directory(path, target, bucket=kwargs.get("bucket"))
            return self.minio.load_file(path, target, bucket=kwargs.get("bucket"))

        if backend == "huggingface":
            self._require("huggingface", self.hf)
            return self.hf.load_dataset(
                dataset_id=kwargs["repo_id"],
                local_dir=target,
                revision=kwargs.get("revision"),
            )

        if backend == "mlflow":
            self._require("mlflow", self.mlflow)
            return self.mlflow.load_data(
                run_id=kwargs["run_id"],
                artifact_path=kwargs.get("artifact_path", "data"),
                dst_path=target,
            )

        raise ValueError(f"Unsupported backend: {backend}")
