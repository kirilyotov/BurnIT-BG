"""MLflow tracking adapter for run lifecycle and artifact operations."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mlflow
from mlflow.exceptions import MlflowException

from data_platform.common.exceptions import TrackingError


class MLflowTracking:
    """Thin convenience wrapper around MLflow tracking and artifact APIs."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        insecure_tls: bool = False,
    ) -> None:
        """Configure MLflow tracking URI, experiment, and optional insecure TLS mode."""

        if insecure_tls:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
        except MlflowException as exc:
            raise TrackingError(f"Failed to configure MLflow: {exc}") from exc

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        log_system_metrics: bool = False,
    ) -> Iterator[mlflow.ActiveRun]:
        """Start an MLflow run context and apply optional tags."""

        try:
            with mlflow.start_run(run_name=run_name, log_system_metrics=log_system_metrics) as run:
                if tags:
                    mlflow.set_tags(tags)
                yield run
        except MlflowException as exc:
            raise TrackingError(f"MLflow run failed: {exc}") from exc

    @staticmethod
    def _require_active_run() -> str:
        """Return active run id or raise when no run is currently open."""

        run = mlflow.active_run()
        if not run:
            raise RuntimeError("An active MLflow run is required for this operation.")
        return run.info.run_id

    def log_source_uri(self, key: str, uri: str) -> None:
        """Record an external data source URI as an MLflow parameter."""

        mlflow.log_param(key, uri)

    def save_model(self, local_model_dir: str | Path, artifact_path: str = "model") -> str:
        """Upload model artifacts from local directory into active MLflow run."""

        run_id = self._require_active_run()
        try:
            mlflow.log_artifacts(str(local_model_dir), artifact_path=artifact_path)
        except MlflowException as exc:
            raise TrackingError(f"Failed to save model artifacts to MLflow: {exc}") from exc
        return f"runs:/{run_id}/{artifact_path}"

    def load_model(self, run_id: str, artifact_path: str = "model", dst_path: str | Path | None = None) -> Path:
        """Download model artifacts from a run into local storage."""

        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{artifact_path}",
                dst_path=str(dst_path) if dst_path else None,
            )
        except MlflowException as exc:
            raise TrackingError(f"Failed to download model artifacts from run '{run_id}': {exc}") from exc
        return Path(local_path)

    def save_data(self, local_path: str | Path, artifact_path: str = "data") -> str:
        """Upload data file or directory into active MLflow run artifacts."""

        run_id = self._require_active_run()
        local_path = Path(local_path)
        try:
            if local_path.is_dir():
                mlflow.log_artifacts(str(local_path), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        except MlflowException as exc:
            raise TrackingError(f"Failed to save data artifacts to MLflow: {exc}") from exc
        return f"runs:/{run_id}/{artifact_path}"

    def load_data(self, run_id: str, artifact_path: str = "data", dst_path: str | Path | None = None) -> Path:
        """Download data artifacts from a run into local storage."""

        try:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{artifact_path}",
                dst_path=str(dst_path) if dst_path else None,
            )
        except MlflowException as exc:
            raise TrackingError(f"Failed to download data artifacts from run '{run_id}': {exc}") from exc
        return Path(local_path)
