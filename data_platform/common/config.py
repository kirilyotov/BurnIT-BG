"""Dataclass-based configuration loaders for MinIO and MLflow."""

from __future__ import annotations

import os
from dataclasses import dataclass

from data_platform.common.exceptions import ConfigurationError


@dataclass(slots=True)
class MinioConfig:
    """Configuration values required to connect to MinIO."""

    endpoint: str
    access_key: str
    secret_key: str
    bucket: str = "raw-data"
    secure: bool = False

    @classmethod
    def from_env(cls, prefix: str = "MINIO") -> "MinioConfig":
        """Build a MinIO config object from environment variables."""

        endpoint = os.getenv(f"{prefix}_ENDPOINT") or os.getenv("URL")
        access_key = os.getenv(f"{prefix}_ACCESS_KEY") or os.getenv("ACCESS_KEY")
        secret_key = os.getenv(f"{prefix}_SECRET_KEY") or os.getenv("SECRET_KEY")
        bucket = os.getenv(f"{prefix}_BUCKET", "raw-data")
        secure = os.getenv(f"{prefix}_SECURE", "false").lower() == "true"

        missing = [
            name
            for name, value in {
                "endpoint": endpoint,
                "access_key": access_key,
                "secret_key": secret_key,
            }.items()
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Missing MinIO env variables: {', '.join(missing)}")

        return cls(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            secure=secure,
        )


@dataclass(slots=True)
class MlflowConfig:
    """Configuration values required to connect to MLflow tracking."""

    tracking_uri: str
    experiment_name: str = "default"
    insecure_tls: bool = False

    @classmethod
    def from_env(cls, prefix: str = "MLFLOW") -> "MlflowConfig":
        """Build an MLflow config object from environment variables."""

        tracking_uri = os.getenv(f"{prefix}_TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ConfigurationError("Missing MLflow tracking URI (set MLFLOW_TRACKING_URI).")
        experiment_name = os.getenv(f"{prefix}_EXPERIMENT_NAME", "default")
        insecure_tls = os.getenv(f"{prefix}_INSECURE_TLS", "false").lower() == "true"
        return cls(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            insecure_tls=insecure_tls,
        )
