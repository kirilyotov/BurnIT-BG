"""Small compatibility helpers for MinIO file transfer via configured backend."""

from __future__ import annotations

from pathlib import Path

from data_platform.storage.minio import MinioStorage


def upload_file(local_path: str | Path, object_path: str, bucket: str | None = None) -> str:
    """Upload a local file to MinIO using environment-based backend settings."""

    storage = MinioStorage.from_env()
    return storage.save_file(local_path=local_path, remote_path=object_path, bucket=bucket)


def download_file(object_path: str, local_path: str | Path, bucket: str | None = None) -> Path:
    """Download a MinIO object to local path using environment-based backend settings."""

    storage = MinioStorage.from_env()
    return storage.load_file(remote_path=object_path, local_path=local_path, bucket=bucket)
