"""MinIO-backed implementation of storage backend operations."""

from __future__ import annotations

import os
from pathlib import Path
from io import BytesIO

from minio import Minio
from minio.error import S3Error

from data_platform.common.exceptions import ConfigurationError, StorageError
from data_platform.storage.base import StorageBackend


def create_client(url: str, access_key: str, secret_key: str, secure: bool = False) -> Minio:
    """Create a MinIO client from explicit credentials and endpoint."""

    return Minio(url, access_key=access_key, secret_key=secret_key, secure=secure)


class MinioStorage(StorageBackend):
    """MinIO-backed storage backend for files and model folders."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        auto_create_bucket: bool = True,
    ) -> None:
        """Initialize MinIO storage backend and target default bucket settings."""

        self.endpoint = endpoint
        self.bucket = bucket
        self.client = create_client(endpoint, access_key, secret_key, secure=secure)
        self.auto_create_bucket = auto_create_bucket

    @classmethod
    def from_env(cls, prefix: str = "MINIO") -> "MinioStorage":
        """Create MinIO storage from environment variables using ``prefix`` keys."""

        # secure = os.getenv(f"{prefix}_SECURE", "false").lower() == "true"
        # if not secure:
        #     os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
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

    def _ensure_bucket(self, bucket: str) -> None:
        """Ensure that target bucket exists, creating it when allowed."""

        try:
            exists = self.client.bucket_exists(bucket)
        except S3Error as exc:
            raise StorageError(f"Failed to check bucket '{bucket}': {exc}") from exc
        except Exception as exc:
            raise StorageError(f"MinIO connection error while checking bucket '{bucket}': {exc}") from exc
        if exists:
            return
        if not self.auto_create_bucket:
            raise StorageError(f"Bucket does not exist: {bucket}")
        try:
            self.client.make_bucket(bucket)
        except S3Error as exc:
            raise StorageError(f"Failed to create bucket '{bucket}': {exc}") from exc

    def _bucket(self, bucket: str | None) -> str:
        """Return explicit bucket override or fallback to default backend bucket."""

        return bucket or self.bucket

    def save_file(self, local_path: str | Path, remote_path: str, **kwargs) -> str:
        """Upload one file to MinIO and return its ``s3://`` URI."""

        local_file = Path(local_path)
        if not local_file.exists() or not local_file.is_file():
            raise FileNotFoundError(f"Local file not found: {local_file}")

        bucket = self._bucket(kwargs.get("bucket"))
        self._ensure_bucket(bucket)
        try:
            self.client.fput_object(bucket, remote_path, str(local_file))
        except S3Error as exc:
            raise StorageError(f"Failed to upload '{local_file}' to '{bucket}/{remote_path}': {exc}") from exc
        except OSError as exc:
            raise StorageError(f"Failed to read local file '{local_file}': {exc}") from exc
        return f"s3://{bucket}/{remote_path}"

    def load_file(self, remote_path: str, local_path: str | Path, **kwargs) -> Path:
        """Download one object from MinIO into a local file path."""

        bucket = self._bucket(kwargs.get("bucket"))
        local_file = Path(local_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.fget_object(bucket, remote_path, str(local_file))
        except S3Error as exc:
            raise StorageError(f"Failed to download '{bucket}/{remote_path}': {exc}") from exc
        except OSError as exc:
            raise StorageError(f"Failed to write local file '{local_file}': {exc}") from exc
        return local_file

    def save_directory(self, local_dir: str | Path, remote_prefix: str, **kwargs) -> str:
        """Upload all files under a directory to a MinIO prefix recursively."""

        local_root = Path(local_dir)
        if not local_root.exists() or not local_root.is_dir():
            raise FileNotFoundError(f"Local directory not found: {local_root}")

        bucket = self._bucket(kwargs.get("bucket"))
        self._ensure_bucket(bucket)
        prefix = remote_prefix.rstrip("/")

        for file_path in local_root.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(local_root).as_posix()
            object_name = f"{prefix}/{rel}" if prefix else rel
            try:
                self.client.fput_object(bucket, object_name, str(file_path))
            except S3Error as exc:
                raise StorageError(
                    f"Failed to upload '{file_path}' to '{bucket}/{object_name}': {exc}"
                ) from exc
        return f"s3://{bucket}/{prefix}"

    def load_directory(self, remote_prefix: str, local_dir: str | Path, **kwargs) -> Path:
        """Download all objects under a MinIO prefix into a local directory."""

        bucket = self._bucket(kwargs.get("bucket"))
        prefix = remote_prefix.rstrip("/")
        local_root = Path(local_dir)
        local_root.mkdir(parents=True, exist_ok=True)

        try:
            objects = list(self.client.list_objects(bucket, prefix=prefix, recursive=True))
        except S3Error as exc:
            raise StorageError(f"Failed to list objects in '{bucket}/{prefix}': {exc}") from exc

        for obj in objects:
            rel_part = obj.object_name[len(prefix):].lstrip("/") if prefix else obj.object_name
            target = local_root / rel_part
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.client.fget_object(bucket, obj.object_name, str(target))
            except S3Error as exc:
                raise StorageError(
                    f"Failed to download '{bucket}/{obj.object_name}': {exc}"
                ) from exc
        return local_root

    def save_bytes(self, data: bytes, remote_path: str, bucket: str | None = None) -> str:
        """Upload in-memory bytes directly to MinIO and return its ``s3://`` URI."""

        target_bucket = self._bucket(bucket)
        self._ensure_bucket(target_bucket)
        stream = BytesIO(data)
        self.client.put_object(target_bucket, remote_path, stream, len(data))
        return f"s3://{target_bucket}/{remote_path}"

    def load_bytes(self, remote_path: str, bucket: str | None = None) -> bytes:
        """Download object bytes from MinIO into memory."""

        target_bucket = self._bucket(bucket)
        response = self.client.get_object(target_bucket, remote_path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def list_objects(self, prefix: str = "", bucket: str | None = None) -> list[str]:
        """List object names stored under a MinIO prefix."""

        target_bucket = self._bucket(bucket)
        return [obj.object_name for obj in self.client.list_objects(target_bucket, prefix=prefix, recursive=True)]