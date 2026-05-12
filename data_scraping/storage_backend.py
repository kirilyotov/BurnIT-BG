"""Storage backend abstraction for book file storage.

Wraps :mod:`data_platform.storage` so the pipeline can target local disk,
MinIO, or Hugging Face with the same API. ``save_file(src, key)`` takes the
full destination key (e.g. ``raw/chitanka/2026-05-14/psychology/epub/6023.epub``)
— the wrapper does **not** rewrite paths.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from data_platform.storage.hugging_face import HuggingFaceStorage
from data_platform.storage.local import LocalStorage
from data_platform.storage.minio import MinioStorage


class BackendType(str, Enum):
    LOCAL = "local"
    MINIO = "minio"
    HUGGING_FACE = "huggingface"


class StorageBackend:
    """Tiny façade over the three concrete storage backends."""

    def __init__(
        self,
        backend: str = BackendType.LOCAL.value,
        bucket: Optional[str] = None,
    ) -> None:
        if backend not in {b.value for b in BackendType}:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Supported: {[b.value for b in BackendType]}"
            )
        self.backend = backend
        self.bucket = bucket

        if backend == BackendType.MINIO.value:
            store = MinioStorage.from_env()
            if bucket:
                store.bucket = bucket
            self.store = store
        elif backend == BackendType.HUGGING_FACE.value:
            self.store = HuggingFaceStorage.from_env()
        else:
            self.store = LocalStorage()

    @classmethod
    def minio(
        cls,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> "StorageBackend":
        """Build a MinIO-backed backend from explicit credentials."""
        instance = cls.__new__(cls)
        instance.backend = BackendType.MINIO.value
        instance.bucket = bucket
        instance.store = MinioStorage(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            secure=secure,
        )
        return instance

    def save_file(self, src_path: Path, remote_key: str) -> str:
        """Upload *src_path* to *remote_key* and return a URI string."""
        if self.backend in {BackendType.MINIO.value, BackendType.HUGGING_FACE.value}:
            return self.store.save_file(str(src_path), remote_key)
        return str(Path(src_path).resolve())

    def ensure_ready(self) -> None:
        """Best-effort bucket creation / connectivity check for MinIO."""
        if self.backend == BackendType.MINIO.value and self.bucket:
            self.store._ensure_bucket(self.bucket)

    def list_objects(self, prefix: str = "") -> list[str]:
        """Forward to the underlying store when available (MinIO)."""
        if self.backend == BackendType.MINIO.value:
            return self.store.list_objects(prefix=prefix)
        return []

    def _minio_key(self, uri_or_key: str) -> str:
        """Strip ``s3://{bucket}/`` from a URI; pass through if already a key."""
        if uri_or_key.startswith(f"s3://{self.bucket}/"):
            return uri_or_key[len(f"s3://{self.bucket}/"):]
        if uri_or_key.startswith("s3://"):
            rest = uri_or_key[len("s3://"):]
            slash = rest.find("/")
            return rest[slash + 1:] if slash >= 0 else rest
        return uri_or_key

    def load_bytes(self, uri_or_key: str) -> bytes:
        """Read a stored object into memory.

        Accepts ``s3://bucket/key`` URIs or a backend-relative key (MinIO) or
        an absolute filesystem path (local). For HuggingFace, downloads via
        ``load_file`` into a temp path and reads it back.
        """
        if self.backend == BackendType.MINIO.value:
            return self.store.load_bytes(self._minio_key(uri_or_key), bucket=self.bucket)
        if self.backend == BackendType.LOCAL.value:
            with open(uri_or_key, "rb") as fh:
                return fh.read()
        if self.backend == BackendType.HUGGING_FACE.value:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                self.store.load_file(uri_or_key, tmp.name)
                with open(tmp.name, "rb") as fh:
                    return fh.read()
        raise ValueError(f"Unknown backend: {self.backend}")

    def remove_object(self, uri_or_key: str) -> None:
        """Delete a stored object."""
        if self.backend == BackendType.MINIO.value:
            self.store.client.remove_object(self.bucket, self._minio_key(uri_or_key))
            return
        if self.backend == BackendType.LOCAL.value:
            Path(uri_or_key).unlink(missing_ok=True)
            return
        if self.backend == BackendType.HUGGING_FACE.value:
            raise NotImplementedError(
                "remove_object is not implemented for the huggingface backend"
            )
        raise ValueError(f"Unknown backend: {self.backend}")
