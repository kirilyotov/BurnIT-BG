"""Local filesystem implementation of the storage backend interface."""

from __future__ import annotations

import shutil
from pathlib import Path

from data_platform.common.exceptions import StorageError
from data_platform.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Filesystem-backed storage backend for local development and testing."""

    def __init__(self, base_dir: str | Path = "./.local_storage") -> None:
        """Create a filesystem-backed storage rooted at ``base_dir``."""

        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        """Resolve a backend-relative path to an absolute local filesystem path."""

        return self.base_dir / relative_path

    def save_file(self, local_path: str | Path, remote_path: str, **kwargs) -> str:
        """Copy a local file into the backend namespace and return destination path."""

        src = Path(local_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Local file not found: {src}")

        dst = self._resolve(remote_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            raise StorageError(f"Failed to copy '{src}' to '{dst}': {exc}") from exc
        return str(dst)

    def load_file(self, remote_path: str, local_path: str | Path, **kwargs) -> Path:
        """Copy a stored file from backend namespace to a local destination path."""

        src = self._resolve(remote_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Storage file not found: {src}")

        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            raise StorageError(f"Failed to copy '{src}' to '{dst}': {exc}") from exc
        return dst

    def save_directory(self, local_dir: str | Path, remote_prefix: str, **kwargs) -> str:
        """Copy a local directory recursively into the backend namespace."""

        src = Path(local_dir)
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"Local directory not found: {src}")

        dst = self._resolve(remote_prefix)
        dst.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Failed to copy directory '{src}' to '{dst}': {exc}") from exc
        return str(dst)

    def load_directory(self, remote_prefix: str, local_dir: str | Path, **kwargs) -> Path:
        """Copy a stored directory recursively from backend namespace to local path."""

        src = self._resolve(remote_prefix)
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"Storage directory not found: {src}")

        dst = Path(local_dir)
        dst.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Failed to copy directory '{src}' to '{dst}': {exc}") from exc
        return dst
