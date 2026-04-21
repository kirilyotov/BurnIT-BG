"""Abstract interface for pluggable artifact storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Minimal backend interface for file and directory operations."""

    @abstractmethod
    def save_file(self, local_path: str | Path, remote_path: str, **kwargs) -> str:
        """Store a local file into backend location and return a URI/path."""

    @abstractmethod
    def load_file(self, remote_path: str, local_path: str | Path, **kwargs) -> Path:
        """Load a backend file into a local destination path."""

    @abstractmethod
    def save_directory(self, local_dir: str | Path, remote_prefix: str, **kwargs) -> str:
        """Store a local directory recursively."""

    @abstractmethod
    def load_directory(self, remote_prefix: str, local_dir: str | Path, **kwargs) -> Path:
        """Load a backend prefix recursively into a local directory."""
