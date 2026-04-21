"""Shared exception hierarchy for data platform components."""

from __future__ import annotations


class DataPlatformError(Exception):
    """Base exception for all data platform errors."""


class StorageError(DataPlatformError):
    """Raised when a storage backend operation fails (I/O, permissions, network)."""


class ConfigurationError(DataPlatformError):
    """Raised when required configuration is missing or invalid."""


class TrackingError(DataPlatformError):
    """Raised when an experiment tracking operation fails."""
