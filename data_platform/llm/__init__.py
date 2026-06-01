"""Lightweight clients for the externally-hosted LLMs used as generators
and judges (NVIDIA-hosted, OpenAI-compatible chat models)."""

from .nvidia_client import (
    MODELS,
    NvidiaChatClient,
    NvidiaChatError,
    NvidiaModel,
    RpmBucket,
    extract_json,
    get_bucket,
)

__all__ = [
    "MODELS",
    "NvidiaChatClient",
    "NvidiaChatError",
    "NvidiaModel",
    "RpmBucket",
    "extract_json",
    "get_bucket",
]
