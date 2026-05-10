"""Hugging Face Hub storage adapter for model and dataset repositories."""

from __future__ import annotations

from pathlib import Path

import huggingface_hub
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from data_platform.common.exceptions import StorageError


class HuggingFaceStorage:
    """
    Hugging Face Hub backend for model, dataset repositories, and bucket storage.
    """

    def __init__(self, token: str | None = None) -> None:
        """Create a Hugging Face Hub client optionally authenticated by token."""

        self.token = token
        self.api = huggingface_hub.HfApi(token=token)
        

    @classmethod
    def from_env(cls, token_env: str = "HF_TOKEN"):
        """
        Initialize HuggingFaceStorage from environment variable for token.
        Args:
            token_env: Environment variable for token.
        """
        import os
        token = os.environ.get(token_env)
        return cls(token=token)

    # --- Bucket API methods ---
    def save_file_to_bucket(self, bucket_id: str, local_path: str, remote_path: str):
        """
        Upload a single file to the Hugging Face bucket.
        """
        self.api.batch_bucket_files(bucket_id, add=[(local_path, remote_path)])
        
        
    def list_buckets(self):
        """
        List all buckets accessible with the current token.
        """
        return self.api.list_buckets()

    def save_directory_to_bucket(self, bucket_id: str, local_dir: str, remote_prefix: str = ""):
        """
        Recursively upload all files under local_dir to the bucket at remote_prefix.
        """
        import os
        add = []
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_dir)
                remote_file = f"{remote_prefix}/{rel_path}" if remote_prefix else rel_path
                add.append((local_file, remote_file))
        self.api.batch_bucket_files(bucket_id, add=add)

    def load_file_from_bucket(self, bucket_id: str, remote_path: str, local_path: str):
        """
        Download a single file from the Hugging Face bucket.
        """
        self.api.download_bucket_files(bucket_id, files=[(remote_path, local_path)])

    def load_directory_from_bucket(self, bucket_id: str, remote_prefix: str, local_dir: str):
        """
        Download all files under remote_prefix in the bucket to local_dir.
        """
        items = self.api.list_bucket_tree(bucket_id, prefix=remote_prefix, recursive=True)
        files = [(item.path, f"{local_dir}/{item.path[len(remote_prefix):].lstrip('/')}")
                 for item in items if item.type == "file"]
        self.api.download_bucket_files(bucket_id, files=files)

    def list_bucket_objects(self, bucket_id: str, prefix: str = "", recursive: bool = True):
        """
        List all objects in the bucket under prefix.
        """
        return list(self.api.list_bucket_tree(bucket_id, prefix=prefix, recursive=recursive))

    def save_model(
        self,
        local_dir: str | Path,
        repo_id: str,
        private: bool = False,
        create_repo_if_missing: bool = True,
        commit_message: str = "Upload model artifacts",
    ) -> str:
        """Upload a local model directory to a Hugging Face model repository."""

        local_path = Path(local_dir)
        if not local_path.exists() or not local_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {local_path}")

        try:
            if create_repo_if_missing:
                self.api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

            self.api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
        except HfHubHTTPError as exc:
            raise StorageError(f"Failed to upload model to HuggingFace repo '{repo_id}': {exc}") from exc
        return f"hf://models/{repo_id}"

    def load_model(self, model_id: str, local_dir: str | Path, revision: str | None = None) -> Path:
        """Download a model repository snapshot into a local directory."""

        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        try:
            huggingface_hub.snapshot_download(
                repo_id=model_id,
                repo_type="model",
                local_dir=str(local_path),
                revision=revision,
                token=self.token,
            )
        except RepositoryNotFoundError as exc:
            raise StorageError(f"HuggingFace model not found: '{model_id}'") from exc
        except HfHubHTTPError as exc:
            raise StorageError(f"Failed to download model '{model_id}': {exc}") from exc
        return local_path

    def save_dataset(
        self,
        local_dir: str | Path,
        dataset_id: str,
        private: bool = False,
        create_repo_if_missing: bool = True,
        commit_message: str = "Upload dataset artifacts",
    ) -> str:
        """Upload a local dataset directory to a Hugging Face dataset repository."""

        local_path = Path(local_dir)
        if not local_path.exists() or not local_path.is_dir():
            raise FileNotFoundError(f"Local dataset directory not found: {local_path}")

        try:
            if create_repo_if_missing:
                self.api.create_repo(repo_id=dataset_id, repo_type="dataset", private=private, exist_ok=True)

            self.api.upload_folder(
                folder_path=str(local_path),
                repo_id=dataset_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
        except HfHubHTTPError as exc:
            raise StorageError(f"Failed to upload dataset to HuggingFace repo '{dataset_id}': {exc}") from exc
        return f"hf://datasets/{dataset_id}"

    def load_dataset(self, dataset_id: str, local_dir: str | Path, revision: str | None = None) -> Path:
        """Download a dataset repository snapshot into a local directory."""

        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        try:
            huggingface_hub.snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=str(local_path),
                revision=revision,
                token=self.token,
            )
        except RepositoryNotFoundError as exc:
            raise StorageError(f"HuggingFace dataset not found: '{dataset_id}'") from exc
        except HfHubHTTPError as exc:
            raise StorageError(f"Failed to download dataset '{dataset_id}': {exc}") from exc
        return local_path


def download_model(model_id: str, local_dir: str | Path) -> Path:
    """Backward-compatible wrapper."""
    return HuggingFaceStorage().load_model(model_id=model_id, local_dir=local_dir)


def download_dataset(dataset_id: str, local_dir: str | Path) -> Path:
    """Backward-compatible wrapper."""
    return HuggingFaceStorage().load_dataset(dataset_id=dataset_id, local_dir=local_dir)
    