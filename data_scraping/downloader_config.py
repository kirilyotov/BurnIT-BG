from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DownloaderConfig:
    feed_url: str
    limit: int = 20
    source: str = "chitanka"
    delay: float = 1.0
    out_dir: str = "data/raw"
    duckdb_path: Optional[str] = None
    duckdb_table: str = "books"
    bucket: Optional[str] = None
    backend: str = "local"  # local, minio, huggingface
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_secure: bool = False
    minio_bucket: Optional[str] = None
    # Add more backend-specific fields as needed
