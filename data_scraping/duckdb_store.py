"""
DuckDB metadata store for book ingestion pipeline.
Provides upsert and schema management for manifest records.
"""
from typing import Dict, Any
import duckdb
import json
from pathlib import Path

class DuckDBStore:
    """
    Handles DuckDB-based metadata storage for book manifests.
    """
    def __init__(self, db_path: Path):
        """
        Args:
            db_path: Path to the DuckDB database file.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self._ensure_schema()

    def _ensure_schema(self):
        """
        Create the manifest table if it does not exist.
        """
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS manifest (
            source VARCHAR,
            entry_id VARCHAR,
            book_id VARCHAR,
            title VARCHAR,
            authors VARCHAR,
            language VARCHAR,
            summary VARCHAR,
            categories VARCHAR,
            catalog_url VARCHAR,
            book_page_url VARCHAR,
            download_url VARCHAR,
            download_format VARCHAR,
            download_mime_type VARCHAR,
            retrieved_at VARCHAR,
            local_path VARCHAR,
            sha256 VARCHAR,
            file_size_bytes BIGINT,
            PRIMARY KEY (source, entry_id)
        )
        ''')

    def upsert_manifest_record(self, record: Dict[str, Any]):
        """
        Upsert a manifest record into DuckDB.
        Args:
            record: Manifest record dict.
        """
        # Convert lists to JSON strings for storage
        authors = json.dumps(record.get("authors", []), ensure_ascii=False)
        categories = json.dumps(record.get("categories", []), ensure_ascii=False)
        self.conn.execute('''
        INSERT INTO manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, entry_id) DO UPDATE SET
            book_id=excluded.book_id,
            title=excluded.title,
            authors=excluded.authors,
            language=excluded.language,
            summary=excluded.summary,
            categories=excluded.categories,
            catalog_url=excluded.catalog_url,
            book_page_url=excluded.book_page_url,
            download_url=excluded.download_url,
            download_format=excluded.download_format,
            download_mime_type=excluded.download_mime_type,
            retrieved_at=excluded.retrieved_at,
            local_path=excluded.local_path,
            sha256=excluded.sha256,
            file_size_bytes=excluded.file_size_bytes
        ''', [
            record.get("source"),
            record.get("entry_id"),
            record.get("book_id"),
            record.get("title"),
            authors,
            record.get("language"),
            record.get("summary"),
            categories,
            record.get("catalog_url"),
            record.get("book_page_url"),
            record.get("download_url"),
            record.get("download_format"),
            record.get("download_mime_type"),
            record.get("retrieved_at"),
            record.get("local_path"),
            record.get("sha256"),
            record.get("file_size_bytes"),
        ])

    def close(self):
        """
        Close the DuckDB connection.
        """
        self.conn.close()
