#!/usr/bin/env bash
# Snapshot tmp/data_scraping/books.duckdb to parquet and upload both to MinIO.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

cd "${REPO_ROOT}"
exec "${PY}" -m data_scraping upload-metadata
