#!/usr/bin/env bash
# End-to-end: chitanka -> gutenberg -> upload DuckDB manifest.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "--- chitanka ---"
"${HERE}/chitanka.sh"

echo
echo "--- project gutenberg ---"
"${HERE}/gutenberg.sh"

echo
echo "--- upload duckdb manifest to minio ---"
"${HERE}/upload_metadata.sh"

echo
echo "done."
