#!/usr/bin/env bash
# Drop Project Gutenberg books from MinIO + DuckDB whose detected language
# is not English. Defaults to a dry run; set APPLY=1 to actually delete.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

SOURCE="${SOURCE:-project_gutenberg}"
KEEP_LANG="${KEEP_LANG:-en}"
BACKEND="${BACKEND:-minio}"
BUCKET="${BUCKET:-}"
LIMIT="${LIMIT:-}"
# Set APPLY=1 to actually delete from the backend and DuckDB (defaults to dry run).
# Note: the script will print the number of books that would be removed in a dry run
APPLY="${APPLY:-1}"

ARGS=( filter-language --source "${SOURCE}" --keep-lang "${KEEP_LANG}" --backend "${BACKEND}" )
if [[ -n "${BUCKET}" ]]; then
  ARGS+=( --bucket "${BUCKET}" )
fi
if [[ -n "${LIMIT}" ]]; then
  ARGS+=( --limit "${LIMIT}" )
fi
if [[ "${APPLY}" == "1" || "${APPLY,,}" == "true" ]]; then
  ARGS+=( --apply )
fi

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
