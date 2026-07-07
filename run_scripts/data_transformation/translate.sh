#!/usr/bin/env bash
# Translate fields in a dataset (jsonl / json / csv / parquet / txt).
# Required env vars: INPUT, OUTPUT.
# Optional: FIELDS (comma-separated), SOURCE_LANG, TARGET_LANG, DELAY, CACHE.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

: "${INPUT:?Set INPUT=path/to/input.jsonl}"
: "${OUTPUT:?Set OUTPUT=path/to/output.jsonl}"
FIELDS="${FIELDS:-}"
SOURCE_LANG="${SOURCE_LANG:-en}"
TARGET_LANG="${TARGET_LANG:-bg}"
DELAY="${DELAY:-0.0}"
CACHE="${CACHE:-${REPO_ROOT}/tmp/data_transformation/translate_cache.json}"

ARGS=( translate --input "${INPUT}" --output "${OUTPUT}" \
       --source-lang "${SOURCE_LANG}" --target-lang "${TARGET_LANG}" \
       --cache "${CACHE}" --delay "${DELAY}" )
if [[ -n "${FIELDS}" ]]; then
  ARGS+=( --fields "${FIELDS}" )
fi

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
