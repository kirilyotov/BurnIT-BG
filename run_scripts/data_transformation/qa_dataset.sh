#!/usr/bin/env bash
# Generate a REAL Bulgarian question->supportive-answer dataset from passages
# using an LLM (Mistral Large 3 on NVIDIA). Distinct from the template-based
# `from-passages` "style" dataset.
#
# The 675B model is slow (~20-60s/passage), so LIMIT defaults to 100. The
# passage-id cache means re-runs skip already-generated passages.
#
# Knobs:
#   INPUT=path/to/passages.jsonl   OUTPUT=path/to/qa.jsonl
#   MODEL=mistral-large-3  LIMIT=100  DELAY=0.0  MAX_TOKENS=1024
#   SKIP_MINIO=1   (write locally only)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

DATE="${DATE:-$(date +%F)}"
EXTRACT_DIR="${REPO_ROOT}/tmp/data_scraping/extracted"
INPUT="${INPUT:-}"
if [[ -z "${INPUT}" ]]; then
  INPUT="$(ls -S "${EXTRACT_DIR}"/*.jsonl 2>/dev/null | head -n1 || true)"
fi
OUTPUT="${OUTPUT:-${REPO_ROOT}/tmp/data_transformation/datasets/chitanka-qa/${DATE}/dataset.jsonl}"
MODEL="${MODEL:-mistral-large-3}"
LIMIT="${LIMIT:-100}"          # set to "" for ALL passages (slow + costly)
DELAY="${DELAY:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
CACHE="${CACHE:-${REPO_ROOT}/tmp/data_transformation/qa_cache.json}"

if [[ -z "${INPUT}" || ! -f "${INPUT}" ]]; then
  echo "error: no passages JSONL found. Run extract_chitanka_mental_health.sh first" >&2
  echo "       or set INPUT=path/to/passages.jsonl" >&2
  exit 1
fi

echo "[preset] input   = ${INPUT}"
echo "[preset] output  = ${OUTPUT}"
echo "[preset] model   = ${MODEL}   limit = ${LIMIT:-ALL}"

ARGS=( qa-from-passages
       --input "${INPUT}"
       --output "${OUTPUT}"
       --model "${MODEL}"
       --delay "${DELAY}"
       --max-tokens "${MAX_TOKENS}"
       --cache "${CACHE}"
       --date "${DATE}" )
if [[ -n "${LIMIT}" ]]; then ARGS+=( --limit "${LIMIT}" ); fi

SKIP_MINIO="${SKIP_MINIO:-1}"
if [[ "${SKIP_MINIO}" == "1" || "${SKIP_MINIO,,}" == "true" ]]; then ARGS+=( --skip-minio ); fi

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
