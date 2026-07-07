#!/usr/bin/env bash
# Rewrite each record's instruction with Mistral so it actually matches its
# passage. Reads the dataset in concurrent batches and writes
# {OUTPUT_NAME}.jsonl under a DATED path (today by default), then uploads to
# MinIO at datasets/chitanka/{DATE}/bg/{OUTPUT_NAME}.jsonl.
#
# SLOW (Mistral 675B ~10-20s/call × concurrency). Cache makes it resumable —
# kill at any time, re-run, already-rewritten passages are skipped.
#
# Knobs:
#   INPUT=path/to/dataset.jsonl          (default: final/bg/dataset.jsonl)
#   OUTPUT=path/to/dataset_ai_improved.jsonl
#   OUTPUT_NAME=dataset_ai_improved      (default basename)
#   DATE=YYYY-MM-DD                       (default: today)
#   CONCURRENCY=10  BATCH_SIZE=50  MODEL=mistral-large-3
#   SKIP_MINIO=1                          (write locally only)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

DATE="${DATE:-$(date +%F)}"
INPUT="${INPUT:-${REPO_ROOT}/tmp/data_transformation/datasets/chitanka/final/bg/dataset.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-dataset_ai_improved}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/tmp/data_transformation/datasets/chitanka/${DATE}/bg/${OUTPUT_NAME}.jsonl}"
CACHE="${CACHE:-${REPO_ROOT}/tmp/data_transformation/rewrite_cache.json}"
MODEL="${MODEL:-mistral-large-3}"
CONCURRENCY="${CONCURRENCY:-10}"
BATCH_SIZE="${BATCH_SIZE:-50}"

if [[ ! -f "${INPUT}" ]]; then
  echo "error: input dataset not found: ${INPUT}" >&2
  exit 1
fi

INPUT_RECS="$(wc -l < "${INPUT}" 2>/dev/null || echo '?')"
echo "[preset] input        = ${INPUT}  (${INPUT_RECS} records)"
echo "[preset] output       = ${OUTPUT}"
echo "[preset] cache        = ${CACHE}"
echo "[preset] model        = ${MODEL}"
echo "[preset] concurrency  = ${CONCURRENCY}    batch-size = ${BATCH_SIZE}"

ARGS=( rewrite-instructions
       --input "${INPUT}"
       --output "${OUTPUT}"
       --model "${MODEL}"
       --concurrency "${CONCURRENCY}"
       --batch-size "${BATCH_SIZE}"
       --cache "${CACHE}"
       --date "${DATE}/bg"
       --source chitanka )

if [[ "${SKIP_MINIO:-0}" == "1" || "${SKIP_MINIO:-}" == "true" ]]; then
  ARGS+=( --skip-minio )
fi

mkdir -p "$(dirname "${OUTPUT}")"
cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
