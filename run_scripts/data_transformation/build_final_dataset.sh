#!/usr/bin/env bash
# Build the CANONICAL Bulgarian dataset from the all-books extract and
# upload to MinIO, then split into per-topic + combined train/eval sets.
#
# Pipeline:
#   1. Pick the all-books extract:
#        tmp/data_scraping/extracted/all-books-no-limits-${DATE}.jsonl
#      Fallbacks (warned loudly):
#        - newest tmp/data_scraping/extracted/all-books-no-limits-*.jsonl
#        - largest *.jsonl in tmp/data_scraping/extracted/
#   2. Build the canonical Alpaca JSONL at
#        tmp/data_transformation/datasets/chitanka/final/bg/dataset.jsonl
#      and upload to s3://${MINIO_BUCKET:-data}/datasets/chitanka/final/bg/dataset.jsonl
#      (unless SKIP_MINIO=1).
#   3. Split into per-topic + combined sets under
#        tmp/data_transformation/datasets/chitanka/final/splits/
#
# Knobs:
#   DATE=YYYY-MM-DD     # default: today
#   MIN_WORDS=12        # passage-length filter
#   SKIP_MINIO=1        # build locally only, no upload
#   MINIO_BUCKET=data   # informational only (the python CLI reads its own env)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

DATE="${DATE:-$(date +%F)}"
EXTRACT_DIR="${REPO_ROOT}/tmp/data_scraping/extracted"
OUTPUT="${REPO_ROOT}/tmp/data_transformation/datasets/chitanka/final/bg/dataset.jsonl"
SPLITS_DIR="${REPO_ROOT}/tmp/data_transformation/datasets/chitanka/final/splits"
MIN_WORDS="${MIN_WORDS:-12}"
SKIP_MINIO="${SKIP_MINIO:-0}"
MINIO_BUCKET="${MINIO_BUCKET:-data}"

# --- Pick INPUT --------------------------------------------------------------
PREFERRED="${EXTRACT_DIR}/all-books-no-limits-${DATE}.jsonl"
INPUT=""

if [[ -f "${PREFERRED}" ]]; then
  INPUT="${PREFERRED}"
elif [[ -d "${EXTRACT_DIR}" ]]; then
  # Newest all-books-no-limits-*.jsonl
  CANDIDATE="$(ls -t "${EXTRACT_DIR}"/all-books-no-limits-*.jsonl 2>/dev/null | head -n1 || true)"
  if [[ -n "${CANDIDATE}" ]]; then
    echo "[warn] expected input not found: ${PREFERRED}" >&2
    echo "[warn] falling back to newest all-books extract: ${CANDIDATE}" >&2
    echo "[warn] (set DATE=YYYY-MM-DD to pin a specific run)" >&2
    INPUT="${CANDIDATE}"
  else
    # Largest *.jsonl in EXTRACT_DIR
    CANDIDATE="$(ls -S "${EXTRACT_DIR}"/*.jsonl 2>/dev/null | head -n1 || true)"
    if [[ -n "${CANDIDATE}" ]]; then
      echo "[warn] no all-books-no-limits-*.jsonl found under ${EXTRACT_DIR}" >&2
      echo "[warn] falling back to LARGEST extracted JSONL: ${CANDIDATE}" >&2
      echo "[warn] this is almost certainly NOT the canonical all-books extract" >&2
      echo "[warn] (set INPUT=path to choose explicitly)" >&2
      INPUT="${CANDIDATE}"
    fi
  fi
fi

if [[ -z "${INPUT}" || ! -f "${INPUT}" ]]; then
  echo "error: no input JSONL found under ${EXTRACT_DIR}" >&2
  echo "       run extract_chitanka_mental_health.sh (or the all-books extract) first," >&2
  echo "       or set INPUT=path explicitly" >&2
  exit 1
fi

INPUT_LINES="$(wc -l < "${INPUT}" 2>/dev/null || echo '?')"

# --- SKIP_MINIO flag ---------------------------------------------------------
SKIP_MINIO_FLAG=""
if [[ "${SKIP_MINIO}" == "1" || "${SKIP_MINIO,,}" == "true" ]]; then
  SKIP_MINIO_FLAG="--skip-minio"
fi

echo "[preset] date        = ${DATE}"
echo "[preset] input       = ${INPUT}  (${INPUT_LINES} passages)"
echo "[preset] output      = ${OUTPUT}"
echo "[preset] splits      = ${SPLITS_DIR}"
echo "[preset] min-words   = ${MIN_WORDS}"
echo "[preset] minio path  = s3://${MINIO_BUCKET}/datasets/chitanka/final/bg/dataset.jsonl"
if [[ -n "${SKIP_MINIO_FLAG}" ]]; then
  echo "[preset] minio       = SKIPPED (SKIP_MINIO=${SKIP_MINIO})"
fi

cd "${REPO_ROOT}"

# --- 1. Build canonical dataset ---------------------------------------------
BUILD_ARGS=( from-passages
             --input "${INPUT}"
             --output "${OUTPUT}"
             --source chitanka
             --date final/bg
             --min-words "${MIN_WORDS}" )
if [[ -n "${SKIP_MINIO_FLAG}" ]]; then
  BUILD_ARGS+=( "${SKIP_MINIO_FLAG}" )
fi

"${PY}" -m data_transformation "${BUILD_ARGS[@]}"

# --- 2. Split by topic -------------------------------------------------------
"${PY}" -m data_transformation split-by-topic \
  --input "${OUTPUT}" \
  --out-dir "${SPLITS_DIR}"

# --- 3. Summary --------------------------------------------------------------
OUT_LINES="$(wc -l < "${OUTPUT}")"
OUT_SIZE="$(du -h "${OUTPUT}" | cut -f1)"

echo "[done] local: ${OUTPUT}    (${OUT_LINES} records, ${OUT_SIZE})"
if [[ -n "${SKIP_MINIO_FLAG}" ]]; then
  echo "[done] minio: s3://${MINIO_BUCKET}/datasets/chitanka/final/bg/dataset.jsonl   (skipped: SKIP_MINIO=${SKIP_MINIO})"
else
  echo "[done] minio: s3://${MINIO_BUCKET}/datasets/chitanka/final/bg/dataset.jsonl"
fi
echo "[done] splits: ${SPLITS_DIR}"
