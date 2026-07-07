#!/usr/bin/env bash
# Build a Bulgarian mental-health Alpaca dataset from the passages extracted
# in the previous step (data_scraping extract-passages).
#
# Pipeline:
#   1. Read tmp/data_scraping/extracted/passages-{PASSAGES_NAME}.jsonl
#      (or whatever INPUT points at).
#   2. For each passage, pair it with a topic-appropriate Bulgarian
#      instruction template. Output = the book passage.
#   3. Apply quality filters (min words, dangerous-phrase drop).
#   4. Write Alpaca JSONL locally and upload to MinIO under
#      datasets/{SOURCE}/{DATASET_NAME}/bg/dataset.jsonl.
#
# Naming knobs (the whole point of this script):
#   PASSAGES_NAME=my-run-v2     # which passages file to read from
#                               # → tmp/data_scraping/extracted/passages-my-run-v2.jsonl
#   DATASET_NAME=clean-set-1    # what to call the produced dataset
#                               # → tmp/data_transformation/datasets/chitanka/clean-set-1/bg/dataset.jsonl
#                               # → s3://{bucket}/datasets/chitanka/clean-set-1/bg/dataset.jsonl
#   (Both default to DATE which defaults to today, so the legacy single-date
#   flow still works.)
#
# Override individual paths if you want:
#   INPUT=path/to/passages.jsonl     OUTPUT=path/to/dataset.jsonl
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

DATE="${DATE:-$(date +%F)}"
# PASSAGES_NAME = date segment in the EXTRACT step's output path.
# PASSAGES_FILENAME = filename portion (no extension) of the extracted file.
PASSAGES_NAME="${PASSAGES_NAME:-${DATE}}"
PASSAGES_FILENAME="${PASSAGES_FILENAME:-passages}"

# DATASET_NAME = date segment in the BUILD step's output path.
# FILENAME = filename portion of the dataset (no extension).
#   local: tmp/data_transformation/datasets/{SOURCE}/{DATASET_NAME}/{FILENAME}.jsonl
#   minio: datasets/{SOURCE}/{DATASET_NAME}/{FILENAME}.jsonl
DATASET_NAME="${DATASET_NAME:-${DATE}}"
FILENAME="${FILENAME:-dataset}"
SOURCE="${SOURCE:-chitanka}"

INPUT="${INPUT:-${REPO_ROOT}/tmp/data_scraping/extracted/${PASSAGES_FILENAME}-${PASSAGES_NAME}.jsonl}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/tmp/data_transformation/datasets/${SOURCE}/${DATASET_NAME}/${FILENAME}.jsonl}"

SEED="${SEED:-42}"
# Book passages are short; 12 keeps far more valid material than the old 30.
MIN_WORDS="${MIN_WORDS:-12}"
MAX_CHARS="${MAX_CHARS:-1500}"
QUALITY_SCORE="${QUALITY_SCORE:-0.80}"
BUCKET="${BUCKET:-}"

EXTRACT_DIR="${REPO_ROOT}/tmp/data_scraping/extracted"
# If the conventional input file isn't there, fall back to the LARGEST
# extracted JSONL so a plain run doesn't silently build from a tiny/old file
# (the classic "my dataset is only a few KB" trap). Set INPUT to override.
if [[ ! -f "${INPUT}" && -d "${EXTRACT_DIR}" ]]; then
  AUTODETECT="$(ls -S "${EXTRACT_DIR}"/*.jsonl 2>/dev/null | head -n1 || true)"
  if [[ -n "${AUTODETECT}" ]]; then
    echo "[warn] expected input not found: ${INPUT}" >&2
    echo "[warn] auto-selecting largest extracted file: ${AUTODETECT}" >&2
    echo "[warn] (set INPUT=... or PASSAGES_FILENAME/PASSAGES_NAME to choose explicitly)" >&2
    INPUT="${AUTODETECT}"
  fi
fi

if [[ ! -f "${INPUT}" ]]; then
  echo "error: input passages JSONL not found: ${INPUT}" >&2
  echo "       run extract_chitanka_mental_health.sh first," >&2
  echo "       or set INPUT=path or PASSAGES_NAME=<extracted-name>" >&2
  exit 1
fi

INPUT_PASSAGES="$(wc -l < "${INPUT}" 2>/dev/null || echo '?')"

echo "[preset] passages date = ${PASSAGES_NAME}  filename = ${PASSAGES_FILENAME}"
echo "[preset] dataset date  = ${DATASET_NAME}   filename = ${FILENAME}"
echo "[preset] input         = ${INPUT}  (${INPUT_PASSAGES} passages)"
echo "[preset] output        = ${OUTPUT}"
echo "[preset] min-words      = ${MIN_WORDS}"
echo "[preset] minio path    = datasets/${SOURCE}/${DATASET_NAME}/${FILENAME}.jsonl"

ARGS=( from-passages
       --input "${INPUT}"
       --output "${OUTPUT}"
       --source "${SOURCE}"
       --date "${DATASET_NAME}"
       --seed "${SEED}"
       --min-words "${MIN_WORDS}"
       --max-chars "${MAX_CHARS}"
       --quality-score "${QUALITY_SCORE}" )

if [[ -n "${BUCKET}" ]]; then ARGS+=( --bucket "${BUCKET}" ); fi

for flag in ALLOW_DANGEROUS SKIP_MINIO; do
  val="${!flag:-0}"
  if [[ "${val}" == "1" || "${val,,}" == "true" ]]; then
    flag_lc="$(echo "${flag}" | tr '[:upper:]_' '[:lower:]-')"
    ARGS+=( "--${flag_lc}" )
  fi
done

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
