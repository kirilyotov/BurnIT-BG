#!/usr/bin/env bash
# Build the BurnIT-BG R-Tuning out-of-domain refusal training datasets
# end-to-end. For each source in SOURCE_LIST:
#   1. Download raw (slimmed) into tmp/
#   2. Upload raw to MinIO (and optionally HF)
#   3. Translate question + answer columns to BG (deep_translator, cached)
#   4. Assemble into Alpaca-style R-Tuning records with rotated BG refusals
#   5. Publish curated to MinIO (and optionally HF)
# Finally combine all curated jsonls into one combined dataset and publish that.
#
# By default this stages everything to MinIO; HF push requires PUSH_TO_HF=1.
#
# Knobs:
#   SOURCE_LIST="triviaqa squadv2"  (default: both)
#   SAMPLE_N=10000                    (cap rows per source; empty = full split)
#   SPLIT=train                       (HF split to fetch)
#   DATE=YYYY-MM-DD                   (default: today)
#   PUSH_TO_HF=0|1                    (default: 0 — stage to MinIO only)
#   PUSH_TO_MINIO=0|1                 (default: 1)
#   PRIVATE=0|1                       (mark HF repos private)
#   HF_USER=kiplayo                   (HF org/user for the dataset repos)
#   CONCURRENCY / FLUSH_EVERY         (translate knobs)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="python"
fi

DATE="2026-06-11"
SOURCE_LIST="${SOURCE_LIST:-triviaqa squadv2}"
SAMPLE_N="${SAMPLE_N:-}"
SPLIT="${SPLIT:-train}"
PUSH_TO_HF="${PUSH_TO_HF:-0}"
PUSH_TO_MINIO="${PUSH_TO_MINIO:-1}"
PRIVATE="${PRIVATE:-0}"
HF_USER="${HF_USER:-kiplayo}"
FLUSH_EVERY="${FLUSH_EVERY:-500}"
CONCURRENCY="${CONCURRENCY:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
# Translation backend: "google" (free, 5 req/sec hard cap) or "libretranslate"
# (self-hosted, unlimited). Spin up LibreTranslate with:
#   docker run -ti --rm -p 5000:5000 libretranslate/libretranslate --load-only en,bg
TRANSLATE_BACKEND="${TRANSLATE_BACKEND:-google}"
TRANSLATE_BACKEND_URL="${TRANSLATE_BACKEND_URL:-}"

ROOT_TMP="${REPO_ROOT}/tmp/data_transformation/datasets/rtuning"
CACHE="${REPO_ROOT}/tmp/data_transformation/rtuning/translate_cache.json"
mkdir -p "${ROOT_TMP}" "$(dirname "${CACHE}")"

LIMIT_ARG=()
if [[ -n "${SAMPLE_N}" ]]; then
  LIMIT_ARG=(--limit "${SAMPLE_N}")
fi

PRIVATE_ARG=()
if [[ "${PRIVATE}" == "1" ]]; then
  PRIVATE_ARG=(--private)
fi

PUSH_HF_ARG_UPLOAD=()
PUSH_HF_ARG_PUBLISH=()
if [[ "${PUSH_TO_HF}" == "1" ]]; then
  PUSH_HF_ARG_UPLOAD=(--push-hf)
  PUSH_HF_ARG_PUBLISH=(--push-hf)
fi

SKIP_MINIO_ARG=()
if [[ "${PUSH_TO_MINIO}" == "0" ]]; then
  SKIP_MINIO_ARG=(--skip-minio)
fi

echo "[rtuning] sources       = ${SOURCE_LIST}"
echo "[rtuning] split         = ${SPLIT}   sample = ${SAMPLE_N:-FULL}"
echo "[rtuning] date          = ${DATE}"
echo "[rtuning] push_minio    = ${PUSH_TO_MINIO}    push_hf = ${PUSH_TO_HF}"
echo "[rtuning] hf_user       = ${HF_USER}"

cd "${REPO_ROOT}"

CURATED_JSONLS=()

for SOURCE in ${SOURCE_LIST}; do
  echo
  echo "=============================================="
  echo "  ${SOURCE}"
  echo "=============================================="

  RAW_DIR="${ROOT_TMP}/${SOURCE}/${DATE}/raw"
  BG_DIR="${ROOT_TMP}/${SOURCE}/${DATE}/bg"
  mkdir -p "${RAW_DIR}" "${BG_DIR}"

  RAW_JSONL="${RAW_DIR}/${SOURCE}-${SPLIT}.jsonl"
  TRANS_JSONL="${BG_DIR}/${SOURCE}-${SPLIT}.translated.jsonl"
  CURATED_JSONL="${BG_DIR}/${SOURCE}-bg.jsonl"

  # ── 1. Download raw ───────────────────────────────────────────────────
  "${PY}" -m data_transformation rtuning-download \
    --source "${SOURCE}" \
    --output-dir "${RAW_DIR}" \
    --split "${SPLIT}" \
    "${LIMIT_ARG[@]}"

  # ── 2. Upload raw ────────────────────────────────────────────────────
  RAW_MINIO_PREFIX="datasets/rtuning/${SOURCE}/${DATE}/raw"
  HF_REPO_RAW="${HF_USER}/rtuning-${SOURCE}-raw"
  "${PY}" -m data_transformation rtuning-upload-raw \
    --source "${SOURCE}" \
    --input "${RAW_JSONL}" \
    --minio-prefix "${RAW_MINIO_PREFIX}" \
    --hf-repo "${HF_REPO_RAW}" \
    "${PUSH_HF_ARG_UPLOAD[@]}" \
    "${PRIVATE_ARG[@]}" \
    "${SKIP_MINIO_ARG[@]}"

  # ── 3. Translate ────────────────────────────────────────────────────
  TRANS_ARGS=(
    --input "${RAW_JSONL}"
    --output "${TRANS_JSONL}"
    --cache "${CACHE}"
    --backend "${TRANSLATE_BACKEND}"
    --concurrency "${CONCURRENCY}"
    --chunk-size "${CHUNK_SIZE}"
    --flush-every "${FLUSH_EVERY}"
  )
  if [[ -n "${TRANSLATE_BACKEND_URL}" ]]; then
    TRANS_ARGS+=(--backend-url "${TRANSLATE_BACKEND_URL}")
  fi
  "${PY}" -m data_transformation rtuning-translate "${TRANS_ARGS[@]}"

  # ── 4. Build Alpaca R-Tuning records ────────────────────────────────
  "${PY}" -m data_transformation rtuning-build \
    --input "${TRANS_JSONL}" \
    --output "${CURATED_JSONL}" \
    --rotate round-robin

  # ── 5. Publish per-source curated ───────────────────────────────────
  CURATED_MINIO_PREFIX="datasets/rtuning/${SOURCE}/${DATE}/bg"
  HF_REPO_BG="${HF_USER}/burnit-bg-rtuning-${SOURCE/squadv2/squad}-bg"
  "${PY}" -m data_transformation rtuning-publish \
    --input "${CURATED_JSONL}" \
    --minio-prefix "${CURATED_MINIO_PREFIX}" \
    --hf-repo "${HF_REPO_BG}" \
    --sources "${SOURCE}" \
    "${PUSH_HF_ARG_PUBLISH[@]}" \
    "${PRIVATE_ARG[@]}" \
    "${SKIP_MINIO_ARG[@]}"

  CURATED_JSONLS+=("${CURATED_JSONL}")
done

# ── 6. Combine all per-source curated jsonls ────────────────────────────
if [[ "${#CURATED_JSONLS[@]}" -gt 1 ]]; then
  echo
  echo "=============================================="
  echo "  combined"
  echo "=============================================="

  COMBINED_DIR="${ROOT_TMP}/combined/${DATE}/bg"
  mkdir -p "${COMBINED_DIR}"
  COMBINED_JSONL="${COMBINED_DIR}/combined-bg.jsonl"

  "${PY}" -m data_transformation rtuning-combine \
    --inputs "${CURATED_JSONLS[@]}" \
    --output "${COMBINED_JSONL}"

  COMBINED_MINIO_PREFIX="datasets/rtuning/combined/${DATE}/bg"
  HF_REPO_COMBINED="${HF_USER}/burnit-bg-rtuning-combined-bg"
  SOURCES_CSV="$(IFS=,; echo "${SOURCE_LIST// /,}")"
  "${PY}" -m data_transformation rtuning-publish \
    --input "${COMBINED_JSONL}" \
    --minio-prefix "${COMBINED_MINIO_PREFIX}" \
    --hf-repo "${HF_REPO_COMBINED}" \
    --sources "${SOURCES_CSV}" \
    "${PUSH_HF_ARG_PUBLISH[@]}" \
    "${PRIVATE_ARG[@]}" \
    "${SKIP_MINIO_ARG[@]}"
fi

echo
echo "[rtuning] done. curated jsonls:"
for p in "${CURATED_JSONLS[@]}"; do echo "  ${p}"; done
if [[ -n "${COMBINED_JSONL:-}" ]]; then echo "  ${COMBINED_JSONL}"; fi
