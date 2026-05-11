#!/usr/bin/env bash
# Preset wrapper for the nguyenletruongthien/mental-health Kaggle dataset.
# Snapshots the dataset, translates the configured FIELDS into Bulgarian,
# and pushes the raw + translated copies to MinIO and the HuggingFace bucket.
#
# All env vars below can be overridden at call time, e.g.
#   TARGET_LANG=de ./run_scripts/data_transformation/kaggle_mental_health_dataset.sh
#   SKIP_TRANSLATE=1 ./run_scripts/data_transformation/kaggle_mental_health_dataset.sh
#
# Kaggle credentials must be discoverable by kagglehub — either
# ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY env vars.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

# Baked-in defaults — what makes this script a preset.
HANDLE="${HANDLE:-nguyenletruongthien/mental-health}"
# Leave FIELDS empty by default — kaggle datasets don't have a stable
# schema, so the CLI's auto-detect (every top-level string field) is the
# safer default. Override per-dataset.
FIELDS="${FIELDS:-}"

# Generic knobs (same shape as hf_mental_health_counseling_conversations_dataset.sh).
SOURCE_LANG="${SOURCE_LANG:-en}"
TARGET_LANG="${TARGET_LANG:-bg}"
DELAY="${DELAY:-0.3}"
VERSION="${VERSION:-}"
BUCKET="${BUCKET:-}"
MINIO_PREFIX="${MINIO_PREFIX:-datasets/kaggle}"
HF_BUCKET="${HF_BUCKET:-data}"
HF_PREFIX="${HF_PREFIX:-datasets/kaggle}"

ARGS=( kaggle-dataset --handle "${HANDLE}"
       --source-lang "${SOURCE_LANG}" --target-lang "${TARGET_LANG}"
       --delay "${DELAY}"
       --minio-prefix "${MINIO_PREFIX}"
       --hf-bucket "${HF_BUCKET}" --hf-prefix "${HF_PREFIX}" )

if [[ -n "${FIELDS}" ]];  then ARGS+=( --fields "${FIELDS}" ); fi
if [[ -n "${BUCKET}" ]];  then ARGS+=( --bucket "${BUCKET}" ); fi
if [[ -n "${VERSION}" ]]; then ARGS+=( --version "${VERSION}" ); fi

for flag in FORCE_DOWNLOAD SKIP_DOWNLOAD SKIP_TRANSLATE SKIP_MINIO SKIP_HF; do
  val="${!flag:-0}"
  if [[ "${val}" == "1" || "${val,,}" == "true" ]]; then
    flag_lc="$(echo "${flag}" | tr '[:upper:]_' '[:lower:]-')"
    ARGS+=( "--${flag_lc}" )
  fi
done

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
