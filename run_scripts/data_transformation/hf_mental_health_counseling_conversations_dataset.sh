#!/usr/bin/env bash
# Preset wrapper for the Amod/mental_health_counseling_conversations dataset.
# Translates Context + Response from English to Bulgarian and pushes the
# raw snapshot plus the translation to MinIO and the HuggingFace bucket.
#
# All env vars below can be overridden at call time, e.g.
#   TARGET_LANG=de ./run_scripts/data_transformation/hf_mental_health_counseling_conversations_dataset.sh
#   SKIP_TRANSLATE=1 ./run_scripts/data_transformation/hf_mental_health_counseling_conversations_dataset.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

# Baked-in defaults — what makes this script a preset.
REPO="${REPO:-Amod/mental_health_counseling_conversations}"
FIELDS="${FIELDS:-Context,Response}"

# Generic knobs (same as hf_dataset.sh).
SOURCE_LANG="${SOURCE_LANG:-en}"
TARGET_LANG="${TARGET_LANG:-bg}"
DELAY="${DELAY:-0.3}"
BUCKET="${BUCKET:-}"
MINIO_PREFIX="${MINIO_PREFIX:-datasets/huggingface}"
HF_BUCKET="${HF_BUCKET:-data}"
HF_PREFIX="${HF_PREFIX:-datasets/huggingface}"

ARGS=( hf-dataset --repo "${REPO}" --fields "${FIELDS}"
       --source-lang "${SOURCE_LANG}" --target-lang "${TARGET_LANG}"
       --delay "${DELAY}"
       --minio-prefix "${MINIO_PREFIX}"
       --hf-bucket "${HF_BUCKET}" --hf-prefix "${HF_PREFIX}" )

if [[ -n "${BUCKET}" ]]; then ARGS+=( --bucket "${BUCKET}" ); fi

for flag in SKIP_DOWNLOAD SKIP_TRANSLATE SKIP_MINIO SKIP_HF; do
  val="${!flag:-0}"
  if [[ "${val}" == "1" || "${val,,}" == "true" ]]; then
    flag_lc="$(echo "${flag}" | tr '[:upper:]_' '[:lower:]-')"
    ARGS+=( "--${flag_lc}" )
  fi
done

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
