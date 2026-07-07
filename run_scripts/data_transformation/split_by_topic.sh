#!/usr/bin/env bash
# Split a built Alpaca dataset into per-topic + combined train/eval sets.
#
#   {OUT_DIR}/all/train.jsonl, all/eval.jsonl
#   {OUT_DIR}/by-topic/{topic}/train.jsonl, eval.jsonl
#
# Knobs:
#   INPUT=path/to/dataset.jsonl   OUT_DIR=path/to/splits
#   EVAL_RATIO=0.1  SEED=42  MIN_PER_TOPIC=10  STRATIFY_BY=category
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

DATE="${DATE:-$(date +%F)}"
INPUT="${INPUT:-${REPO_ROOT}/tmp/data_transformation/datasets/chitanka/${DATE}/dataset.jsonl}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/tmp/data_transformation/splits/${DATE}}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
SEED="${SEED:-42}"
MIN_PER_TOPIC="${MIN_PER_TOPIC:-10}"
STRATIFY_BY="${STRATIFY_BY:-category}"

if [[ ! -f "${INPUT}" ]]; then
  echo "error: input dataset JSONL not found: ${INPUT}" >&2
  echo "       build it first (chitanka_mental_health_dataset.sh) or set INPUT=path" >&2
  exit 1
fi

echo "[preset] input    = ${INPUT}"
echo "[preset] out-dir  = ${OUT_DIR}"

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation split-by-topic \
  --input "${INPUT}" \
  --out-dir "${OUT_DIR}" \
  --eval-ratio "${EVAL_RATIO}" \
  --seed "${SEED}" \
  --min-per-topic "${MIN_PER_TOPIC}" \
  --stratify-by "${STRATIFY_BY}"
