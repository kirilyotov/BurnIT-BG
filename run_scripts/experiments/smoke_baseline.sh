#!/usr/bin/env bash
# Smoke-test the baseline finetuning pipeline.
#
# Runs every helper the baseline notebook touches (load_model_unsloth,
# apply_lora, SFTConfig, SFTTrainer.train, compute_perplexity, MLflow) on a
# 135M-param tiny model + 16-record synthetic dataset. Should finish in
# <60s on CPU. Exit code = number of failed stages.
#
# Knobs:
#   --gpu                     Use the 4-bit QLoRA path (needs CUDA + bnb)
#   SMOKE_MODEL=...           Override the tiny model
#   SMOKE_MAX_STEPS=2         Number of training steps
#   SMOKE_MLFLOW_DIR=tmp/...  Where to put the local file:// MLflow store
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="python"
fi

cd "${REPO_ROOT}"
exec "${PY}" -m experiments.smoke.baseline_smoke "$@"
