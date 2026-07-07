#!/usr/bin/env bash
# Stage (and, ONLY when PUSH=1, upload) the mental-health dataset(s) to a
# public HuggingFace dataset repo.
#
# SAFETY: defaults to --stage-only. It will NOT upload anything unless you
# explicitly run with PUSH=1. The `style` split contains verbatim copyrighted
# book excerpts — review before making it public.
#
# Knobs (set what you want to include):
#   REPO=username/burnit-bg-mental-health   (required)
#   STYLE=path/to/style/dataset.jsonl
#   QA=path/to/qa/dataset.jsonl
#   SPLITS_DIR=path/to/splits
#   PRIVATE=1     (create repo private instead of public)
#   PUSH=1        (actually upload; otherwise only stages locally)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

REPO="${REPO:-}"
if [[ -z "${REPO}" ]]; then
  echo "error: set REPO=username/dataset-name" >&2
  exit 1
fi

STAGING="${STAGING:-${REPO_ROOT}/tmp/data_transformation/hf_dataset_staging}"
LICENSE="${LICENSE:-cc-by-nc-4.0}"

ARGS=( publish-dataset --repo "${REPO}" --staging-dir "${STAGING}" --license "${LICENSE}" )
[[ -n "${STYLE:-}" ]]      && ARGS+=( --style "${STYLE}" )
[[ -n "${QA:-}" ]]         && ARGS+=( --qa "${QA}" )
[[ -n "${SPLITS_DIR:-}" ]] && ARGS+=( --splits-dir "${SPLITS_DIR}" )
[[ "${PRIVATE:-0}" == "1" ]] && ARGS+=( --private )

# Default to stage-only unless PUSH=1 is explicitly set.
if [[ "${PUSH:-0}" != "1" ]]; then
  ARGS+=( --stage-only )
  echo "[safety] staging only — re-run with PUSH=1 to actually upload."
fi

cd "${REPO_ROOT}"
exec "${PY}" -m data_transformation "${ARGS[@]}"
