#!/usr/bin/env bash
# Download Project Gutenberg books in psychology / self-help topics into default storage MinIO.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

PER_CATEGORY="${PER_CATEGORY:-100}"
LIMIT="${LIMIT:-400}"
DELAY="${DELAY:-2.0}"
FORMATS="${FORMATS:-epub,txt,pdf}"
CATEGORIES="${CATEGORIES:-psychology self-improvement mental-health philosophy-of-mind}"

cd "${REPO_ROOT}"
exec "${PY}" -m data_scraping gutenberg \
    --per-category "${PER_CATEGORY}" \
    --limit "${LIMIT}" \
    --delay "${DELAY}" \
    --formats "${FORMATS}" \
    --categories ${CATEGORIES}
