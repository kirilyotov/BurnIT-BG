#!/usr/bin/env bash
# Download chitanka.info books in psychology / self-help categories into default storage.
# Adjust the env vars at the top — flags below them stay defaulted.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

PER_CATEGORY="${PER_CATEGORY:-100}"
LIMIT="${LIMIT:-500}"
DELAY="${DELAY:-3.0}"
FORMATS="${FORMATS:-epub,fb2,txt, pdf}"
CATEGORIES="${CATEGORIES:-psychology applied-psychology self-improvement health-and-alt-medicine self-help-manuals}"

cd "${REPO_ROOT}"
exec "${PY}" -m data_scraping chitanka \
    --per-category "${PER_CATEGORY}" \
    --limit "${LIMIT}" \
    --delay "${DELAY}" \
    --formats "${FORMATS}" \
    --categories ${CATEGORIES}
