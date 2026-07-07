#!/usr/bin/env bash
# Pull mental-health passages from the books already in MinIO.
#
# Default behaviour (no env vars set):
#   * Extracts from EVERY chitanka book in tmp/data_scraping/books.duckdb,
#     regardless of when it was downloaded.
#   * Tags today's date on the output: passages-{today}.jsonl and
#     s3://{bucket}/extracted/chitanka/{today}/passages.jsonl.
#   * Uses all 23 mental-health topics, formats epub/fb2/txt,
#     keeps up to 50 passages per book.
#
# Naming knob:
#   PASSAGES_NAME=my-run-v2   # used for the JSONL filename and the MinIO prefix
#                             # → tmp/data_scraping/extracted/passages-my-run-v2.jsonl
#                             # → s3://{bucket}/extracted/chitanka/my-run-v2/passages.jsonl
#   (Defaults to EXTRACT_DATE which defaults to today.)
#
# Scope knobs:
#   BOOKS_DATE=2026-05-14  TOPICS=anxiety,depression  FORMATS=epub
#   BOOK_IDS=10007,10100   PER_BOOK=20  LIMIT=200
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REPO_ROOT}/venv/bin/python"

EXTRACT_DATE="${EXTRACT_DATE:-$(date +%F)}"
# PASSAGES_NAME is the DATE SEGMENT in the MinIO path (and the suffix on the
# local filename). Defaults to EXTRACT_DATE so legacy usage is unchanged.
#   path: {prefix}/{source}/{PASSAGES_NAME}/{FILENAME}.jsonl
PASSAGES_NAME="${PASSAGES_NAME:-${EXTRACT_DATE}}"
# FILENAME is the FINAL FILE-NAME segment (no extension, no date).
#   local:  tmp/data_scraping/extracted/{FILENAME}-{PASSAGES_NAME}.jsonl
#   minio:  {prefix}/{source}/{PASSAGES_NAME}/{FILENAME}.jsonl
FILENAME="${FILENAME:-passages}"
BOOKS_DATE="${BOOKS_DATE:-}"           # empty == no manifest date filter

SOURCE="${SOURCE:-chitanka}"
TOPICS="${TOPICS:-}"                   # empty == all topics
FORMATS="${FORMATS:-epub,fb2,txt}"
BOOK_IDS="${BOOK_IDS:-}"
MIN_CHARS="${MIN_CHARS:-80}"
MAX_CHARS="${MAX_CHARS:-1200}"
PER_BOOK="${PER_BOOK:-}"             # cap per book — set "" for unlimited
LIMIT="${LIMIT:-}"                     # overall cap, "" = unlimited

BACKEND="${BACKEND:-minio}"
BUCKET="${BUCKET:-}"
REMOTE_PREFIX="${REMOTE_PREFIX:-extracted}"
DUCKDB="${DUCKDB:-${REPO_ROOT}/tmp/data_scraping/books.duckdb}"
TMP_DIR="${TMP_DIR:-${REPO_ROOT}/tmp/data_scraping/extracted}"

# The CLI uses --extract-date for both the filename suffix and the MinIO
# prefix segment; pass PASSAGES_NAME so re-running with the same EXTRACT_DATE
# but a different PASSAGES_NAME writes a separate file.
ARGS=( extract-passages
       --source "${SOURCE}"
       --extract-date "${PASSAGES_NAME}"
       --filename "${FILENAME}"
       --formats "${FORMATS}"
       --min-chars "${MIN_CHARS}"
       --max-chars "${MAX_CHARS}"
       --remote-prefix "${REMOTE_PREFIX}"
       --duckdb "${DUCKDB}"
       --tmp-dir "${TMP_DIR}"
       --backend "${BACKEND}" )

if [[ -n "${BOOKS_DATE}" ]]; then ARGS+=( --books-date "${BOOKS_DATE}" ); fi
if [[ -n "${TOPICS}" ]];     then ARGS+=( --topics    "${TOPICS}" ); fi
if [[ -n "${BOOK_IDS}" ]];   then ARGS+=( --book-ids  "${BOOK_IDS}" ); fi
if [[ -n "${PER_BOOK}" ]];   then ARGS+=( --per-book  "${PER_BOOK}" ); fi
if [[ -n "${LIMIT}" ]];      then ARGS+=( --limit     "${LIMIT}" ); fi
if [[ -n "${BUCKET}" ]];     then ARGS+=( --bucket    "${BUCKET}" ); fi

echo "[preset] date segment = ${PASSAGES_NAME}"
echo "[preset] filename     = ${FILENAME}"
echo "[preset] local output = ${TMP_DIR}/${FILENAME}-${PASSAGES_NAME}.jsonl"
echo "[preset] minio path   = ${REMOTE_PREFIX}/${SOURCE}/${PASSAGES_NAME}/${FILENAME}.jsonl"

cd "${REPO_ROOT}"
exec "${PY}" -m data_scraping "${ARGS[@]}"
