# run_scripts/data_scraping — bash wrappers for the ingestion pipeline

All Python lives in [`../../data_scraping/`](../../data_scraping/). This
folder only contains shell scripts that drive its CLI, plus runtime
working files (`books.duckdb`, `_tmp/`) which are gitignored.

## Bucket layout produced

```
data/
  raw/
    chitanka/
      YYYY-MM-DD/
        psychology/{epub,fb2,txt}/*
        applied-psychology/...
        self-improvement/...
        health-and-alt-medicine/...
        self-help-manuals/...
        metadata/manifest.jsonl
    project_gutenberg/
      YYYY-MM-DD/
        psychology/{epub,txt}/*
        self-improvement/...
        mental-health/...
        philosophy-of-mind/...
        metadata/manifest.jsonl
    metadata/
      books.duckdb               # union manifest (all runs, all sources)
      books.parquet              # portable snapshot of the manifest table
      books-YYYY-MM-DD.duckdb    # per-run archive
      books-YYYY-MM-DD.parquet
```

## Prerequisites

* `venv/bin/python` (the repo's virtualenv).
* The repo `.env` with `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
  `MINIO_SECRET_KEY`, optionally `MINIO_BUCKET` (defaults to `data`).
* The MinIO endpoint `:9000` serves plain HTTP, so the CLI defaults to
  `secure=False`. Set `MINIO_FORCE_SECURE=true` (env) or `--secure true`
  (flag) if MinIO is fronted by TLS.

## Scripts

| File | Wraps |
| --- | --- |
| `chitanka.sh` | `python -m data_scraping chitanka …` |
| `gutenberg.sh` | `python -m data_scraping gutenberg …` |
| `upload_metadata.sh` | `python -m data_scraping upload-metadata …` |
| `run_all.sh` | Runs the three above in order. |

Each script honours environment overrides for the four pipeline knobs:

```bash
PER_CATEGORY=10 LIMIT=50 DELAY=4.0 FORMATS=epub \
    ./run_scripts/data_scraping/chitanka.sh

CATEGORIES="psychology mental-health" PER_CATEGORY=8 \
    ./run_scripts/data_scraping/gutenberg.sh
```

`CATEGORIES` is a **space-separated** list — it expands as multiple
`--categories` args. Skip it to use the source's defaults.

### Knob reference

| Variable | Default (chitanka / gutenberg) | What it does |
| --- | --- | --- |
| `PER_CATEGORY` | `100 / 100` | Max books per category before moving on. |
| `LIMIT` | `500 / 400` | Global cap across **all** categories combined. |
| `DELAY` | `3.0 / 2.0` | Seconds between HTTP requests (plus backoff on 429). |
| `FORMATS` | `epub,fb2,txt,pdf / epub,txt,pdf` | Priority list. Downloads **one** file per book: the first format the book offers. Skips the book if none match. |
| `CATEGORIES` | source default set | Space-separated subset of category names. |

To override destination (bucket, key prefix, date partition, alternate
backend, etc.) pass flags directly to the CLI — see *Direct CLI use* below.

## Full pipeline

```bash
./run_scripts/data_scraping/run_all.sh
```

## Direct CLI use (no shell wrapper)

```bash
venv/bin/python -m data_scraping chitanka \
    --categories psychology applied-psychology \
    --per-category 5 --limit 25 --delay 3.0 \
    --formats epub,fb2,txt

venv/bin/python -m data_scraping gutenberg \
    --categories psychology mental-health \
    --per-category 5 --limit 20 --delay 2.0 \
    --formats epub,txt

venv/bin/python -m data_scraping upload-metadata
```

`--help` on any subcommand lists every flag (date override, remote prefix,
alternate backends, MinIO credentials override, custom DuckDB path, etc.).

## Inspecting results

```bash
# DuckDB row counts (read-only, so a running `duckdb -ui` doesn't block)
venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("run_scripts/data_scraping/books.duckdb", read_only=True)
print(con.execute("SELECT source, count(*) FROM manifest GROUP BY 1").fetchall())
print(con.execute("SELECT source, title FROM manifest LIMIT 5").fetchall())
PY

# Bucket listing
venv/bin/python - <<'PY'
import os
from pathlib import Path
from dotenv import load_dotenv; load_dotenv(Path(".env"))
from data_platform.storage.minio import MinioStorage
s = MinioStorage(
    endpoint=os.environ["MINIO_ENDPOINT"],
    access_key=os.environ["MINIO_ACCESS_KEY"],
    secret_key=os.environ["MINIO_SECRET_KEY"],
    bucket=os.getenv("MINIO_BUCKET", "data"),
    secure=False,
)
for k in sorted(s.list_objects(prefix="raw/")):
    print(k)
PY

# Parquet snapshot (also usable with pandas / polars / DuckDB httpfs)
venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect(":memory:")
df = con.execute(
    "SELECT source, download_format, count(*) AS n "
    "FROM read_parquet('run_scripts/data_scraping/_tmp/books.parquet') "
    "GROUP BY 1,2 ORDER BY 1,2"
).fetchall()
for row in df: print(row)
PY
```

## Heads-up: stale paths

A few things still reference the old `Local/` location. If you hit them,
update accordingly:

* `data_scraping/cli.py` defaults `DEFAULT_DUCKDB = REPO_ROOT / "Local" /
  "books.duckdb"` and `DEFAULT_TMP = REPO_ROOT / "Local" / "_tmp"`. If you
  want the CLI to land its DuckDB and scratch in this directory by
  default, change those constants — or pass `--duckdb
  run_scripts/data_scraping/books.duckdb --tmp-dir
  run_scripts/data_scraping/_tmp` on every invocation.
* The bash scripts compute `REPO_ROOT="$(... /.. && pwd)"`. From
  `run_scripts/data_scraping/`, that resolves to `run_scripts/`, not the
  repo root, so `${REPO_ROOT}/venv/bin/python` is wrong. Change `/..` to
  `/../..` (or `cd "$(dirname "${BASH_SOURCE[0]}")/../.."`) so the
  wrappers find the repo `venv` and `.env`.
