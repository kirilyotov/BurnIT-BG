# data_scraping

Book ingestion pipeline. Walks OPDS catalogs, downloads each book in a
chosen format, and uploads the file plus a normalised metadata record to a
storage backend (local / MinIO / Hugging Face). All metadata also lands in
a local DuckDB store that can be snapshotted to parquet and pushed to the
same backend.

Bash wrappers that drive this CLI live in
[`../run_scripts/data_scraping/`](../run_scripts/data_scraping/).

## Layout

```
data_scraping/
  __main__.py         # `python -m data_scraping ...` dispatches to cli.main()
  cli.py              # argparse subcommands: chitanka / gutenberg / upload-metadata
  download_books.py   # source-agnostic PipelineConfig + run_pipeline()
  duckdb_store.py     # DuckDB-backed manifest store (upsert)
  storage_backend.py  # façade over data_platform.storage (full remote keys)
  http.py             # request_with_retries + stream_to_file (429/5xx backoff)
  downloader_config.py
  sources/
    __init__.py       # SOURCES registry
    base.py           # BookEntry, AcquisitionLink, BookSource, pick_acquisition
    opds.py           # low-level OPDS feed walker (pagination + retries, urljoin'd)
    chitanka.py       # ChitankaSource — 5 Bulgarian categories, MIME → format map
    gutenberg.py      # GutenbergSource — 4 topical queries, per-book opds dereference
```

## CLI

```bash
python -m data_scraping chitanka \
    --categories psychology applied-psychology self-improvement \
    --per-category 5 --limit 25 --delay 3.0 \
    --formats epub,fb2,txt \
    --backend minio --bucket data --remote-prefix raw

python -m data_scraping gutenberg \
    --categories psychology mental-health \
    --per-category 5 --limit 20 --delay 2.0 \
    --formats epub,txt

python -m data_scraping upload-metadata
```

Use `--help` on any subcommand for the full flag list.

## How the flags actually behave

| Flag | Meaning |
| --- | --- |
| `--categories` | Space-separated subset of the source's known category/topic names. Omit to use all of them. |
| `--per-category` | Cap on how many books to take from each category before moving on. |
| `--limit` | Global cap across **all** categories combined; checked after every successful download. Whichever cap is reached first wins. |
| `--delay` | Seconds slept between HTTP requests. Polite-scraping baseline; 429/5xx also triggers exponential backoff on top. |
| `--formats` | Priority list. The pipeline downloads **one** file per book: the first format from this list that the book actually offers. If none match, the book is skipped. |
| `--backend` | `minio` (default), `local`, or `huggingface`. |
| `--bucket` | Bucket name. Defaults to `MINIO_BUCKET` env (`data` if unset). |
| `--remote-prefix` | Object-key prefix. Defaults to `raw`. |
| `--date` | Date partition (folder name). Defaults to today (`YYYY-MM-DD`). |
| `--duckdb` | Local DuckDB file. Code default: `Local/books.duckdb`. Override or move to `run_scripts/data_scraping/books.duckdb` to keep state next to the wrappers. |
| `--tmp-dir` | Scratch directory for streamed downloads + manifest staging. Code default: `Local/_tmp`. |
| `--secure` | Force MinIO TLS on/off. Defaults to env `MINIO_FORCE_SECURE` (false). |

### Format selection example

`--formats epub,fb2,txt,pdf`:

1. If the book offers EPUB → download EPUB, done.
2. Else fall through to FB2, then TXT, then PDF.
3. If none match → the book is skipped (logged as `[skip]`).

Chitanka serves epub / fb2 / txt-zip (no PDFs over OPDS); listing `pdf`
in the priority list is harmless but never matches. Gutenberg serves EPUB
for almost every book and PDF only on a minority.

## Bucket layout produced

```
{remote_prefix}/{source}/{date}/{category}/{format}/{book_id}{ext}
{remote_prefix}/{source}/{date}/metadata/manifest.jsonl
```

`upload-metadata` writes the union DuckDB store + parquet snapshot to:

```
{remote_prefix}/metadata/books.duckdb
{remote_prefix}/metadata/books.parquet
{remote_prefix}/metadata/books-{date}.{duckdb,parquet}
```

The parquet is a portable snapshot of the `manifest` table. Schema and
sample queries are documented under *Inspecting results* below.

## Sources

### Chitanka — `python -m data_scraping chitanka`

* OPDS catalog: `https://chitanka.info/books.opds`
* Categories shipped (see [`sources/chitanka.py`](sources/chitanka.py)):
  `psychology`, `applied-psychology`, `self-improvement`,
  `health-and-alt-medicine`, `self-help-manuals`.
* MIME map: epub / fb2 / txt-zip (and `application/pdf` declared but the
  feed never offers PDFs in practice).
* Behind Cloudflare; use `--delay >= 3.0` to avoid 429.

### Project Gutenberg — `python -m data_scraping gutenberg`

* Catalog: `https://www.gutenberg.org/ebooks.opds/`
* Each "category" is actually a relevance-sorted search query
  (see [`sources/gutenberg.py`](sources/gutenberg.py)):
  `psychology`, `self-improvement`, `mental-health`, `philosophy-of-mind`.
* Search results are navigation entries; the source dereferences each
  per-book `/ebooks/{id}.opds` page to find acquisition links.
* MIME map: epub / txt / html / mobi / pdf.

### Adding a new source

1. Create `data_scraping/sources/<name>.py` exposing a class with
   `name`, `category_choices()`, and `iter_books(category, limit, delay)`
   yielding `BookEntry` objects.
2. Register it in [`sources/__init__.py`](sources/__init__.py) under `SOURCES`.
3. Add a subparser block in [`cli.py`](cli.py) mirroring the chitanka /
   gutenberg ones.

## Environment

The CLI auto-loads `.env` from the repo root via `python-dotenv`.
MinIO credentials are read from `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
`MINIO_SECRET_KEY`, and optionally `MINIO_BUCKET`. The repo's MinIO is on
plain HTTP at `:9000`, so the CLI defaults `secure=False` unless
`MINIO_FORCE_SECURE=true` is set.

## Resilience

* **Pagination URLs** — `OPDSSource` resolves the `rel=next` link with
  `urljoin`, so chitanka's relative `/books/category/.../2` style is
  followed correctly across pages.
* **Per-category isolation** — if `source.iter_books(category)` raises
  mid-walk (bad URL, parse error, network blip), the pipeline logs
  `[error] {source} :: {cat} aborted (...); moving on` and continues
  with the next category instead of killing the whole run.
* **HTTP retries** — `data_scraping/http.py` retries 429 / 500 / 502 /
  503 / 504 with exponential backoff, honouring `Retry-After`.
* **DuckDB lock** — if another process (often `duckdb -ui`) holds an
  exclusive lock on the manifest store, the pipeline logs a warning and
  falls back to JSONL-only for that run. `upload-metadata` sidesteps the
  lock by reading from a temp copy of the file.

## Programmatic use

```python
from pathlib import Path

from data_scraping.download_books import PipelineConfig, run_pipeline
from data_scraping.sources import ChitankaSource
from data_scraping.storage_backend import StorageBackend

storage = StorageBackend.minio(
    endpoint="minio.example:9000",
    access_key="…", secret_key="…", bucket="data", secure=False,
)
cfg = PipelineConfig(
    storage=storage,
    bucket="data",
    remote_prefix="raw",
    formats=("epub", "fb2"),
    delay=3.0,
    per_category=5,
    limit=25,
    duckdb_path=Path("run_scripts/data_scraping/books.duckdb"),
)
run_pipeline(ChitankaSource(), categories=["psychology"], cfg=cfg)
```

## Inspecting results

### DuckDB manifest table

```bash
venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("run_scripts/data_scraping/books.duckdb", read_only=True)
print(con.execute("SELECT source, count(*) FROM manifest GROUP BY 1").fetchall())
print(con.execute("SELECT source, title FROM manifest LIMIT 5").fetchall())
PY
```

### Parquet snapshot (one row per downloaded file)

| Column | Notes |
| --- | --- |
| `source` | `chitanka` / `project_gutenberg` |
| `entry_id`, `book_id` | catalog IDs (entry_id is the primary key with `source`) |
| `title` | book title |
| `authors`, `categories` | **JSON strings** (lists) — use `json_extract(...)` or `json.loads` |
| `language`, `summary` | metadata fields (summary often empty for both sources) |
| `catalog_url`, `book_page_url`, `download_url` | provenance |
| `download_format`, `download_mime_type` | what we actually downloaded |
| `retrieved_at` | ISO-8601 timestamp |
| `local_path` | `s3://{bucket}/{key}` URI for the uploaded file |
| `sha256`, `file_size_bytes` | integrity + size |

```bash
# Local parquet (written by upload-metadata into the tmp dir)
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

# Direct from MinIO via DuckDB httpfs
venv/bin/python - <<'PY'
import duckdb, os
from pathlib import Path
from dotenv import load_dotenv; load_dotenv(Path(".env"))
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"SET s3_endpoint='{os.environ['MINIO_ENDPOINT']}'")
con.execute("SET s3_use_ssl=false")
con.execute(f"SET s3_access_key_id='{os.environ['MINIO_ACCESS_KEY']}'")
con.execute(f"SET s3_secret_access_key='{os.environ['MINIO_SECRET_KEY']}'")
print(con.execute(
    "SELECT source, count(*) FROM read_parquet('s3://data/raw/metadata/books.parquet') GROUP BY 1"
).fetchall())
PY
```

### MinIO bucket listing

```bash
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
```
