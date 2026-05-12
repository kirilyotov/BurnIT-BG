# data_transformation

Post-download cleanup and enrichment for the book corpus.

Three subcommands today:

| Subcommand | What it does |
| --- | --- |
| `translate` | Translate string fields in any dataset (JSONL / JSON / CSV / parquet / plain text) using the free Google Translate endpoint via `deep-translator`. |
| `filter-language` | For a given manifest source (default `project_gutenberg`), detect each book's actual language from its content, optionally delete non-matching ones from MinIO + DuckDB, and re-upload the refreshed `books.duckdb` / `books.parquet`. |
| `hf-dataset` | Download a HuggingFace dataset, translate selected fields, then push both the raw snapshot and the translated copy to MinIO **and** a HuggingFace bucket. |

Bash wrappers driving these live in
[`../run_scripts/data_transformation/`](../run_scripts/data_transformation/).

## Layout

```
data_transformation/
  __main__.py          # `python -m data_transformation ...` dispatches cli.main()
  cli.py               # argparse subcommands
  translate.py         # Translator (cache + chunking + backoff)
  filter_language.py   # evaluate / apply_removals / reupload_manifest
  hf_dataset.py        # download_repo / translate_dataset / upload_to_minio / upload_to_hf_bucket
  io_utils.py          # format-aware readers + RecordWriter (JSONL-in-.json fallback)
```

## CLI

```bash
# Translate
python -m data_transformation translate \
    --input tmp/data_scraping/_tmp/books.parquet \
    --output tmp/data_transformation/books.bg.jsonl \
    --fields title,summary \
    --source-lang en --target-lang bg \
    --delay 0.4

# Language filter (dry run by default)
python -m data_transformation filter-language \
    --source project_gutenberg --keep-lang en

python -m data_transformation filter-language \
    --source project_gutenberg --keep-lang en --apply

# Same shape, but reading from a local-disk backend instead of MinIO
python -m data_transformation filter-language \
    --backend local --source project_gutenberg --keep-lang en --apply
```

## `translate` — how it actually works

* **Provider** — `deep_translator.GoogleTranslator` (no API key, no auth).
  Free tier has soft rate limits; the script handles them with exponential
  backoff. Set `--delay 0.3` if you start seeing transient errors.
* **Field selection** — `--fields title,summary` picks specific keys.
  Dotted notation works for nested dicts (e.g. `meta.summary`). Lists of
  strings are translated element-wise. Omit `--fields` to translate every
  top-level string value.
* **Format handling** — extension chooses the parser/writer. The same
  flag works for `.jsonl`, `.ndjson`, `.json`, `.csv`, `.parquet`, and
  plain text (split into paragraphs).
* **Caching** — `--cache PATH` (default `tmp/data_transformation/translate_cache.json`).
  Repeated runs only call the API for new strings. The cache key includes
  source + target language, so en→bg and en→de coexist safely.
* **Long inputs** — strings over ~4500 chars are split on paragraph /
  sentence breaks before being sent; translated chunks are joined back
  with `\n\n` separators.

## `filter-language` — how it actually works

1. Reads the manifest rows for `--source` (defaults to
   `project_gutenberg`) from `tmp/data_scraping/books.duckdb`. Reads
   from a temp copy so a running `duckdb -ui` doesn't block the scan.
2. Streams each book file straight from MinIO into memory (no temp file
   on disk). EPUBs are unzipped in-memory and their HTML/XHTML stripped
   with BeautifulSoup; plain `.txt` and `.html` are sampled directly.
3. Runs `langdetect.detect()` on the first ~64 KB of text. Detection is
   seeded for determinism.
4. Prints a per-book report (`book_id`, detected language, format,
   title) and a summary.
5. Without `--apply` the command stops here (dry run).
6. With `--apply`:
   * Deletes each non-matching object from MinIO (`client.remove_object`).
   * Deletes the corresponding rows from `manifest` in DuckDB.
   * Re-exports `books.parquet` and uploads both `books.duckdb` and
     `books.parquet` (latest + per-date archive) to
     `raw/metadata/` so the published manifest matches reality.
   * Skip the re-upload with `--skip-metadata-upload` if you'd rather
     run `python -m data_scraping upload-metadata` separately.

The default `--keep-lang en` only makes sense for `project_gutenberg`;
chitanka books are all Bulgarian, so leave them alone (or pass
`--source chitanka --keep-lang bg` if you want a sanity check there too).

### Storage backends

`--backend` matches the corresponding flag in `data_scraping`:

* `minio` (default) — uses `MINIO_ENDPOINT`/`MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY`
  from `.env`. Object deletion goes through `Minio.remove_object`.
* `local` — for runs that wrote to disk. The `local_path` column in the
  DuckDB manifest is treated as an absolute filesystem path; the file is
  read in-memory and `unlink()`'d if the row is dropped.
* `huggingface` — read-only support (we can `load_bytes` via a temp
  download for detection), but `--apply` will fail when it tries to
  delete. Run on a different backend if you need actual removals.

## Detection failures

If `langdetect` cannot decide (very short text, mixed scripts, etc.) the
book is **kept** and the report shows `lang=?`. That's the safe default
— a false-positive deletion is more painful than a false-negative.

If a book file is missing from MinIO but still in DuckDB (a previous run
crashed mid-cleanup, say), the row is kept and the report notes the
download error.

## `hf-dataset` — end-to-end HuggingFace dataset pipeline

One command that:

1. Snapshots a HuggingFace dataset repo into
   `tmp/data_transformation/datasets/{slug}/raw/`.
2. Translates each data file (`.jsonl`, `.ndjson`, `.json`, `.csv`,
   `.parquet`) into `tmp/data_transformation/datasets/{slug}/{target_lang}/`.
   Non-data files (`LICENSE*`, `README*`, `COPYING*`, `NOTICE*`,
   `CHANGELOG*`, `.gitattributes`, etc.) are shipped under `raw/` but not
   translated.
3. Mirrors the same two folders to
   `s3://{bucket}/datasets/huggingface/{slug}/{raw,{target_lang}}/...`.
4. Mirrors the same two folders to the HF Bucket
   `{whoami}/{hf_bucket}/datasets/huggingface/{slug}/{raw,{target_lang}}/...`.

JSON files that actually contain newline-delimited records (common
HuggingFace convention) are auto-detected and translated row by row;
the output gets a `.jsonl` extension so downstream consumers don't need
the same heuristic.

```bash
# Full pipeline: download + translate + push to MinIO and HF
venv/bin/python -m data_transformation hf-dataset \
    --repo Amod/mental_health_counseling_conversations \
    --fields Context,Response \
    --source-lang en --target-lang bg \
    --delay 0.3

# Run individual steps (resumable):
venv/bin/python -m data_transformation hf-dataset \
    --repo Amod/mental_health_counseling_conversations \
    --skip-translate --skip-minio --skip-hf       # download only

venv/bin/python -m data_transformation hf-dataset \
    --repo Amod/mental_health_counseling_conversations \
    --fields Context,Response \
    --skip-download --skip-minio --skip-hf        # translate only

venv/bin/python -m data_transformation hf-dataset \
    --repo Amod/mental_health_counseling_conversations \
    --skip-download --skip-translate --skip-hf    # MinIO upload only
```

The bash wrapper at
[`../run_scripts/data_transformation/hf_dataset.sh`](../run_scripts/data_transformation/hf_dataset.sh)
exposes the same flags via env vars (`REPO`, `FIELDS`, `SOURCE_LANG`,
`TARGET_LANG`, `DELAY`, `BUCKET`, `MINIO_PREFIX`, `HF_BUCKET`,
`HF_PREFIX`, `SKIP_DOWNLOAD`, `SKIP_TRANSLATE`, `SKIP_MINIO`, `SKIP_HF`).

### HF bucket name handling

`--hf-bucket data` (bare name) is automatically rewritten to
`{whoami}/data` at runtime, so the same script works for any account that
has a bucket named `data`. Pass `--hf-bucket someorg/something` if you
need an organisation-owned bucket.

### Resume / cache

The translation cache lives at `tmp/data_transformation/translate_cache.json`
by default and is keyed on `(source_lang, target_lang, text)`. Re-running
the same dataset only calls the API for strings it hasn't seen — useful
when you need to re-translate after fixing one bad row.

## Requirements

Install everything used by this package with:

```bash
venv/bin/pip install -r requirements_data_transformation.txt
```

Bundles: `deep-translator`, `langdetect`, `ebooklib`, `beautifulsoup4`,
`duckdb`, `python-dotenv`, `minio`.
