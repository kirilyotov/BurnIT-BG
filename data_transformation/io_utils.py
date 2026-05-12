"""Lightweight format-aware readers / writers used by the translator.

Format is picked from the file extension:

* ``.jsonl`` / ``.ndjson`` — newline-delimited JSON, streamed row by row.
* ``.json``                — single JSON document (object or array).
* ``.csv``                 — DictReader / DictWriter.
* ``.parquet``             — read/written through DuckDB.
* anything else            — treated as a plain UTF-8 text file split into paragraphs.

Only used internally by :mod:`data_transformation.translate`. If you need a
format we don't support, convert the data to JSONL first.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional


def detect_format(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suf == ".json":
        return "json"
    if suf == ".csv":
        return "csv"
    if suf == ".parquet":
        return "parquet"
    return "text"


def read_records(path: Path) -> Iterator[dict]:
    """Yield dicts from any structured format. Plain text → ``{"text": "..."}``."""
    fmt = detect_format(path)
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif fmt == "json":
        # Many HuggingFace datasets ship JSON-Lines content with a `.json`
        # extension. Try a strict parse first; on failure, fall back to
        # line-by-line JSON.
        text = Path(path).read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                for item in payload:
                    yield item if isinstance(item, dict) else {"value": item}
            elif isinstance(payload, dict):
                yield payload
            else:
                yield {"value": payload}
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif fmt == "csv":
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                yield dict(row)
    elif fmt == "parquet":
        import duckdb
        con = duckdb.connect(":memory:")
        try:
            cols = [c[0] for c in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()]
            for row in con.execute(f"SELECT * FROM read_parquet('{path}')").fetchall():
                yield dict(zip(cols, row))
        finally:
            con.close()
    else:  # text — yield non-empty paragraphs as {'text': ...}
        with open(path, "r", encoding="utf-8") as fh:
            buf: List[str] = []
            for line in fh:
                if line.strip():
                    buf.append(line.rstrip("\n"))
                elif buf:
                    yield {"text": "\n".join(buf)}
                    buf = []
            if buf:
                yield {"text": "\n".join(buf)}


class RecordWriter:
    """Streaming writer for jsonl / json / csv / parquet / text outputs."""

    def __init__(self, path: Path, fieldnames: Optional[List[str]] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fmt = detect_format(self.path)
        self._fieldnames = fieldnames
        self._buffer: List[dict] = []
        self._fh = None
        self._csv_writer = None

    def __enter__(self) -> "RecordWriter":
        if self.fmt == "jsonl":
            self._fh = open(self.path, "w", encoding="utf-8")
        elif self.fmt == "text":
            self._fh = open(self.path, "w", encoding="utf-8")
        elif self.fmt == "csv":
            self._fh = open(self.path, "w", encoding="utf-8", newline="")
        # json / parquet are written all at once on close.
        return self

    def write(self, record: dict) -> None:
        if self.fmt == "jsonl":
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        elif self.fmt == "text":
            text = record.get("text") if "text" in record else next(iter(record.values()), "")
            self._fh.write(str(text) + "\n\n")
        elif self.fmt == "csv":
            if self._csv_writer is None:
                keys = self._fieldnames or list(record.keys())
                self._csv_writer = csv.DictWriter(self._fh, fieldnames=keys)
                self._csv_writer.writeheader()
                self._fieldnames = keys
            self._csv_writer.writerow({k: record.get(k, "") for k in self._fieldnames})
        else:  # json / parquet — buffer
            self._buffer.append(record)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
        if self.fmt == "json":
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._buffer, fh, ensure_ascii=False, indent=2)
        elif self.fmt == "parquet":
            import duckdb
            con = duckdb.connect(":memory:")
            try:
                # Materialise records via DuckDB by registering a pandas-like
                # bridge through json. Cheap and robust for moderate sizes.
                tmp_path = self.path.with_suffix(self.path.suffix + ".jsonl")
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    for rec in self._buffer:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                con.execute(
                    "COPY (SELECT * FROM read_json_auto(?)) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                    [str(tmp_path), str(self.path)],
                )
                tmp_path.unlink(missing_ok=True)
            finally:
                con.close()


def keys_in_path(record: dict, path: str) -> List[str]:
    """Return the field-path components for a dotted reference, e.g. ``a.b.c``."""
    return [p for p in path.split(".") if p]


def get_nested(record: dict, path: str) -> Any:
    parts = keys_in_path(record, path)
    cur: Any = record
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def set_nested(record: dict, path: str, value: Any) -> None:
    parts = keys_in_path(record, path)
    if not parts:
        return
    cur: Any = record
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
