"""Helpers to log a Jupyter notebook WITH its executed cell outputs to MLflow.

mlflow.log_artifact on a raw .ipynb captures whatever is on disk; in Colab
that is the pre-run version without outputs. This module snapshots the LIVE
in-memory notebook (with outputs) and logs both the .ipynb and an HTML
rendering (when nbconvert is available).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _fetch_colab_ipynb() -> dict | None:
    """Return the live in-memory Colab notebook JSON, or None outside Colab."""
    try:
        from google.colab import _message
    except Exception:
        return None
    try:
        resp = _message.blocking_request("get_ipynb", timeout_sec=30)
        return resp.get("ipynb") if isinstance(resp, dict) else resp
    except Exception as exc:
        log.warning("Colab get_ipynb failed: %s", exc)
        return None


def _fetch_jupyter_rest_ipynb(notebook_path: str | None) -> dict | None:
    """Try the local Jupyter Server REST API. Returns notebook JSON or None."""
    if not notebook_path:
        return None
    try:
        import requests
    except Exception:
        return None
    token = os.getenv("JUPYTER_TOKEN", "")
    port = os.getenv("JUPYTER_PORT", "8888")
    try:
        url = f"http://localhost:{port}/api/contents/{notebook_path.lstrip('/')}"
        r = requests.get(url, params={"token": token} if token else None, timeout=5)
        if r.status_code != 200:
            return None
        return r.json().get("content")
    except Exception:
        return None


def _has_outputs(nb_json: dict) -> bool:
    cells = nb_json.get("cells", []) if isinstance(nb_json, dict) else []
    return any(c.get("outputs") for c in cells if c.get("cell_type") == "code")


def _render_html(nb_path: Path) -> Path | None:
    try:
        out_dir = nb_path.parent
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "html", "--embed-images",
             "--output", nb_path.with_suffix(".html").name,
             "--output-dir", str(out_dir), str(nb_path)],
            check=True, capture_output=True, timeout=180,
        )
        html = nb_path.with_suffix(".html")
        return html if html.exists() else None
    except Exception as exc:
        log.warning("nbconvert html failed: %s", exc)
        return None


def log_executed_notebook(
    tracking: Any,
    *,
    notebook_path: str | None = None,
    artifact_subdir: str = "notebook",
    also_html: bool = True,
    require_outputs: bool = True,
) -> dict[str, str | None]:
    """Snapshot the running notebook (with outputs) and log it to MLflow.

    Resolution order: Colab _message API -> Jupyter REST -> disk fallback.
    Validates outputs via nbformat (raises when require_outputs and none found)
    so we never silently upload an empty notebook.
    Logs <artifact_subdir>/<name>.ipynb and (when also_html) the HTML render.
    Returns {"ipynb": local_path, "html": local_path|None, "source": which-path-used}.
    """
    import nbformat

    source = "colab-live"
    nb_json = _fetch_colab_ipynb()
    if nb_json is None:
        source = "jupyter-rest"
        nb_json = _fetch_jupyter_rest_ipynb(notebook_path)
    if nb_json is None and notebook_path and Path(notebook_path).exists():
        source = "disk"
        nb_json = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    if nb_json is None:
        raise RuntimeError(
            "Could not locate the notebook (pass notebook_path=... or "
            "run inside Colab / a local Jupyter server)."
        )

    if require_outputs and not _has_outputs(nb_json):
        raise RuntimeError(
            "Notebook has no cell outputs - refusing to log an empty notebook. "
            "Re-run the cells first, or call with require_outputs=False."
        )

    name = Path(notebook_path or "notebook.ipynb").stem
    tmp = Path(tempfile.mkdtemp(prefix="exec_nb_"))
    nb_path = tmp / f"{name}.ipynb"
    nb = nbformat.from_dict(nb_json) if isinstance(nb_json, dict) else nbformat.reads(json.dumps(nb_json), as_version=4)
    nbformat.write(nb, str(nb_path))
    tracking.save_data(nb_path, artifact_path=artifact_subdir)

    html_path: Path | None = None
    if also_html:
        html_path = _render_html(nb_path)
        if html_path is not None:
            tracking.save_data(html_path, artifact_path=artifact_subdir)

    log.info("logged executed notebook (source=%s) -> %s", source, nb_path.name)
    return {"ipynb": str(nb_path), "html": (str(html_path) if html_path else None), "source": source}
