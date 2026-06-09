"""Helpers to log a Jupyter notebook WITH its executed cell outputs to MLflow.

mlflow.log_artifact on a raw .ipynb captures whatever is on disk; in Colab
that is the pre-run version without outputs. This module snapshots the LIVE
in-memory notebook (with outputs) and logs both the .ipynb and an HTML
rendering (when nbconvert is available).
"""


import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _fetch_colab_ipynb(timeout_sec: float = 60.0) -> tuple[dict | None, str]:
    """Return ``(notebook_json_or_None, status_string)`` from the Colab message API.

    The Colab ``_message.blocking_request("get_ipynb")`` returns the LIVE
    in-memory notebook (including outputs of every cell that has finished
    rendering). Long timeout helps in busy kernels.
    """
    try:
        from google.colab import _message  # type: ignore[import-not-found]  # pylint: disable=import-error,no-name-in-module
    except Exception as exc:  # noqa: BLE001
        return None, f"not in Colab ({type(exc).__name__})"
    try:
        resp = _message.blocking_request("get_ipynb", timeout_sec=timeout_sec)
        nb = resp.get("ipynb") if isinstance(resp, dict) else resp
        if nb is None:
            return None, f"empty response: {type(resp).__name__}"
        return nb, "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _guess_notebook_path(extra_dirs: list[str] | None = None) -> str | None:
    """Best-effort search for the running notebook's path on disk.

    Resolution order:
      1. Colab default locations (``/content``, ``/content/drive/MyDrive/Colab Notebooks``).
      2. Caller-supplied ``extra_dirs`` (e.g. the running cwd, the repo root).
      3. As a last resort, walk the current working directory.

    Returns the most-recently-modified ``.ipynb`` (excluding ``.ipynb_checkpoints``)
    — that's almost always the notebook the user is editing.
    """
    candidates: list[Path] = []
    roots: list[Path] = [
        Path("/content"),
        Path("/content/drive/MyDrive/Colab Notebooks"),
    ]
    if extra_dirs:
        roots.extend(Path(d) for d in extra_dirs)
    roots.append(Path.cwd())
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*.ipynb"):
                if ".ipynb_checkpoints" in p.parts:
                    continue
                # Skip cached/stale artifact-dir copies under tmp/ or .git/.
                if any(part in {"tmp", ".git", "node_modules", "__pycache__"} for part in p.parts):
                    continue
                candidates.append(p)
        except Exception:  # noqa: BLE001
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return str(candidates[0])


# Back-compat alias for older callers.
_guess_colab_notebook_path = _guess_notebook_path


def _detect_vscode_or_jupyter_path() -> str | None:
    """Look for editor-specific globals that hold the running notebook's path.

    * VSCode notebook kernels inject ``__vsc_ipynb_file__`` into the user
      namespace as the absolute path.
    * JupyterLab >=3 sets ``__session__`` (a notebook URL / path) in some
      configurations.

    We can only see globals INSIDE the running kernel — peek via the
    IPython get_ipython().user_ns dict.
    """
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return None
    ip = get_ipython()
    if ip is None:
        return None
    ns = getattr(ip, "user_ns", {}) or {}
    for key in ("__vsc_ipynb_file__", "__session__"):
        val = ns.get(key)
        if isinstance(val, str) and val.endswith(".ipynb") and Path(val).exists():
            return val
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


def _render_html(nb_path: Path) -> tuple[Path | None, str]:
    """Render a notebook to HTML. Returns (path_or_None, status string).

    Primary path: in-process ``nbconvert.HTMLExporter`` — no subprocess, no
    PATH dependency, works on Colab where the ``jupyter`` CLI is usually
    not in the runtime's PATH even though the Python ``nbconvert`` library
    is installed.

    Fallback: subprocess ``jupyter nbconvert`` — kept for systems where the
    Python API errors but the CLI works (rare).
    """
    html_out = nb_path.with_suffix(".html")

    # ── 1) In-process nbconvert.HTMLExporter ──────────────────────────────
    try:
        import nbformat
        from nbconvert import HTMLExporter

        nb = nbformat.read(str(nb_path), as_version=4)
        exporter = HTMLExporter()
        exporter.exclude_input_prompt = False
        exporter.exclude_output_prompt = False
        body, _resources = exporter.from_notebook_node(nb)
        html_out.write_text(body, encoding="utf-8")
        if html_out.exists() and html_out.stat().st_size > 0:
            return html_out, "ok (nbconvert API)"
    except Exception as exc:  # noqa: BLE001
        in_proc_err = f"{type(exc).__name__}: {exc}"
    else:
        in_proc_err = ""

    # ── 2) Subprocess fallback ────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "html", "--embed-images",
             "--output", html_out.name,
             "--output-dir", str(nb_path.parent), str(nb_path)],
            check=False, capture_output=True, timeout=180, text=True,
        )
        if result.returncode == 0 and html_out.exists():
            return html_out, "ok (cli)"
        cli_err = (result.stderr or result.stdout or f"exit={result.returncode}")[:300]
    except FileNotFoundError:
        cli_err = "jupyter CLI not on PATH"
    except Exception as exc:  # noqa: BLE001
        cli_err = f"{type(exc).__name__}: {exc}"

    return None, f"in-process: {in_proc_err}; cli: {cli_err}"[:400]


def log_executed_notebook(
    tracking: Any,
    *,
    notebook_path: str | None = None,
    artifact_subdir: str = "notebook",
    also_html: bool = True,
    require_outputs: bool = False,
    verbose: bool = True,
) -> dict[str, str | None]:
    """Snapshot the running notebook (with outputs) and log it to MLflow.

    Resolution order:
      1. Colab ``_message.blocking_request("get_ipynb")`` — live JSON, no save needed.
      2. Local Jupyter Server REST API — when ``JUPYTER_TOKEN`` and ``notebook_path`` set.
      3. Disk fallback — reads ``notebook_path`` directly (only has outputs if saved).

    Logs ``<artifact_subdir>/<name>.ipynb`` and (when ``also_html``) the HTML render.
    With ``verbose=True`` (default) prints every step + the artifact path so you
    can see in Colab whether the upload happened. The default for
    ``require_outputs`` is now ``False`` so a save-without-outputs still uploads
    (you'll see a clear warning in the log).

    Returns ``{"ipynb": local_path, "html": local_path|None, "source": str,
    "html_status": str, "had_outputs": bool}``.
    """
    import nbformat

    say = print if verbose else (lambda *a, **k: None)

    # Pick up notebook_path from env (Colab Secrets / .env) when not passed.
    if notebook_path is None:
        notebook_path = os.getenv("NOTEBOOK_PATH") or None

    say(f"[publish_notebook] resolving notebook (notebook_path={notebook_path!r})")
    source = "colab-live"
    nb_json, status = _fetch_colab_ipynb()
    say(f"[publish_notebook] colab _message.get_ipynb -> {status}")

    if nb_json is None:
        source = "jupyter-rest"
        nb_json = _fetch_jupyter_rest_ipynb(notebook_path)
        say(f"[publish_notebook] jupyter REST -> {'ok' if nb_json else 'unavailable'}")

    # Editor-injected globals (VSCode __vsc_ipynb_file__, JupyterLab __session__).
    if nb_json is None and notebook_path is None:
        editor_path = _detect_vscode_or_jupyter_path()
        if editor_path:
            notebook_path = editor_path
            say(f"[publish_notebook] detected editor notebook path -> {editor_path}")

    # Disk fallback — if no explicit path, try to guess across common roots.
    if nb_json is None:
        path_to_try = notebook_path or _guess_notebook_path()
        if path_to_try and Path(path_to_try).exists():
            source = f"disk ({path_to_try})"
            nb_json = json.loads(Path(path_to_try).read_text(encoding="utf-8"))
            notebook_path = path_to_try
            say(f"[publish_notebook] read from disk -> {path_to_try}")
        else:
            say(f"[publish_notebook] disk fallback: no notebook found "
                f"(tried notebook_path={notebook_path!r}, guess={path_to_try!r})")

    if nb_json is None:
        raise RuntimeError(
            "Could not locate the notebook.\n"
            "Tried:\n"
            "  1. Colab _message.get_ipynb (this is the LIVE in-memory copy).\n"
            "  2. Jupyter Server REST API (needs JUPYTER_TOKEN + notebook_path).\n"
            "  3. Disk fallback (notebook_path arg + auto-search in /content/).\n"
            "\nFix options:\n"
            "  * In Colab — save the notebook (Ctrl/Cmd-S), then re-run this cell.\n"
            "  * Pass notebook_path=<absolute path> to log_executed_notebook().\n"
            "  * Set NOTEBOOK_PATH=<path> in env (we'll pick it up automatically)."
        )

    had_outputs = _has_outputs(nb_json)
    say(f"[publish_notebook] source={source}  cell_outputs_present={had_outputs}")
    if not had_outputs:
        say("[publish_notebook] WARNING — notebook has no cell outputs. "
            "On Colab this usually means _message returned the saved (pre-run) "
            "copy. Save the notebook (Cmd/Ctrl-S) and re-run this cell to "
            "capture the live version with outputs.")
        if require_outputs:
            raise RuntimeError(
                "Notebook has no cell outputs (require_outputs=True). "
                "Re-run the cells first, or call with require_outputs=False."
            )

    name = Path(notebook_path or "notebook.ipynb").stem
    tmp = Path(tempfile.mkdtemp(prefix="exec_nb_"))
    nb_path = tmp / f"{name}.ipynb"
    nb = nbformat.from_dict(nb_json) if isinstance(nb_json, dict) else nbformat.reads(json.dumps(nb_json), as_version=4)
    nbformat.write(nb, str(nb_path))
    tracking.save_data(nb_path, artifact_path=artifact_subdir)
    say(f"[publish_notebook] uploaded .ipynb -> artifact://{artifact_subdir}/{nb_path.name}")

    html_path: Path | None = None
    html_status = "skipped"
    if also_html:
        html_path, html_status = _render_html(nb_path)
        if html_path is not None:
            tracking.save_data(html_path, artifact_path=artifact_subdir)
            say(f"[publish_notebook] uploaded HTML  -> artifact://{artifact_subdir}/{html_path.name}")
        else:
            say(f"[publish_notebook] HTML render FAILED ({html_status}). "
                f"pip install nbconvert if it's missing.")

    log.info("logged executed notebook (source=%s) -> %s", source, nb_path.name)
    return {
        "ipynb": str(nb_path),
        "html": str(html_path) if html_path else None,
        "source": source,
        "html_status": html_status,
        "had_outputs": had_outputs,
    }
