"""One-shot Colab bootstrap for notebooks.

The intended notebook cell is just three lines::

    if IN_COLAB:
        !pip install -q git+https://github.com/kirilyotov/BurnIT-BG.git
    from utils.colab import bootstrap
    bootstrap()

What :func:`bootstrap` does (in order):

1. Loads secrets from Colab ``userdata`` into ``os.environ`` via
   :func:`data_platform.common.set_env`. Outside Colab this picks up
   the local ``.env`` instead.
2. If running in Colab: downloads ``requirements_experiments.txt``
   straight from GitHub and pip-installs it. This keeps the ML stack
   in one place (the file) instead of duplicating package lists across
   every notebook.
3. If running in Colab: installs Tailscale and brings it up using
   ``TAILSCALE_AUTHKEY`` so MLflow/MinIO on ``*.ts.net`` are reachable.

For local development is requred to instal requirements_all.txt
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import Iterable

REPO = "kirilyotov/BurnIT-BG"
BRANCH = "master"
REQUIREMENTS_FILE = "requirements_experiments.txt"


def _in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # type: ignore[import-not-found]  # pylint: disable=import-error,no-name-in-module,unused-import  # noqa: F401
        return True
    except ImportError:
        return False


def _github_raw_url(filename: str, *, repo: str = REPO, branch: str = BRANCH) -> str:
    """Build a raw.githubusercontent.com URL for a file in the repo."""
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{filename}"


def install_package_from_github(
    *,
    repo: str = REPO,
    branch: str = BRANCH,
    extras: Iterable[str] | None = None,
    quiet: bool = True,
) -> None:
    """``pip install`` the project package directly from GitHub.

    With ``extras=["experiments"]`` this becomes
    ``pip install "burnit_bg[experiments] @ git+https://github.com/.../...git"``,
    which lets `setup.py`'s ``extras_require`` pull the ML stack in one go.
    """
    if extras:
        spec = f'burnit_bg[{",".join(extras)}] @ git+https://github.com/{repo}.git@{branch}'
    else:
        spec = f"git+https://github.com/{repo}.git@{branch}"
    args = [sys.executable, "-m", "pip", "install"]
    if quiet:
        args.append("-q")
    args.append(spec)
    print(f"[colab] {' '.join(args)}")
    subprocess.check_call(args)


def install_requirements_from_github(
    *,
    filename: str = REQUIREMENTS_FILE,
    repo: str = REPO,
    branch: str = BRANCH,
    quiet: bool = True,
) -> None:
    """Fetch a requirements file from the repo and ``pip install -r`` it.

    pip's ``-r`` flag expects a local path, so the file is downloaded to a
    temp location first. The temp file is removed on success.
    """
    url = _github_raw_url(filename, repo=repo, branch=branch)
    print(f"[colab] downloading {url}")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="requirements_", delete=False,
    ) as fh:
        with urllib.request.urlopen(url, timeout=30) as resp:
            fh.write(resp.read().decode("utf-8"))
        tmp_path = fh.name

    try:
        args = [sys.executable, "-m", "pip", "install"]
        if quiet:
            args.append("-q")
        args += ["-r", tmp_path]
        print(f"[colab] pip install -r {filename}")
        subprocess.check_call(args)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _have_command(name: str) -> bool:
    return shutil.which(name) is not None


def bootstrap(
    *,
    requirements: str | None = None,
    repo: str = REPO,
    branch: str = BRANCH,
    install_tailscale: bool = True,
    install_package: bool = False,
    extras: Iterable[str] | None = None,
    quiet: bool = False,
) -> dict[str, bool]:
    """End-to-end Colab bootstrap; returns a dict of what was done.

    By default this does **not** install anything via pip — the
    expectation is that the notebook ran::

        !pip install "burnit_bg[experiments] @ git+https://github.com/kirilyotov/BurnIT-BG.git"

    which uses ``setup.py``'s ``extras_require`` to pull both the
    package's own deps and ``requirements_experiments.txt`` in one shot.

    Args:
        requirements: Optional filename to fetch from the repo and
            ``pip install -r``. Pass e.g. ``"requirements_experiments.txt"``
            to force-reinstall a specific requirements file (useful when
            iterating on the file without bumping the package install).
            Defaults to ``None`` — assume extras already installed it.
        repo: GitHub ``owner/name``.
        branch: Branch / tag / commit to install from.
        install_tailscale: Whether to install + bring up Tailscale on Colab.
        install_package: If True, pip-install the project from GitHub
            *here* — useful if the notebook didn't run ``!pip install ...``
            already. Default False because the notebook bootstrap cell does it.
        extras: Extras to request when ``install_package=True``
            (forwarded to ``install_package_from_github``).
        quiet: Suppress chatter.

    Returns:
        ``{"in_colab", "secrets_loaded", "package_installed",
        "requirements_installed", "tailscale_connected"}``.
    """
    out = {
        "in_colab": False,
        "secrets_loaded": False,
        "package_installed": False,
        "requirements_installed": False,
        "tailscale_connected": False,
    }
    out["in_colab"] = _in_colab()

    # 1. Secrets — runs both locally (.env) and in Colab (userdata).
    try:
        from data_platform.common import set_env
        set_env(quiet=quiet)
        out["secrets_loaded"] = True
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"[colab] set_env failed (continuing): {exc}")

    if not out["in_colab"]:
        if not quiet:
            print("[colab] not in Colab — skipped GitHub install + Tailscale.")
        # Check tailnet status anyway so the return dict is meaningful.
        try:
            from utils.tailscale import is_connected
            out["tailscale_connected"] = is_connected()
        except Exception:  # noqa: BLE001
            pass
        return out

    # 2. Optional: install the project itself (when the notebook didn't).
    if install_package:
        try:
            install_package_from_github(repo=repo, branch=branch, extras=extras, quiet=quiet)
            out["package_installed"] = True
        except subprocess.CalledProcessError as exc:
            if not quiet:
                print(f"[colab] package install failed: {exc}")

    # 3. Install the ML requirements from the repo.
    if requirements:
        try:
            install_requirements_from_github(
                filename=requirements, repo=repo, branch=branch, quiet=quiet,
            )
            out["requirements_installed"] = True
        except (subprocess.CalledProcessError, urllib.error.URLError, OSError) as exc:
            if not quiet:
                print(f"[colab] requirements install failed: {exc}")

    # 4. Tailscale bring-up.
    if install_tailscale:
        try:
            from utils.tailscale import setup_in_colab
            out["tailscale_connected"] = setup_in_colab(quiet=quiet)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                print(f"[colab] tailscale setup failed: {exc}")

    if not quiet:
        print(f"[colab] bootstrap done: {out}")
    return out


__all__ = [
    "REPO",
    "BRANCH",
    "REQUIREMENTS_FILE",
    "bootstrap",
    "install_package_from_github",
    "install_requirements_from_github",
]
