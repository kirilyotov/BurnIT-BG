"""Tailscale bootstrap helper for Colab / headless notebooks.

The k3s cluster's MLflow and MinIO are exposed on a Tailscale tailnet
(``*.tail1e4f6a.ts.net``). To reach them from a fresh Colab runtime you
need to install Tailscale and bring it up with a one-time auth key.

Auth key flow
-------------

1. Mint an auth key at https://login.tailscale.com/admin/settings/keys.
   *Recommendation:* tick **reusable** and **ephemeral** so a disposable
   Colab node doesn't leave a stale device behind.
2. In Colab: open the **Secrets** panel (key icon in the left sidebar),
   add a secret named ``TAILSCALE_AUTHKEY`` with the key value, and
   toggle "Notebook access" ON.
3. In your first notebook cell::

       from data_platform.common import set_env
       from utils.tailscale import setup_in_colab

       set_env(quiet=True)        # pulls TAILSCALE_AUTHKEY into os.environ
       setup_in_colab()           # installs Tailscale and brings it up

On a local machine that's already on the tailnet, ``setup_in_colab()``
is a no-op (detects you're not in Colab and returns).

Notes on the install script
---------------------------

We use the official ``curl -fsSL https://tailscale.com/install.sh | sh``
one-liner, which needs ``sudo``. That's fine in Colab — sudo is
passwordless there. On a hardened host you'd want to download +
inspect the script first, but Colab's runtime is a throwaway VM so the
risk profile matches.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

log = logging.getLogger(__name__)


def _in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        # noqa/pylint suppressions: see data_platform/common/env.py for context.
        import google.colab  # type: ignore[import-not-found]  # pylint: disable=import-error,no-name-in-module,unused-import  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# State queries
# ---------------------------------------------------------------------------


def is_installed() -> bool:
    """Return True if the ``tailscale`` binary is on PATH."""
    return shutil.which("tailscale") is not None


def is_connected() -> bool:
    """Return True if the tailscale daemon reports itself as Online."""
    data = status()
    if not data:
        return False
    return bool(data.get("Self", {}).get("Online", False))


def status() -> dict[str, Any] | None:
    """Return parsed ``tailscale status --json`` output, or ``None`` on failure."""
    if not is_installed():
        return None
    try:
        out = subprocess.check_output(
            ["tailscale", "status", "--json"],
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def hostname() -> str | None:
    """Return the tailscale-assigned hostname for this node, or ``None``."""
    data = status()
    if not data:
        return None
    return data.get("Self", {}).get("HostName")


# ---------------------------------------------------------------------------
# Mutating operations
# ---------------------------------------------------------------------------


def install_tailscale(*, force: bool = False) -> None:
    """Run the official Tailscale install script (Linux only, needs sudo).

    Idempotent: skips when the binary is already present unless ``force``.
    """
    if is_installed() and not force:
        log.info("tailscale already installed; skipping")
        return
    log.info("installing tailscale via official script")
    subprocess.check_call(
        "curl -fsSL https://tailscale.com/install.sh | sh",
        shell=True,
    )
    if not is_installed():
        raise RuntimeError("`tailscale` binary not found after install")


def bring_up(
    auth_key: str,
    *,
    node_hostname: str | None = None,
    ssh: bool = False,
    accept_routes: bool = True,
    accept_dns: bool = True,
    extra_args: list[str] | None = None,
) -> None:
    """Run ``sudo tailscale up`` with the given auth key.

    Args:
        auth_key: Tailscale auth key (one-time or reusable).
        node_hostname: Override hostname this node advertises on the tailnet.
        ssh: Enable Tailscale SSH on this node.
        accept_routes: Accept subnet routes advertised by other nodes
            (needed when services are exposed via subnet routers).
        accept_dns: Use the tailnet's MagicDNS (required to resolve
            ``*.ts.net`` hostnames).
        extra_args: Additional flags passed verbatim to ``tailscale up``.
    """
    if not is_installed():
        raise RuntimeError("tailscale is not installed; call install_tailscale() first")
    cmd = ["sudo", "tailscale", "up", f"--authkey={auth_key}"]
    if node_hostname:
        cmd.append(f"--hostname={node_hostname}")
    if ssh:
        cmd.append("--ssh")
    if accept_routes:
        cmd.append("--accept-routes")
    if accept_dns:
        cmd.append("--accept-dns")
    if extra_args:
        cmd += list(extra_args)
    redacted = [c if "authkey" not in c else "--authkey=***" for c in cmd]
    log.info("running: %s", " ".join(redacted))
    subprocess.check_call(cmd)


def setup(
    auth_key: str | None = None,
    *,
    node_hostname: str | None = None,
    extra_args: list[str] | None = None,
) -> bool:
    """Install Tailscale + bring it up. Returns True if the daemon is Online.

    ``auth_key`` defaults to ``$TAILSCALE_AUTHKEY``. Raises ``ValueError``
    when neither is present.
    """
    auth_key = auth_key or os.environ.get("TAILSCALE_AUTHKEY")
    if not auth_key:
        raise ValueError(
            "No auth key. Set TAILSCALE_AUTHKEY (Colab Secrets / .env / shell) "
            "or pass auth_key=... to setup()."
        )
    install_tailscale()
    bring_up(auth_key, node_hostname=node_hostname, extra_args=extra_args)
    if not is_connected():
        log.warning("`tailscale up` returned 0 but daemon does not report Online")
    return is_connected()


def setup_in_colab(
    auth_key: str | None = None,
    *,
    node_hostname: str = "colab-burnit",
    quiet: bool = False,
) -> bool:
    """One-line Colab bootstrap. No-op when not in Colab.

    Returns True if Tailscale is up after the call. Returns the current
    state (``is_connected()``) without trying to set up when already
    connected — running ``tailscale up`` twice in a row works, but the
    second call is wasted work.
    """
    if not _in_colab():
        if not quiet:
            print("[tailscale] not in Colab; skipping (assumed already on tailnet).")
        return is_connected()

    if is_connected():
        if not quiet:
            print(f"[tailscale] already connected as {hostname()!r}; skipping setup.")
        return True

    auth_key = auth_key or os.environ.get("TAILSCALE_AUTHKEY")
    if not auth_key:
        if not quiet:
            print(
                "[tailscale] no TAILSCALE_AUTHKEY in env. Add it to Colab Secrets "
                "(key icon → Add new secret → TAILSCALE_AUTHKEY) and re-run "
                "set_env() before this call."
            )
        return False

    if not quiet:
        print("[tailscale] installing + bringing up …")
    try:
        ok = setup(auth_key=auth_key, node_hostname=node_hostname)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        if not quiet:
            print(f"[tailscale] setup failed: {exc}")
        return False

    if not quiet:
        info = status() or {}
        peers = info.get("Peer") or {}
        print(f"[tailscale] connected={ok}, host={hostname()!r}, peers={len(peers)}")
    return ok


__all__ = [
    "is_installed",
    "is_connected",
    "status",
    "hostname",
    "install_tailscale",
    "bring_up",
    "setup",
    "setup_in_colab",
]
