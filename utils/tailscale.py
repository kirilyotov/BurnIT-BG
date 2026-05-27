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
       setup_in_colab()           # installs + starts daemon + brings up + sets proxy

On a local machine that's already on the tailnet, ``setup_in_colab()``
is a no-op (detects you're not in Colab and returns).

Notes on the install script
---------------------------

We use the official ``curl -fsSL https://tailscale.com/install.sh |
sudo sh`` one-liner. Colab's sudo is passwordless. On a hardened host
you'd want to download + inspect the script first, but Colab's runtime
is a throwaway VM so the risk profile matches.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

# Local SOCKS5 / HTTP-CONNECT proxy port that ``tailscaled`` exposes when
# started with ``--socks5-server`` + ``--outbound-http-proxy-listen``.
# Set HTTP(S)_PROXY/ALL_PROXY to this and any Python HTTP client will
# transparently reach ``*.ts.net`` hostnames through the tailnet.
DEFAULT_PROXY_PORT = 1055
DEFAULT_TAILSCALED_LOG = Path("/tmp/tailscaled.log")
DEFAULT_TAILSCALED_SOCKET = Path("/var/run/tailscale/tailscaled.sock")
DAEMON_STARTUP_TIMEOUT = 10.0  # seconds to wait for the socket to appear

log = logging.getLogger(__name__)


def _in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        # noqa/pylint suppressions: see data_platform/common/env.py for context.
        import google.colab  # type: ignore[import-not-found]  # pylint: disable=import-error,no-name-in-module,unused-import  # noqa: F401
        return True
    except ImportError:
        return False


# ##########################################################################
# State queries
# ##########################################################################


def is_installed() -> bool:
    """Return True if the ``tailscale`` binary is on PATH."""
    return shutil.which("tailscale") is not None


def is_daemon_running() -> bool:
    """Return True if a ``tailscaled`` process is currently running.

    Uses ``pgrep -x`` so we don't accidentally match unrelated processes
    that happen to contain the substring "tailscaled" in their argv.
    """
    try:
        proc = subprocess.run(
            ["pgrep", "-x", "tailscaled"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _daemon_socket_exists(socket_path: Path = DEFAULT_TAILSCALED_SOCKET) -> bool:
    """Return True if the tailscaled control socket exists and is a socket.

    The path is owned by root, so we shell out via ``sudo test`` — that
    works in Colab where sudo is passwordless. Falls back to ``os.path``
    when sudo isn't available.
    """
    try:
        proc = subprocess.run(
            ["sudo", "test", "-S", str(socket_path)],
            capture_output=True,
            timeout=5,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return socket_path.exists()


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


# ##########################################################################
# Mutating operations
# ##########################################################################


def install_tailscale(*, force: bool = False) -> None:
    """Run the official Tailscale install script (Linux only, needs sudo).

    Idempotent: skips when the binary is already present unless ``force``.
    """
    if is_installed() and not force:
        log.info("tailscale already installed; skipping")
        return
    log.info("installing tailscale via official script")
    subprocess.check_call(
        "curl -fsSL https://tailscale.com/install.sh | sudo sh",
        shell=True,
    )
    if not is_installed():
        raise RuntimeError("`tailscale` binary not found after install")


def start_daemon_userspace(
    *,
    proxy_port: int = DEFAULT_PROXY_PORT,
    log_path: Path = DEFAULT_TAILSCALED_LOG,
    socket_path: Path = DEFAULT_TAILSCALED_SOCKET,
    timeout: float = DAEMON_STARTUP_TIMEOUT,
) -> bool:
    """Start ``tailscaled`` in userspace networking mode (Colab-friendly).

    Spawns the daemon as a detached background process and waits up to
    *timeout* seconds for the control socket to appear. Idempotent — if a
    daemon is already running this is a no-op that returns True.

    The ``--socks5-server`` + ``--outbound-http-proxy-listen`` flags
    expose a local proxy on *proxy_port* that callers can point
    HTTP(S)_PROXY at via :func:`set_proxy_env`.

    Args:
        proxy_port: TCP port for the local Tailscale proxy.
        log_path: File to redirect daemon stdout/stderr into.
        socket_path: tailscaled's control socket. Used only for readiness.
        timeout: Maximum seconds to wait for the socket after spawning.

    Returns:
        True if the daemon ends up running (socket reachable). False
        otherwise — in which case *log_path* will have stderr from the
        failed launch.
    """
    if is_daemon_running() and _daemon_socket_exists(socket_path):
        log.info("tailscaled already running; skipping start")
        return True
    if not is_installed():
        raise RuntimeError("tailscale is not installed; call install_tailscale() first")
    log.info("starting tailscaled in userspace mode")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Open the log file outside the Popen call so the fd lives as long as
    # we hold a reference; subprocess inherits it for the daemon's life.
    log_fh = open(log_path, "a", buffering=1)  # noqa: SIM115
    subprocess.Popen(
        [
            "sudo", "tailscaled",
            "--tun=userspace-networking",
            f"--socks5-server=localhost:{proxy_port}",
            f"--outbound-http-proxy-listen=localhost:{proxy_port}",
        ],
        stdout=log_fh,
        stderr=log_fh,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _daemon_socket_exists(socket_path):
            return True
        time.sleep(0.2)
    log.warning(
        "tailscaled did not produce a socket within %.1fs (see %s)",
        timeout, log_path,
    )
    return False


def set_proxy_env(
    proxy_port: int = DEFAULT_PROXY_PORT,
    *,
    extra_no_proxy: list[str] | None = None,
) -> None:
    """Point HTTP(S)/ALL proxy env vars at the local tailscaled proxy.

    With userspace networking, the kernel resolver can't find
    ``*.ts.net`` hostnames. Setting the proxy means any client that
    honours these standard env vars (``requests``, ``urllib``, ``minio``,
    ``mlflow``, ``httpx``) will transparently reach the tailnet.

    Args:
        proxy_port: Port :func:`start_daemon_userspace` published on.
        extra_no_proxy: Hostnames/CIDRs to bypass the proxy for, in
            addition to ``localhost,127.0.0.1``.
    """
    http_url = f"http://localhost:{proxy_port}"
    socks_url = f"socks5://localhost:{proxy_port}"
    os.environ["HTTP_PROXY"] = http_url
    os.environ["HTTPS_PROXY"] = http_url
    os.environ["ALL_PROXY"] = socks_url
    # Lowercase versions for libs that only honour those (curl, wget).
    os.environ["http_proxy"] = http_url
    os.environ["https_proxy"] = http_url
    os.environ["all_proxy"] = socks_url

    no_proxy = ["localhost", "127.0.0.1", "::1"]
    if extra_no_proxy:
        no_proxy.extend(extra_no_proxy)
    no_proxy_value = ",".join(no_proxy)
    os.environ["NO_PROXY"] = no_proxy_value
    os.environ["no_proxy"] = no_proxy_value


def _redact_authkey(args: list[str]) -> list[str]:
    """Return ``args`` with any ``--authkey=...`` value replaced by ``***``."""
    return [a if not a.startswith("--authkey=") else "--authkey=***" for a in args]


class TailscaleUpError(RuntimeError):
    """Raised when ``tailscale up`` exits non-zero. Always carries a redacted message."""


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

    On failure raises :class:`TailscaleUpError` whose message contains the
    redacted command and tailscale's stderr — never the real auth key.
    The native ``CalledProcessError`` includes the full argv in its
    string representation, so we explicitly avoid letting it bubble.

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
    redacted = _redact_authkey(cmd)
    log.info("running: %s", " ".join(redacted))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit status {proc.returncode}"
        # Belt-and-braces: if the auth key ever appears in tailscale's own
        # output, scrub it before we raise.
        detail = detail.replace(auth_key, "***") if auth_key else detail
        raise TailscaleUpError(
            f"`{' '.join(redacted)}` failed (exit {proc.returncode}): {detail}"
        )


def setup(
    auth_key: str | None = None,
    *,
    node_hostname: str | None = None,
    extra_args: list[str] | None = None,
    userspace_daemon: bool = False,
    proxy_port: int = DEFAULT_PROXY_PORT,
    set_proxy: bool = False,
) -> bool:
    """Install Tailscale + bring it up. Returns True if the daemon is Online.

    ``auth_key`` defaults to ``$TAILSCALE_AUTHKEY``. Raises ``ValueError``
    when neither is present.

    Args:
        auth_key: Tailscale auth key. Defaults to ``$TAILSCALE_AUTHKEY``.
        node_hostname: Override the tailnet hostname this node advertises.
        extra_args: Additional flags forwarded to ``tailscale up``.
        userspace_daemon: If True, start ``tailscaled`` in userspace
            networking mode before calling ``tailscale up``. Required on
            container runtimes (Colab, Docker without ``--privileged``).
        proxy_port: Local port to expose the Tailscale proxy on when
            ``userspace_daemon=True`` or ``set_proxy=True``.
        set_proxy: If True, set HTTP(S)_PROXY/ALL_PROXY env vars after a
            successful bring-up so Python HTTP clients can reach the
            tailnet via the local proxy.
    """
    auth_key = auth_key or os.environ.get("TAILSCALE_AUTHKEY")
    if not auth_key:
        raise ValueError(
            "No auth key. Set TAILSCALE_AUTHKEY (Colab Secrets / .env / shell) "
            "or pass auth_key=... to setup()."
        )
    install_tailscale()
    if userspace_daemon:
        ok = start_daemon_userspace(proxy_port=proxy_port)
        if not ok:
            raise RuntimeError(
                f"tailscaled failed to start in userspace mode; "
                f"see {DEFAULT_TAILSCALED_LOG} for stderr"
            )
    bring_up(auth_key, node_hostname=node_hostname, extra_args=extra_args)
    if not is_connected():
        log.warning("`tailscale up` returned 0 but daemon does not report Online")
    if set_proxy:
        set_proxy_env(proxy_port)
    return is_connected()


def setup_in_colab(
    auth_key: str | None = None,
    *,
    node_hostname: str = "colab-burnit",
    proxy_port: int = DEFAULT_PROXY_PORT,
    set_proxy: bool = True,
    quiet: bool = False,
) -> bool:
    """One-line Colab bootstrap. No-op when not in Colab.

    Runs the full install → start-daemon → bring-up → set-proxy sequence.
    Returns True if the tailnet daemon is Online after the call.
    Idempotent: skips the install / start steps when already done.

    Args:
        auth_key: Tailscale auth key. Defaults to ``$TAILSCALE_AUTHKEY``.
        node_hostname: Hostname this node advertises on the tailnet.
        proxy_port: Local port for the userspace Tailscale proxy.
        set_proxy: If True (default), export HTTP(S)_PROXY / ALL_PROXY so
            Python HTTP clients can reach ``*.ts.net`` hosts through the
            local proxy. Required for MinIO / MLflow access on Colab.
        quiet: Suppress per-step status lines.
    """
    if not _in_colab():
        if not quiet:
            print("[tailscale] not in Colab; skipping (assumed already on tailnet).")
        return is_connected()

    if is_connected():
        if not quiet:
            print(f"[tailscale] already connected as {hostname()!r}; skipping setup.")
        if set_proxy:
            set_proxy_env(proxy_port)
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
        print("[tailscale] installing + starting daemon + bringing up …")
    try:
        ok = setup(
            auth_key=auth_key,
            node_hostname=node_hostname,
            userspace_daemon=True,
            proxy_port=proxy_port,
            set_proxy=set_proxy,
        )
    except TailscaleUpError as exc:
        # Already-scrubbed message from bring_up — safe to print verbatim.
        if not quiet:
            print(f"[tailscale] setup failed: {exc}")
        return False
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        # Defensive scrub in case a different code path leaks the key.
        msg = str(exc)
        if auth_key:
            msg = msg.replace(auth_key, "***")
        if not quiet:
            print(f"[tailscale] setup failed: {msg}")
        return False

    if not quiet:
        info = status() or {}
        peers = info.get("Peer") or {}
        proxy_note = f", proxy=http://localhost:{proxy_port}" if set_proxy else ""
        print(
            f"[tailscale] connected={ok}, host={hostname()!r}, "
            f"peers={len(peers)}{proxy_note}"
        )
    return ok


__all__ = [
    "DEFAULT_PROXY_PORT",
    "TailscaleUpError",
    "bring_up",
    "hostname",
    "install_tailscale",
    "is_connected",
    "is_daemon_running",
    "is_installed",
    "set_proxy_env",
    "setup",
    "setup_in_colab",
    "start_daemon_userspace",
    "status",
]
