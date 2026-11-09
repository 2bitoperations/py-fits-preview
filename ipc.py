"""Single-instance IPC for py-fits-preview.

Deliberately stdlib-only (no numpy/Qt/astropy/etc). ``launcher.py`` imports
this module *before* deciding whether to pay the cost of importing the rest
of the app, so keeping this import-light is the whole point: probing for a
running daemon must be near-free.

Uses multiprocessing.connection, which transparently maps to a Unix domain
socket on macOS/Linux and a named pipe on Windows.
"""
import os
import sys
import tempfile
from multiprocessing.connection import Client, Listener


def _state_dir() -> str:
    """Directory for the daemon socket/authkey. Falls back to a per-user temp
    directory if the preferred config location isn't writable (daemon mode
    is a nice-to-have -- that should never be a reason the app fails to
    start), so this never raises.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        preferred = os.path.join(base, "py-fits-preview")
    else:
        preferred = os.path.join(os.path.expanduser("~"), ".config", "py-fits-preview")
    fallback = os.path.join(tempfile.gettempdir(), f"py-fits-preview-{os.getuid() if hasattr(os, 'getuid') else 'user'}")
    for d in (preferred, fallback):
        try:
            os.makedirs(d, exist_ok=True, mode=0o700)
            return d
        except OSError:
            continue
    return fallback


def _address() -> str:
    if sys.platform == "win32":
        return r"\\.\pipe\py-fits-preview-{}".format(os.environ.get("USERNAME", "user"))
    return os.path.join(_state_dir(), "daemon.sock")


def _authkey() -> bytes:
    """Per-user shared secret so only our own processes can talk to the daemon."""
    path = os.path.join(_state_dir(), "authkey")
    try:
        with open(path, "rb") as f:
            key = f.read()
        if key:
            return key
    except FileNotFoundError:
        pass
    key = os.urandom(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


ADDRESS = _address()


def connect_to_daemon(file_path: str | None, timeout: float = 0.5) -> bool:
    """Hand a file path off to an already-running daemon, if one exists.

    Returns True if a daemon accepted the handoff (caller should exit
    immediately without building a window), False if no daemon is running
    (caller should become the daemon / show its own window).
    """
    try:
        conn = Client(ADDRESS, authkey=_authkey())
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    try:
        conn.send(file_path)
    finally:
        conn.close()
    return True


def start_listener() -> Listener | None:
    """Try to become the daemon by binding the well-known address.

    Returns a Listener on success, or None if another process already holds
    it (or the bind otherwise fails) -- callers should fall back to running
    as a normal, non-resident single window in that case.
    """
    try:
        return Listener(ADDRESS, authkey=_authkey())
    except OSError:
        return None
