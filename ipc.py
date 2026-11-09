"""Single-instance IPC for py-fits-preview.

Deliberately stdlib-only (no numpy/Qt/astropy/etc). ``launcher.py`` imports
this module *before* deciding whether to pay the cost of importing the rest
of the app, so keeping this import-light is the whole point: probing for a
running daemon must be near-free.

Uses multiprocessing.connection, which transparently maps to a Unix domain
socket on macOS/Linux and a named pipe on Windows.
"""
import errno
import os
import socket
import sys
import tempfile
import threading
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


def connect_to_daemon(file_path: str | None, timeout: float = 2.0) -> bool:
    """Hand a file path off to an already-running daemon, if one exists.

    Returns True if a daemon accepted the handoff (caller should exit
    immediately without building a window), False if no daemon is running
    or it doesn't answer within ``timeout`` (caller should run its own
    window).  The timeout matters: a wedged daemon must never turn a
    double-click into a launcher that hangs forever.
    """
    # multiprocessing's Client has no timeout, and a socket-level timeout
    # would make its raw-fd reads non-blocking (spurious failures against a
    # healthy daemon).  Run the whole handshake in a worker thread instead;
    # if it doesn't finish in time, give up and leave the thread blocked
    # (daemon thread: it dies with the process).
    result = [False]

    def _handoff():
        try:
            conn = Client(ADDRESS, authkey=_authkey())
            try:
                conn.send(file_path)
            finally:
                conn.close()
            result[0] = True
        except Exception:  # not running, refused, auth mismatch...
            pass

    t = threading.Thread(target=_handoff, name="fits-ipc-client", daemon=True)
    t.start()
    t.join(timeout)
    return result[0] and not t.is_alive()


def _socket_is_stale() -> bool:
    """True if the socket file exists but nothing is listening on it
    (the previous daemon crashed or was killed without cleaning up)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(ADDRESS)
        return False
    except ConnectionRefusedError:
        return True
    except OSError:
        return False
    finally:
        s.close()


def start_listener() -> Listener | None:
    """Try to become the daemon by binding the well-known address.

    Returns a Listener on success, or None if another live process already
    holds it (or the bind otherwise fails) -- callers should fall back to
    running as a normal, non-resident single window in that case.  A socket
    file left behind by a crashed daemon is removed and the bind retried;
    otherwise no later launch could ever become the daemon again.
    """
    try:
        return Listener(ADDRESS, authkey=_authkey())
    except OSError as e:
        if (sys.platform == "win32" or e.errno != errno.EADDRINUSE
                or not _socket_is_stale()):
            return None
    try:
        os.unlink(ADDRESS)
    except FileNotFoundError:
        pass
    except OSError:
        return None
    try:
        return Listener(ADDRESS, authkey=_authkey())
    except OSError:
        return None
