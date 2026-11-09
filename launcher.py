#!/usr/bin/env python3
"""Thin dispatcher for OS "open with" integration.

Every double-click on a .fits file (macOS Finder wrapper, Linux .desktop
entry, Windows registry handler) runs this script. If a py-fits-preview
daemon is already resident, we hand it the file path over a local socket and
exit in a few milliseconds. Otherwise we exec into main.py, which becomes
the daemon (see ipc.start_listener in main.py).

Kept import-light (stdlib only, same as ipc.py) so the common case -- a
daemon is already running -- never pays PySide6/astropy/numba/opencv import
cost; that cost is what makes launching the app directly take ~2 seconds.
"""
import os
import sys

import ipc


def main() -> None:
    file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if ipc.connect_to_daemon(file_arg):
        return

    here = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(here, "main.py")
    os.execv(sys.executable, [sys.executable, main_py, *sys.argv[1:]])


if __name__ == "__main__":
    main()
