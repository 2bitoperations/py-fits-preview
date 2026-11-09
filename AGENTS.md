# Agent Instructions

## Python Environment
- **Always use `uv`** to invoke any Python code authored or present in this repository. For example, use `uv run python script.py` instead of just `python script.py`.

## General Notes
- *Add future crucial notes for follow-on agentic work here.*

## Loading pipeline invariants (main.py `MainWindow`)
- macOS delivers the launch file **twice** (argv, then a `QFileOpenEvent` from Cocoa). `_load_fits` must stay idempotent for the path already shown/loading. Offscreen (`QT_QPA_PLATFORM=offscreen`) runs do *not* get the second event, so test launch behaviour with a real window.
- At most one pipeline per path (`_inflight`). The current file and preloads share it; never start a second load for a path that is already in flight.
- All reads go through `_IoScheduler` (one thread, lowest priority first, queued reads truly cancellable). Current file = `_PRIO_CURRENT`, preloads = 2·distance. Images live on an SMB NAS at ~35 MB/s (~2.5 s per 93 MB frame), so queue order *is* the UI latency.
- Every finished load is cached in `_on_pipeline_done`, even if the user moved on; it's displayed only if it is still `_current_path`.
