# Performance notes

## Where the time goes (2026-11, 8288×5644 mono 16-bit, 93 MB, SMB NAS)

| Stage                                   | Time       | Where          |
|-----------------------------------------|------------|----------------|
| Read (`_load_path_raw_data`)            | ~2.2–2.8 s | `fits-io` thread |
| Quantize + FWHM + MTF (`_compute_from_raw`) | ~0.27 s | compute pool (4) |
| Show cached image (`FitsView` + histogram) | ~30 ms  | main thread    |

The network link (~35 MB/s here) is the bottleneck by an order of
magnitude, so the pipeline is organised around **read order**, not compute:

- One I/O thread (`_IoScheduler`) does all reads: current image first,
  then preloads nearest-first (priority 2·distance, travel direction
  winning ties). Queued reads are re-prioritised on every navigation and
  removed when they leave the window.
- Each path has at most one load in flight (`MainWindow._inflight`); the
  current image and preloads share it.
- Every finished load is cached; it is displayed only if still current.
- Preload window: 12 ahead / 10 behind (configurable). Cache limit: 100
  images or 10 % of RAM, measured from real per-image size; the window is
  never evicted.
- `fits.open(memmap=False)`: one bulk read is faster than mmap page faults
  over SMB and keeps network I/O on the I/O thread.

## Remaining levers

- **Network**: 93 MB at gigabit wire speed is ~0.85 s vs ~2.5 s observed.
  Wired link / SMB settings matter more than any code change.
- Compute is already small; the Numba colour MTF kernel and OpenCV
  debayer are single passes. Not worth optimising further until the read
  is much faster.

## Pitfalls seen

- macOS sends the launch file twice (argv + `QFileOpenEvent`); `_load_fits`
  must be idempotent. Offscreen Qt runs don't reproduce this.
- FWHM: star candidates include galaxy knots / HII regions on extended
  targets (M33 centre reads ~7.8 px vs ~4.2 px in the corners), so the
  gauge is best for comparing frames of the same target, not as an
  absolute seeing number.
