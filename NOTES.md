# Performance Plan

## Time budget per file (steady state, 2 preload workers)

| Stage                          | Time     | % of total |
|-------------------------------|----------|------------|
| I/O (`_load_path_raw_data`)   | ~100 ms  | 4%         |
| Debayer                       | ~540 ms  | 22%        |
| Quantize (clip + astype + flipud) | ~660 ms | 27%    |
| MTF render (`_apply_mtf_color`) | ~1,100 ms | 45%    |
| **Total**                     | **~2,400 ms** |       |

First cold load spikes to ~7 s because all three workers start simultaneously,
thrashing the memory bus and numpy thread pool.

## Root cause

All stages operate on a ~242 MB float32 array (3672×5496×3). The pipeline
makes ~5 full passes over that data. Cores stall on memory bandwidth, not
instruction throughput — which is why peak aggregate CPU was only ~50% even
with work queued. Adding workers past a threshold makes this worse.

## I/O architecture (do this first)

**Funnel all file reads through a single dedicated I/O worker.**

Right now the preload pool (N workers) and the stretch executor each call
`_load_path_raw_data` independently, meaning multiple threads can be
seeking/reading different FITS files concurrently. Even on NVMe this is
counter-productive: the OS prefetcher works best with sequential reads, and
competing threads cause seek amplification and cache thrashing.

Proposed model:
- One `_io_executor` (1 worker, thread name `fits-io`) owns all disk reads.
- After a read completes, the raw `(data, header)` is handed off to a
  `_compute_executor` (N workers) that runs debayer + quantize + MTF.
- The stretch executor for the current file becomes a priority lane in the
  same compute pool (submit first, or use a separate 1-worker queue).

This serializes I/O, maximizes read throughput per file, and lets the compute
workers stay busy without competing for disk bandwidth.

## Tier 1 — drop-in, low risk

1. **Single I/O worker** (see above) — serializes reads, lets OS prefetch
   efficiently, eliminates disk contention between preload threads.

2. **Switch debayer to OpenCV** — `cv2.cvtColor(raw, cv2.COLOR_BAYER_RG2RGB)`
   runs in C++. Drops debayer from ~540 ms to ~30–50 ms. One import, two lines
   changed, ~480 ms saved per file, no algorithmic change. Need to map our
   RGGB/BGGR/GRBG/GBRG patterns to OpenCV's `BAYER_*` constants.

3. **Use subsampled data for percentile stats in `_build_stretch_data`** —
   `np.nanpercentile` is called on the full 20 MP array to find `lo`/`hi`.
   We already have `_subsample` (500 K pixels); just apply it here. Saves
   most of the "quantize" time since the clip/astype/flipud itself is fast.

4. **Increase compute worker count to 4–5** — with debayer and stats sped up,
   memory bandwidth per worker drops and more parallelism helps. With 4
   workers at ~1 s each (post-OpenCV), throughput is ~4 files/sec — enough
   that a 3-ahead preload never misses under normal browsing.

## Tier 2 — medium effort, large payoff

5. **MTF render on GPU via Apple MLX** — `_apply_mtf_color` makes ~8 passes
   over the 242 MB array (per-channel stats ×3, bg_sub, luma, safe_luma,
   ratio, result). MLX has a near-identical API to NumPy; the rational
   function, elementwise ops, and weighted sums are all natively supported.
   Estimated drop: 1,100 ms → ~50–80 ms. ~20-line change.

## Tier 3 — maximum performance, higher complexity

6. **Full pipeline on GPU in one pass** — combine debayer + quantize + MTF
   into a single MLX (or Metal) computation graph. Raw 16-bit pixels go in,
   8-bit RGB comes out, with zero intermediate host copies. Eliminates
   essentially all memory-bandwidth overhead. Rough estimate: ~200 ms total
   per file (dominated by I/O), vs ~2,400 ms today.

## Expected outcomes

| Config                              | Time/file | Throughput   |
|------------------------------------|-----------|--------------|
| Current (2 workers, NumPy)         | ~2,400 ms | ~0.8 files/s |
| +single I/O + OpenCV debayer       | ~1,800 ms | ~1.1 files/s |
| +subsampled stats + 4 workers      | ~1,100 ms | ~3.6 files/s |
| +MLX MTF                           | ~200 ms   | >>10 files/s |

At ~200 ms/file a 3-ahead preload means the cache is always warm unless the
user navigates faster than ~5 files/second.
