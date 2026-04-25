# FITS Preview

A blazingly fast, multi-threaded FITS image viewer designed for rapid triage of astrophotography and scientific data. Built with `PySide6` and `astropy`.

FITS Preview bypasses the traditional sluggish FITS viewers by utilizing predictive multi-threaded preloading, GPU-accelerated MTF stretch reconstruction, and dynamic RAM-aware caching, allowing you to instantly scrub backward and forward through gigabytes of raw sensor data in real-time.

## Features

- **Asynchronous Pipeline**: Decouples disk I/O, debayering, and quantization to never block the main UI thread.
- **Dynamic Preloading**: Automatically probes your system memory, allocating up to 10% of physical RAM to cache adjacent frames and look-ahead buffers.
- **Persistent Geometry**: Flawlessly persists your viewport zoom and pan coordinates when scrubbing across multiple FITS files of identical geometry.
- **Hardware-Accelerated Stretches**: Rapid, automated application of MTF (PixInsight style), ASINH, and ZScale functions.

## Hotkeys

| Action | Keybinding |
| --- | --- |
| **Next / Previous Image** | `Right Arrow` / `Left Arrow` |
| **Zoom In / Out** | `Scroll Wheel` |
| **Pan** | `Click and Drag` |
| **Zoom to 1:1 Scale** | `Ctrl + 0` (or `Cmd + 0`) |
| **Fit to Window** | `Ctrl + 9` (or `Cmd + 9`) |
| **Mark Bad / Reject** | `Spacebar` (Flags with 🤮 icon) |
| **Trash Bad Images** | `Enter` / `Return` |
| **Toggle Header Panel** | `I` (Cycles Hidden → Transparent → Opaque) |
| **Toggle Histogram** | `H` |
| **MTF Stretch** | `Ctrl + 1` (or `Cmd + 1`) |
| **ASINH Stretch** | `Ctrl + 2` (or `Cmd + 2`) |
| **ZScale Stretch** | `Ctrl + 3` (or `Cmd + 3`) |
| **Auto Stretch** | `Ctrl + 4` (or `Cmd + 4`) |

## HUD Explanations

- **Buffer Gauges (Bottom Right)**: The two vertical bars indicate the current status of the RAM preload buffer. The left bar tracks the "look behind" cache (images you recently viewed), while the right bar tracks the "look ahead" cache. The bars will fade out completely when both queues are fully populated, keeping your view clear.
- **Histogram (Bottom Left)**: A live, floating read-out of the pixel density across the stretched image. This display is purely informational and ignores black-border clipping. It can be toggled at any time using `H`.

## Configuration

Configurations are persisted globally in `~/.config/py-fits-preview.conf`.
These include options for the `header_state`, the `preload_ahead` depth, `cache_max` depth, and custom header columns. 

You can manually override the RAM-computed buffer queues using CLI arguments:
```bash
uv run python3 main.py --ahead 10 --behind 2 --cache 15 /path/to/image.fits
```

## OS Integration (Set as Default Viewer)

You can install FITS Preview as the default double-click handler for `.fit` and `.fits` files natively in your operating system.

**macOS (Finder)**:
Run `./install_mac.sh`. This generates an AppleScript Application Wrapper inside `~/Applications/py-fits-preview.app`. You can then Right Click any `.fits` file in Finder → Get Info → Open With: "py-fits-preview" → Change All.

**Linux (GNOME / KDE)**:
Run `./install_linux.sh`. This generates a `py-fits-preview.desktop` shortcut inside `~/.local/share/applications` and updates your XDG mime database so your file manager natively knows to route FITS documents to the pipeline.

## License

This software is released under the **Creative Commons Attribution-NonCommercial 4.0 International** (CC BY-NC 4.0) License. You are free to modify and adapt this code for non-commercial purposes, provided appropriate attribution is given.
