import logging
import os
import sys
import time
import traceback
import argparse
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
import cv2
import numpy as np

_log = logging.getLogger(__name__)
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.stats import median_abs_deviation
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSizePolicy, QWidget,
    QVBoxLayout, QProgressBar, QGraphicsOpacityEffect,
    QSplitter, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMenu
)
from PySide6.QtGui import (
    QImage, QPixmap, QFileOpenEvent, QKeySequence, QShortcut,
    QPainter, QColor, QPen, QBrush, QPolygon,
)
from PySide6.QtCore import Qt, QEvent, QTimer, Signal, QPoint, QPropertyAnimation, QRectF


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray,
               vmin: float | None = None,
               vmax: float | None = None,
               gamma: float = 0.5) -> np.ndarray:
    """Stretch arr → uint8.

    vmin/vmax: data values for black/white points (default: 1st/99th percentile).
    gamma: mid-point position as a fraction of [vmin, vmax] (0–1, default 0.5 = linear).
    """
    a = arr.astype(np.float32)
    if vmin is None:
        vmin = float(np.nanpercentile(a, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(a, 99))
    if vmax == vmin:
        vmax = vmin + 1.0
    t = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    if abs(gamma - 0.5) > 1e-4:
        g = float(np.clip(gamma, 1e-6, 1.0 - 1e-6))
        exp = np.log(0.5) / np.log(g)
        t = np.power(np.maximum(t, 0.0), exp)
    return (t * 255.0).astype(np.uint8)


def _gray_to_qimage(gray: np.ndarray) -> QImage:
    """(H, W) uint8 → QImage grayscale. Thread-safe."""
    gray = np.ascontiguousarray(gray)
    h, w = gray.shape
    return QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """(H, W, 3) uint8 → QImage RGB888. Thread-safe."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Bayer demosaicing
# ---------------------------------------------------------------------------

# OpenCV Bayer → BGR conversion codes.
#
# WARNING: OpenCV's Bayer codes are notoriously misnamed due to historical
# internal representations. The code named `COLOR_BAYER_BG2BGR` actually
# expects an `RGGB` pattern, and `COLOR_BAYER_RG2BGR` expects `BGGR`.
# If mapped naively by name, Red and Blue channels are swapped, creating
# a cyan cast in the final image.
#
# Mapping: FITS BAYERPAT value → OpenCV 2BGR code that correctly reads it.
_BAYER_CV2_CODE: dict[str, int] = {
    "RGGB": cv2.COLOR_BAYER_BG2BGR,
    "BGGR": cv2.COLOR_BAYER_RG2BGR,
    "GRBG": cv2.COLOR_BAYER_GB2BGR,
    "GBRG": cv2.COLOR_BAYER_GR2BGR,
}


def _debayer(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Demosaic a 2-D Bayer array → (H, W, 3) float32 via OpenCV.

    Accepts any numeric dtype; non-uint16 input is clipped to [0, 65535]
    and cast to uint16 before passing to cv2 (which requires u8 or u16).
    Output is RGB (channel 0 = R) — OpenCV produces BGR; we flip here.
    """
    pat = pattern.upper().strip()
    code = _BAYER_CV2_CODE.get(pat)
    if code is None:
        raise ValueError(f"Unsupported Bayer pattern: {pattern!r}")
    if raw.dtype != np.uint16:
        raw = np.clip(raw, 0, 65535).astype(np.uint16)
    bgr = cv2.cvtColor(raw, code)
    return np.ascontiguousarray(bgr[:, :, ::-1]).astype(np.float32)  # BGR → RGB


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def fits_data_to_qimage(data: np.ndarray,
                        header=None,
                        vmin: float | None = None,
                        vmax: float | None = None,
                        gamma: float = 0.5) -> QImage:
    """Convert FITS data → QImage. Thread-safe.

    With vmin/vmax=None uses per-channel auto-stretch (preload path).
    With explicit vmin/vmax applies global stretch to all channels.
    """
    data = np.squeeze(data)

    if data.ndim == 3:
        if data.shape[0] == 3:
            channels = [data[i] for i in range(3)]
        elif data.shape[2] == 3:
            channels = [data[:, :, i] for i in range(3)]
        else:
            return fits_data_to_qimage(data[0], header, vmin, vmax, gamma)
        rgb = np.stack([_normalize(c, vmin, vmax, gamma) for c in channels], axis=-1)
        return _rgb_to_qimage(np.flipud(rgb))

    if data.ndim == 2:
        bayer = None
        if header is not None:
            bayer = (header.get("BAYERPAT") or
                     header.get("COLORTYP") or
                     header.get("BAYER"))
        if bayer and str(bayer).upper().strip() in _BAYER_CV2_CODE:
            rgb_f = _debayer(data, str(bayer))
            rgb = np.stack([_normalize(rgb_f[:, :, i], vmin, vmax, gamma)
                            for i in range(3)], axis=-1)
            return _rgb_to_qimage(np.flipud(rgb))
        return _gray_to_qimage(np.flipud(_normalize(data, vmin, vmax, gamma)))

    raise ValueError(f"Cannot display data with shape {data.shape}")


def fits_data_to_pixmap(data, header=None, vmin=None, vmax=None, gamma=0.5):
    return QPixmap.fromImage(fits_data_to_qimage(data, header, vmin, vmax, gamma))



def _load_path_raw_data(path: str) -> tuple[np.ndarray, object] | tuple[None, None]:
    """Load raw array + header for histogram/stretch use. Main-thread only."""
    t0   = time.perf_counter()
    name = os.path.basename(path)
    try:
        with fits.open(path) as hdul:
            hdu = next((h for h in hdul if h.data is not None
                        and h.data.ndim >= 2), None)
            if hdu is None:
                return None, None
            # Do NOT use .copy(); returning a view allows astropy to use mmap
            data, header = hdu.data, hdu.header
        _log.debug("_load_path_raw_data: %.1f ms  %s",
                   (time.perf_counter() - t0) * 1000, name)
        return data, header
    except Exception:
        _log.exception("_load_path_raw_data failed: %s", name)
        return None, None


def _extract_wb_multipliers(header) -> tuple[float, float, float] | None:
    """Try to extract R/G/B white-balance gain multipliers from common FITS keywords.

    Returns (r_gain, g_gain, b_gain) normalised so g_gain == 1.0, or None if no
    recognised keyword set is found.  Checked keyword triplets (in priority order):
        RSCALE / GSCALE / BSCALE2   — INDI / KStars capture
        WB_RED / WB_GREEN / WB_BLUE — some DSO capture tools
        MULR   / MULG   / MULB      — miscellaneous convention
    Note: the standard FITS keyword BSCALE encodes data scaling (not blue gain)
    and is intentionally excluded.
    """
    if header is None:
        return None
    for r_key, g_key, b_key in [
        ("RSCALE",  "GSCALE",    "BSCALE2"),
        ("WB_RED",  "WB_GREEN",  "WB_BLUE"),
        ("MULR",    "MULG",      "MULB"),
    ]:
        try:
            r = np.asarray(header[r_key]).item()
            g = np.asarray(header[g_key]).item()
            b = np.asarray(header[b_key]).item()
            if isinstance(g, (int, float)) and g > 0:
                return float(r) / float(g), 1.0, float(b) / float(g)
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _build_stretch_data(data: np.ndarray,
                        header=None) -> tuple[np.ndarray, float, float]:
    """Pre-process raw FITS data into a uint16 array for fast LUT-based stretch.

    Handles grayscale, Bayer (debayered → colour), and pre-stacked colour.
    White-balance multipliers are applied to colour arrays when present in the
    FITS header (see _extract_wb_multipliers).
    Returns (u16, lo, hi) where u16 is (H, W) or (H, W, 3) already
    flipped for display, and lo/hi are the data values that map to 0 / 65535.
    """
    t0 = time.perf_counter()
    raw = np.squeeze(data)

    bayer = None
    if header is not None:
        bayer = (header.get("BAYERPAT") or
                 header.get("COLORTYP") or
                 header.get("BAYER"))

    if raw.ndim == 2 and bayer and str(bayer).upper().strip() in _BAYER_CV2_CODE:
        t_deb = time.perf_counter()
        # Pass the original (non-float) array; _debayer converts to uint16 for OpenCV.
        d = _debayer(raw, str(bayer))         # → (H, W, 3) float32
        _log.debug("_build_stretch_data: debayer %.1f ms  shape=%s bayer=%s",
                   (time.perf_counter() - t_deb) * 1000, d.shape, bayer)
    elif raw.ndim == 3:
        d = raw.astype(np.float32)
        if d.shape[0] == 3:
            d = np.transpose(d, (1, 2, 0))   # (3, H, W) → (H, W, 3)
        elif d.shape[2] != 3:
            d = d[0]                          # unknown cube → first plane
    else:
        d = raw.astype(np.float32)            # 2-D grayscale, no bayer

    # Apply white-balance correction on colour arrays
    if d.ndim == 3:
        wb = _extract_wb_multipliers(header)
        if wb is not None:
            _log.debug("_build_stretch_data: applying WB multipliers R=%.3f B=%.3f", wb[0], wb[2])
            d = d * np.array(wb, dtype=np.float64)[np.newaxis, np.newaxis, :]

    t_q = time.perf_counter()
    lo = float(np.min(d))
    hi = float(np.max(d))
    if hi <= lo:
        hi = lo + 1.0

    u16 = np.clip((d - lo) / (hi - lo) * 65535, 0, 65535).astype(np.uint16)
    result = np.ascontiguousarray(np.flipud(u16))
    
    # Fast FWHM geometric estimation (background thread)
    t_fwhm = time.perf_counter()
    fwhm = 0.0
    try:
        d_plane = d[:, :, 1] if d.ndim == 3 and d.shape[2] == 3 else d[:, :, 0] if d.ndim == 3 else d
        
        # We already extracted a random subsample for quantisation logic. 
        # Use its 99.5th percentile to threshold the full image plane.
        sub_c = _subsample(d_plane.ravel())
        if sub_c.size > 0:
            thresh = float(np.nanpercentile(sub_c, 99.5))
            mask = (d_plane > thresh).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            areas = stats[:, cv2.CC_STAT_AREA]
            valid = (areas > 4) & (areas < 150)
            valid[0] = False # Ignore vast background blob
            
            valid_centroids = centroids[valid]
            if len(valid_centroids) > 50:
                valid_centroids = valid_centroids[:50]
                
            if len(valid_centroids) > 0:
                fwhm = compute_backend.estimate_fwhm(d_plane, valid_centroids.astype(np.int32))
    except Exception as e:
        _log.error("FWHM estimation failed: %s", e)
        
    _log.debug("_build_stretch_data: quantize %.1f ms | fwhm %.2fpx (%.1f ms) | total %.1f ms",
               (t_fwhm - t_q) * 1000, fwhm, (time.perf_counter() - t_fwhm) * 1000,
               (time.perf_counter() - t0) * 1000)
    return result, lo, hi, fwhm


# ---------------------------------------------------------------------------
# LUT-building helpers for non-linear stretch algorithms
# ---------------------------------------------------------------------------

def _compute_asinh_lut(stretch_lo: float, stretch_hi: float,
                        vmin: float, vmax: float,
                        stretch_factor: float = 5.0) -> np.ndarray:
    """Build a 65536-entry uint8 LUT for asinh stretch.

    Data is clipped to [vmin, vmax], normalized to [0, 1], then:
        t' = arcsinh(stretch_factor * t) / arcsinh(stretch_factor)
    """
    idx      = np.arange(65536, dtype=np.float64)
    data_val = stretch_lo + idx / 65535.0 * (stretch_hi - stretch_lo)
    span     = vmax - vmin if vmax > vmin else 1.0
    t        = np.clip((data_val - vmin) / span, 0.0, 1.0)
    if stretch_factor > 0:
        denom = np.arcsinh(stretch_factor)
        if denom > 0:
            t = np.arcsinh(stretch_factor * t) / denom
    return (np.clip(t, 0.0, 1.0) * 255.0).astype(np.uint8)


def _compute_zscale_lut(stretch_lo: float, stretch_hi: float,
                         raw_flat: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Build a uint8 LUT using ZScale limits (returns lut, z1, z2)."""
    finite = raw_flat[np.isfinite(raw_flat)]
    z1, z2 = ZScaleInterval().get_limits(finite)
    span     = z2 - z1 if z2 > z1 else 1.0
    idx      = np.arange(65536, dtype=np.float64)
    data_val = stretch_lo + idx / 65535.0 * (stretch_hi - stretch_lo)
    t        = np.clip((data_val - z1) / span, 0.0, 1.0)
    return (t * 255.0).astype(np.uint8), float(z1), float(z2)


def _mtf_rational(t: np.ndarray, M: float) -> np.ndarray:
    """PixInsight MTF rational function applied element-wise to t ∈ [0, 1].

        f(0) = 0
        f(x) = (M−1)·x / ((2M−1)·x − M)   for x > 0
        f(1) = 1

    M is the midtone-balance parameter: the input value that maps to 0.5.
    """
    denom  = (2.0 * M - 1.0) * t - M
    safe   = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    result = (M - 1.0) * t / safe
    return np.where(t == 0.0, 0.0, np.clip(result, 0.0, 1.0))


def _mtf_stats(samples: np.ndarray,
               shadow_clip: float = -2.8) -> tuple[float, float, float]:
    """Compute MTF black point, normalization span, and midtone-balance M.

    Excludes zero (and near-zero) pixels before computing statistics to avoid
    the "black border" problem where stacking artifacts drag the median to zero.

    Returns (black, span, M) where:
        black — value mapping to 0 (shadow clip = median + shadow_clip * MAD,
                shadow_clip < 0 so black is *below* the sky median)
        span  — range [black, nonzero_max] used to normalise samples to [0, 1]
        M     — midtone-balance for _mtf_rational; maps the normalised sky
                median to target_bkg = 0.25 by solving
                    MTF(m_norm, M) = 0.25  analytically:
                    M = 3·m_norm / (2·m_norm + 1)
                This keeps M in (0, 1) for all m_norm in (0, 1) and
                ensures the sky background appears at 25% grey, not black.
    """
    sub    = _subsample(samples)
    nonzero = sub[sub > 1e-6 * float(sub.max()) + 1e-9]
    if nonzero.size < 100:                          # fall back if almost nothing
        nonzero = sub
    med   = float(np.median(nonzero))
    mad   = float(median_abs_deviation(nonzero, scale="normal"))
    black = med + shadow_clip * mad                 # shadow_clip < 0 → below median
    span  = float(nonzero.max()) - black
    if span <= 0:
        span = 1.0
    m_norm = float(np.clip((med - black) / span, 1e-9, 1.0 - 1e-9))
    # Correct M: solve MTF(m_norm, M) = 0.25  →  M = 3·m_norm / (2·m_norm + 1)
    # (Wrong formula M = 0.25 / m_norm gives M >> 1 for typical m_norm ~ 0.1,
    #  which maps the sky median to a *negative* value that clips to black.)
    M = 3.0 * m_norm / (2.0 * m_norm + 1.0)
    return black, span, M


def _compute_mtf_lut(u16_flat: np.ndarray) -> np.ndarray:
    """Build a 65536-entry uint8 MTF LUT from a flat u16 sample array.

    Used for grayscale images where per-channel split is not applicable.
    Uses the same span that _mtf_stats used to compute M — essential so the
    LUT normalization is consistent with the M value.
    """
    black, span, M = _mtf_stats(u16_flat.astype(np.float64))
    _log.debug("_compute_mtf_lut: black=%.1f  span=%.1f  M=%.5f", black, span, M)
    idx = np.arange(65536, dtype=np.float64)
    t   = np.clip((idx - black) / span, 0.0, 1.0)
    return (_mtf_rational(t, M) * 255.0).astype(np.uint8)


import compute_backend

def _apply_mtf_color(u16: np.ndarray) -> np.ndarray:
    """Per-channel unlinked MTF with luminance-based colour preservation.
    
    Uses compute_backend (Numba/GPU) for the heavy math, avoiding intermediate memory allocations.
    """
    # ── Subsample for statistics gathering (avoids full array allocation) ──
    H, W, C = u16.shape
    N = H * W
    
    if N > _SUBSAMPLE_N:
        idx = _GLOBAL_RNG.choice(N, size=_SUBSAMPLE_N, replace=False)
        u16_flat = u16.reshape(-1, 3)
        sub_u16 = u16_flat[idx]
    else:
        sub_u16 = u16.reshape(-1, 3)
        
    sub_lin = sub_u16.astype(np.float32) / 65535.0

    # ── Per-channel black points and spans ──
    black = np.zeros(3, dtype=np.float32)
    span  = np.ones(3, dtype=np.float32)
    for c in range(3):
        # We can pass the subsampled channel directly to _mtf_stats
        b, s, M_c = _mtf_stats(sub_lin[:, c])
        black[c] = b
        span[c]  = s
        _log.debug("_apply_mtf_color: ch=%d  black=%.5f  span=%.5f  M=%.5f",
                   c, b, s, M_c)

    # ── Derive Luminance MTF parameter (M_luma) from subsample ──
    sub_bg_sub = np.clip(
        (sub_lin - black[np.newaxis, :]) / span[np.newaxis, :], 
        0.0, 1.0
    )
    luma_lin = (0.2126 * sub_bg_sub[:, 0] +
                0.7152 * sub_bg_sub[:, 1] +
                0.0722 * sub_bg_sub[:, 2])

    luma_pos = luma_lin[luma_lin > 1e-6]
    luma_med = float(np.median(luma_pos)) if luma_pos.size >= 100 else float(np.median(luma_lin))
    m_norm_luma = float(np.clip(luma_med, 1e-9, 1.0 - 1e-9))
    M_luma = 3.0 * m_norm_luma / (2.0 * m_norm_luma + 1.0)
    _log.debug("_apply_mtf_color: luma_med=%.5f  m_norm=%.5f  M=%.5f",
               luma_med, m_norm_luma, M_luma)

    # ── Dispatch to backend for full-image execution ──
    return compute_backend.apply_mtf_color(u16, black, span, M_luma)


# ---------------------------------------------------------------------------
# Preload / cache / sampling constants
# ---------------------------------------------------------------------------

_COMPUTE_WORKERS   = 4          # CPU workers for debayer + quantize + MTF
_SUBSAMPLE_N       = 500_000    # pixels used for median/MAD/histogram stats
_GLOBAL_RNG        = np.random.default_rng(0)


def _subsample(arr: np.ndarray, n: int = _SUBSAMPLE_N) -> np.ndarray:
    """Return a flat random subsample of arr, at most n elements.

    Used for all statistical operations (median, MAD, percentile, histogram)
    where the full 60M-element array is overkill.  500 K pixels gives
    statistically equivalent results for sky-background estimation.
    """
    flat = arr.ravel()
    if flat.size <= n:
        return flat
    return _GLOBAL_RNG.choice(flat, size=n, replace=False)

# ---------------------------------------------------------------------------
# Directory navigation helpers
# ---------------------------------------------------------------------------

_FITS_EXTENSIONS = frozenset({".fits", ".fit", ".fts"})


import re

def _natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def _fits_siblings(path: str) -> list[str]:
    directory = os.path.dirname(os.path.abspath(path))
    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in _FITS_EXTENSIONS
    ]
    return sorted(files, key=_natural_sort_key)


# ---------------------------------------------------------------------------
# Histogram overlay widget
# ---------------------------------------------------------------------------

class HistogramOverlay(QWidget):
    """Semi-transparent histogram showing active stretch bounds."""

    W, H      = 300, 84
    HIST_TOP  = 4
    HIST_BOT  = 54       # bottom of primary histogram bar area
    FULL_TOP  = 66       # top of full-range histogram
    FULL_BOT  = 78       # bottom of full-range histogram
    MX        = 10       # left/right content margin

    # Marker colours
    _COLORS = {
        "black": QColor(220, 220, 220),
        "mid":   QColor(255, 185, 30),
        "white": QColor(220, 220, 220),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.W, self.H)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._log_counts: np.ndarray | None = None
        self._full_log_counts: np.ndarray | None = None
        
        self._data_lo = 0.0
        self._data_hi = 1.0
        self._abs_lo  = 0.0
        self._abs_hi  = 1.0
        
        self._vmin  = 0.0
        self._vmax  = 1.0
        self._gamma = 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, data: np.ndarray,
                 vmin: float, vmax: float, gamma: float = 0.5):
        flat = _subsample(np.squeeze(data).ravel().astype(np.float32))
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        lo = float(np.percentile(flat, 0.1))
        hi = float(np.percentile(flat, 99.9))
        if hi == lo:
            hi = lo + 1.0
        self._data_lo, self._data_hi = lo, hi

        counts, _ = np.histogram(flat, bins=256, range=(lo, hi))
        log_c = np.log1p(counts.astype(np.float64))
        peak  = log_c.max()
        self._log_counts = log_c / peak if peak > 0 else log_c
        
        # Absolute full-range histogram
        abs_lo = float(np.min(flat))
        abs_hi = float(np.max(flat))
        if abs_hi == abs_lo:
            abs_hi = abs_lo + 1.0
        self._abs_lo, self._abs_hi = abs_lo, abs_hi
        
        counts_full, _ = np.histogram(flat, bins=256, range=(abs_lo, abs_hi))
        log_c_full = np.log1p(counts_full.astype(np.float64))
        peak_full = log_c_full.max()
        self._full_log_counts = log_c_full / peak_full if peak_full > 0 else log_c_full

        self._vmin, self._vmax, self._gamma = vmin, vmax, gamma
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _content_w(self) -> float:
        return self.W - 2 * self.MX

    def _data_to_x(self, v: float) -> float:
        t = (v - self._data_lo) / (self._data_hi - self._data_lo)
        return self.MX + t * self._content_w()

    def _abs_to_x(self, v: float) -> float:
        t = (v - self._abs_lo) / (self._abs_hi - self._abs_lo)
        return self.MX + t * self._content_w()

    def _gamma_x(self) -> float:
        xb = self._data_to_x(self._vmin)
        xw = self._data_to_x(self._vmax)
        return xb + self._gamma * (xw - xb)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(18, 18, 18, 210)))
        p.drawRoundedRect(0, 0, self.W, self.H, 6, 6)

        # Primary Histogram bars
        if self._log_counts is not None:
            n   = len(self._log_counts)
            bw  = self._content_w() / n
            bar_max_h = self.HIST_BOT - self.HIST_TOP - 2
            p.setPen(Qt.PenStyle.NoPen)
            for i, v in enumerate(self._log_counts):
                bh = int(v * bar_max_h)
                if bh < 1:
                    continue
                bx = int(self.MX + i * bw)
                p.setBrush(QBrush(QColor(170, 170, 170, 210)))
                p.drawRect(bx, self.HIST_BOT - bh, max(1, int(bw)), bh)
                
        # Full-Range Sub-Histogram
        if self._full_log_counts is not None:
            n = len(self._full_log_counts)
            bw = self._content_w() / n
            bar_max_h = self.FULL_BOT - self.FULL_TOP
            p.setPen(Qt.PenStyle.NoPen)
            for i, v in enumerate(self._full_log_counts):
                bh = int(v * bar_max_h)
                if bh < 1:
                    continue
                bx = int(self.MX + i * bw)
                # Paint absolute minimum/maximum bins red if clipped, else dark grey
                if i == 0 and v > 0.1:
                    p.setBrush(QBrush(QColor(255, 80, 80, 210)))
                elif i == n - 1 and v > 0.1:
                    p.setBrush(QBrush(QColor(255, 80, 80, 210)))
                else:
                    p.setBrush(QBrush(QColor(100, 100, 100, 180)))
                p.drawRect(bx, self.FULL_BOT - bh, max(1, int(bw)), bh)

        # Primary Markers
        marker_xs = {
            "black": self._data_to_x(self._vmin),
            "mid":   self._gamma_x(),
            "white": self._data_to_x(self._vmax),
        }
        for name, mx in marker_xs.items():
            col = self._COLORS[name]
            ix  = int(round(mx))

            # Dashed vertical line through primary histogram
            pen = QPen(QColor(col.red(), col.green(), col.blue(), 140), 1,
                       Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(ix, self.HIST_TOP, ix, self.HIST_BOT)
            
        # Sub-layer bounds indicators (Draw black/white stretch bounds on the full range graph)
        b_x = int(round(self._abs_to_x(self._vmin)))
        w_x = int(round(self._abs_to_x(self._vmax)))
        p.setPen(QPen(self._COLORS["black"], 1))
        p.drawLine(b_x, self.FULL_TOP, b_x, self.FULL_BOT)
        p.setPen(QPen(self._COLORS["white"], 1))
        p.drawLine(w_x, self.FULL_TOP, w_x, self.FULL_BOT)

        # Labels for min/max boundaries
        font = p.font()
        font.setPixelSize(9)
        p.setFont(font)
        p.setPen(QColor(170, 170, 170, 255))
        
        lo_str = f"{self._data_lo:.1f}" if self._data_lo < 100 else f"{int(self._data_lo)}"
        p.drawText(self.MX, self.HIST_BOT + 9, lo_str)
        
        hi_str = f"{self._data_hi:.1f}" if self._data_hi < 100 else f"{int(self._data_hi)}"
        hi_w = p.fontMetrics().horizontalAdvance(hi_str)
        p.drawText(self.W - self.MX - hi_w, self.HIST_BOT + 9, hi_str)

# ---------------------------------------------------------------------------
# Stretch data container
# ---------------------------------------------------------------------------

class _StretchData:
    """Holds the pre-quantised u16 array and associated metadata needed for
    interactive stretch.  Kept separate from QImage so the two can be cached
    and delivered independently (QImage for immediate display, u16 for controls).
    """
    __slots__ = ("u16", "lo", "hi", "raw_flat", "mtf_qimage", "mtf_rgb", "raw_header", "fwhm")

    def __init__(self, u16: np.ndarray, lo: float, hi: float,
                 raw_flat: np.ndarray, mtf_qimage: "QImage | None" = None,
                 mtf_rgb: np.ndarray | None = None, raw_header: dict | None = None,
                 fwhm: float = 0.0):
        self.u16        = u16         # (H, W) or (H, W, 3) uint16, flipped
        self.lo         = lo          # data value → u16 == 0
        self.hi         = hi          # data value → u16 == 65535
        self.raw_flat   = raw_flat    # float32 flat of pre-debayer data for histogram
        self.mtf_qimage = mtf_qimage  # pre-rendered MTF QImage for instant cache-hit display
        self.mtf_rgb    = mtf_rgb     # raw uint8 array for headless mode
        self.raw_header = raw_header
        self.fwhm       = fwhm


def _compute_from_raw(path: str, raw_data: np.ndarray,
                      raw_header, headless: bool = False) -> "_StretchData | None":
    """Compute stage: debayer + quantize + MTF render.  Runs in compute pool.

    Receives raw FITS data already loaded from disk by the I/O worker.
    Returns None on any error.
    """
    name = os.path.basename(path)
    t0 = time.perf_counter()
    u16, lo, hi, fwhm = _build_stretch_data(raw_data, raw_header)
    raw_flat = _subsample(np.squeeze(raw_data).ravel().astype(np.float32))
    t_mtf = time.perf_counter()
    if u16.ndim == 2:
        lut    = _compute_mtf_lut(u16.ravel())
        mtf_rgb = lut[u16]
    else:
        mtf_rgb = _apply_mtf_color(u16)
        
    mtf_qi = None
    if not headless:
        mtf_qi = _gray_to_qimage(mtf_rgb) if mtf_rgb.ndim == 2 else _rgb_to_qimage(mtf_rgb)
    _log.debug("[compute] debayer+quantize %.1f ms  MTF %.1f ms  total %.1f ms  %s",
               (t_mtf - t0) * 1000,
               (time.perf_counter() - t_mtf) * 1000,
               (time.perf_counter() - t0) * 1000, name)
    return _StretchData(u16, lo, hi, raw_flat, mtf_qi, mtf_rgb, dict(raw_header), fwhm=fwhm)

import json
from pathlib import Path

import os

class ConfigManager:
    def __init__(self):
        if sys.platform == "win32":
            config_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "py-fits-preview"
        else:
            config_dir = Path.home() / ".config"
        self.config_path = config_dir / "py-fits-preview.conf"
        
        try:
            if sys.platform == "win32":
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                ram_bytes = stat.ullTotalPhys
            else:
                ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except Exception:
            ram_bytes = 16 * 1024**3
        
        allowed = int((ram_bytes * 0.10) / (121 * 1024**2))
        allowed = max(5, allowed)
        
        self.config = {
            "header_state": 0, # 0=Hidden, 1=Transparent, 2=Opaque
            "header_width_pct": 0.15,
            "show_all_headers": False,
            "checked_headers": [],
            "preload_ahead": int(allowed * 0.6),
            "preload_behind": int(allowed * 0.15),
            "cache_max": allowed - int(allowed * 0.6) - int(allowed * 0.15),
            "histogram_visible": True
        }
        self.load()

    def load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    self.config.update(data)
            except Exception as e:
                _log.error("Failed to load config: %s", e)

    def save(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            _log.error("Failed to save config: %s", e)

_config = ConfigManager()

class HeaderPanelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setShowGrid(False)
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(34, 34, 34, 0.85);")
        self.table.setStyleSheet("background-color: transparent; color: white;")
        
        layout.addWidget(self.table)
        
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.table.cellChanged.connect(self._on_cell_changed)
        self._raw_header = None
        
    def _show_context_menu(self, pos):
        menu = QMenu(self)
        show_all_action = menu.addAction("Show All")
        show_all_action.setCheckable(True)
        show_all_action.setChecked(_config.config["show_all_headers"])
        
        action = menu.exec(self.mapToGlobal(pos))
        if action == show_all_action:
            _config.config["show_all_headers"] = action.isChecked()
            _config.save()
            self.update_header(self._raw_header)
            
    def _on_cell_changed(self, row, col):
        if col != 0: return
        item = self.table.item(row, 0)
        if not item: return
        key = item.text()
        checked_list = _config.config["checked_headers"]
        
        changed = False
        if item.checkState() == Qt.CheckState.Checked:
            if key not in checked_list:
                checked_list.append(key)
                changed = True
        else:
            if key in checked_list:
                checked_list.remove(key)
                changed = True
                
        if changed:
            _config.config["checked_headers"] = checked_list
            _config.save()
            
    def update_header(self, header):
        self._raw_header = header
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        
        if not header:
            self.table.blockSignals(False)
            return
            
        show_all = _config.config["show_all_headers"]
        checked_list = _config.config["checked_headers"]
        
        for k, v in header.items():
            if not show_all and k not in checked_list:
                continue
                
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            key_item = QTableWidgetItem(str(k))
            if show_all:
                key_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                key_item.setCheckState(Qt.CheckState.Checked if k in checked_list else Qt.CheckState.Unchecked)
            else:
                key_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                
            val_item = QTableWidgetItem(str(v))
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, val_item)
            
        self.table.blockSignals(False)

class MainContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)
        
        self.header_panel = HeaderPanelWidget()
        self.header_panel.hide()
        
        self.view_container = QWidget()
        self.view_layout = QVBoxLayout(self.view_container)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter.addWidget(self.header_panel)
        self.splitter.addWidget(self.view_container)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        
        self._current_view = None
        self._gauges = BufferGauges(self.view_container)
        self._histogram = HistogramOverlay(self.view_container)
        self._histogram.setVisible(_config.config.get("histogram_visible", True))
        
        self._fwhm_gauge = FwhmGaugeOverlay(self.view_container)
        self._fwhm_tooltip = QLabel(self.view_container)
        self._fwhm_tooltip.setStyleSheet("color: white; background: rgba(0,0,0,180); border-radius: 4px; padding: 4px; font-size: 11px;")
        self._fwhm_tooltip.hide()
        self._fwhm_gauge.hover_changed.connect(self._on_fwhm_hover)
        
        self.view_container.installEventFilter(self)
        
        self.apply_header_state()
        
    def _on_fwhm_hover(self, active: bool):
        if active:
            self._fwhm_tooltip.setText(self._fwhm_gauge.get_tooltip_text())
            self._fwhm_tooltip.adjustSize()
            gx = self._fwhm_gauge.x()
            gy = self._fwhm_gauge.y()
            self._fwhm_tooltip.move(gx - self._fwhm_tooltip.width() - 8,
                                    gy + self._fwhm_gauge.height() // 2 - self._fwhm_tooltip.height() // 2)
            self._fwhm_tooltip.show()
            self._fwhm_tooltip.raise_()
        else:
            self._fwhm_tooltip.hide()
            
    def update_fwhm_db(self, db, bad, current):
        self._fwhm_gauge.update_db(db, bad, current)
        
    def set_view(self, view: QWidget, header_data=None):
        if self._current_view:
            self.view_layout.removeWidget(self._current_view)
            self._current_view.setParent(None)
            if hasattr(self._current_view, 'teardown'):
                self._current_view.teardown()
            self._current_view.deleteLater()
        self._current_view = view
        self.view_layout.addWidget(self._current_view)
        
        if hasattr(self._current_view, 'stretch_applied'):
            self._current_view.stretch_applied.connect(self._histogram.set_data)
            if hasattr(self._current_view, '_raw_flat') and self._current_view._raw_flat is not None:
                self._histogram.set_data(self._current_view._raw_flat,
                                         self._current_view._vmin,
                                         self._current_view._vmax, 0.5)
            
        self._gauges.raise_()
        self._histogram.raise_()
        self._fwhm_gauge.raise_()
        self._fwhm_tooltip.raise_()
        self.header_panel.update_header(header_data)
        
    def eventFilter(self, obj, event):
        if obj == self.view_container and event.type() == QEvent.Type.Resize:
            margin = 12
            self._gauges.move(self.view_container.width() - self._gauges.width() - margin,
                              self.view_container.height() - self._gauges.height() - margin)
            self._gauges.raise_()
            
            self._histogram.move(margin, self.view_container.height() - self._histogram.height() - margin)
            self._histogram.raise_()
            
            self._fwhm_gauge.move(self.view_container.width() - self._fwhm_gauge.width() - margin, margin)
            self._fwhm_gauge.raise_()
            if self._fwhm_tooltip.isVisible():
                self._on_fwhm_hover(True)
                self._fwhm_tooltip.raise_()
        return super().eventFilter(obj, event)

    def update_gauges(self, back_pct: float, fwd_pct: float):
        self._gauges.update_gauges(back_pct, fwd_pct)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        state = _config.config["header_state"]
        if state == 1:
            hw = int(self.width() * _config.config["header_width_pct"])
            self.header_panel.setGeometry(0, 0, hw, self.height())

    def toggle_histogram(self):
        vis = not self._histogram.isVisible()
        self._histogram.setVisible(vis)
        _config.config["histogram_visible"] = vis
        _config.save()

    def get_view(self) -> QWidget:
        return self._current_view
        
    def toggle_header_state(self):
        state = _config.config["header_state"]
        state = (state + 1) % 3
        _config.config["header_state"] = state
        _config.save()
        self.apply_header_state()
        
    def apply_header_state(self):
        state = _config.config["header_state"]
        vp_width = self.width() if self.width() > 100 else 1000
        hw = int(vp_width * _config.config["header_width_pct"])
        
        if state == 0:
            self.header_panel.setParent(self.splitter)
            self.splitter.insertWidget(0, self.header_panel)
            self.header_panel.hide()
        elif state == 1:
            self.header_panel.setParent(self)
            self.header_panel.setStyleSheet("background-color: rgba(34, 34, 34, 0.85);")
            self.header_panel.setGeometry(0, 0, hw, self.height())
            self.header_panel.show()
            self.header_panel.raise_()
        elif state == 2:
            self.header_panel.setParent(self.splitter)
            self.splitter.insertWidget(0, self.header_panel)
            self.header_panel.setStyleSheet("background-color: #222222;")
            self.header_panel.show()
            self.splitter.setSizes([hw, vp_width - hw])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        state = _config.config["header_state"]
        if state == 1:
            hw = int(self.width() * _config.config["header_width_pct"])
            self.header_panel.setGeometry(0, 0, hw, self.height())

from PySide6.QtCore import Property

class BufferGauges(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.back_pct = 0.0
        self.fwd_pct = 0.0
        
        self.setFixedSize(16, 60)
        
        self._opacity = 0.9
        self._anim = QPropertyAnimation(self, b"opacity")
        self._anim.setDuration(500)
        self._is_fading_out = False

    def get_opacity(self):
        return self._opacity

    def set_opacity(self, value):
        self._opacity = value
        self.update()

    opacity = Property(float, get_opacity, set_opacity)

    def update_gauges(self, back_pct: float, fwd_pct: float):
        self.back_pct = back_pct
        self.fwd_pct = fwd_pct
        self.update()
        
        is_full = (self.back_pct >= 0.99) and (self.fwd_pct >= 0.99)
        if is_full and not self._is_fading_out and self._opacity > 0.0:
            self._anim.stop()
            self._anim.setStartValue(self._opacity)
            self._anim.setEndValue(0.0)
            self._anim.start()
            self._is_fading_out = True
        elif not is_full and (self._is_fading_out or self._opacity < 0.9):
            self._anim.stop()
            self._anim.setStartValue(self._opacity)
            self._anim.setEndValue(0.9)
            self._anim.start()
            self._is_fading_out = False

    def paintEvent(self, event):
        if self._opacity <= 0.0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)
        w, h = self.width(), self.height()
        bar_w = int(w * 0.4)
        
        def draw_bar(x, pct):
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(x, 0, bar_w, h - 1)
            
            if pct > 0:
                color = QColor.fromHsv(int(pct * 120), 200, 200)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                fill_h = int((h - 2) * pct)
                painter.drawRect(x + 1, h - 1 - fill_h, bar_w - 1, fill_h)

        draw_bar(0, self.back_pct)
        draw_bar(w - bar_w, self.fwd_pct)

class FwhmGaugeOverlay(QWidget):
    """Top-right translucent gauge tracking session FWHMs."""
    hover_changed = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 200)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self._db: dict[str, float] = {}
        self._bad: set[str] = set()
        self._current_path: str = ""
        self._valid_fwhms = []
        self._avg = 0.0

    def update_db(self, db: dict[str, float], bad: set[str], current: str):
        self._db = db.copy()
        self._bad = bad.copy()
        self._current_path = current
        self._valid_fwhms = [v for k, v in self._db.items() if k not in self._bad and v > 0.0]
        self._avg = sum(self._valid_fwhms) / len(self._valid_fwhms) if self._valid_fwhms else 0.0
        self.update()

    def get_tooltip_text(self) -> str:
        current_val = self._db.get(self._current_path, 0.0)
        n = len(self._valid_fwhms)
        return f"FWHM: {current_val:.2f}px | Avg: {self._avg:.2f} (n={n})"

    def enterEvent(self, event):
        self.hover_changed.emit(True)
        
    def leaveEvent(self, event):
        self.hover_changed.emit(False)
        
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        # Explicitly erase the backing store to prevent alpha accumulation
        # and "ghosting" when the dynamic scale boundaries change.
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillRect(rect, Qt.GlobalColor.transparent)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(18, 18, 18, 180)))
        p.drawRoundedRect(rect, 4, 4)
        
        if not self._valid_fwhms: return
            
        min_f = min(self._valid_fwhms)
        max_f = max(self._valid_fwhms)
        if max_f - min_f < 0.01:
            min_f -= 1.0
            max_f += 1.0
            
        pad = 8
        draw_h = rect.height() - 2 * pad
        
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(100, 150, 255, 60)))
        for val in self._valid_fwhms:
            t = (val - min_f) / (max_f - min_f)
            y = rect.height() - pad - int(t * draw_h)
            p.drawRect(4, y, rect.width() - 8, 2)
            
        current_val = self._db.get(self._current_path, 0.0)
        if current_val > 0.0:
            t = (current_val - min_f) / (max_f - min_f)
            t = max(0.0, min(1.0, t))
            y = rect.height() - pad - int(t * draw_h)
            p.setBrush(QBrush(QColor(255, 255, 255, 255)))
            p.drawRect(2, y - 1, rect.width() - 4, 4)

class LoadingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        label = QLabel("u know i'm loading dem imgs bro 🙃")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        prog = QProgressBar()
        prog.setRange(0, 0)
        prog.setFixedWidth(200)
        
        layout.addWidget(label)
        layout.addWidget(prog)

# ---------------------------------------------------------------------------
# Qt widgets
# ---------------------------------------------------------------------------

class MagnifierOverlay(QWidget):
    """Floating magnifying glass overlay showing nearest-neighbor zoomed pixels and raw values."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(160, 160)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._crop = None
        self._text = ""

    def set_data(self, crop: QPixmap, text: str):
        self._crop = crop
        self._text = text
        self.update()

    def paintEvent(self, event):
        if self._crop is None or self._crop.isNull():
            return
            
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        # Draw the scaled nearest-neighbor crop
        rect = self.rect()
        p.drawPixmap(rect, self._crop)
        
        # Outer Border
        p.setPen(QPen(QColor(255, 255, 255, 100), 2))
        p.drawRect(0, 0, rect.width() - 1, rect.height() - 1)
        
        # Center Crosshair
        cx, cy = rect.width() // 2, rect.height() // 2
        p.setPen(QPen(QColor(255, 0, 0, 200), 1))
        p.drawLine(cx, cy - 8, cx, cy + 8)
        p.drawLine(cx - 8, cy, cx + 8, cy)
        
        # Text Readout Background
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(self._text)
        th = fm.height()
        
        text_bg = QRectF(5, rect.height() - th - 10, tw + 10, th + 5)
        p.setBrush(QColor(0, 0, 0, 180))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(text_bg, 4, 4)
        
        # Text Render
        p.setPen(QColor(255, 255, 255))
        p.drawText(text_bg, Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()


class FitsView(QGraphicsView):
    stretch_applied = Signal(object, float, float, float)

    def __init__(self, qimage: QImage,
                 stretch: _StretchData | None = None,
                 saved_viewport=None, is_bad: bool = False, parent=None):
        pixmap = QPixmap.fromImage(qimage)
        super().__init__(parent)

        scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem(pixmap)
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        scene.addItem(self._item)
        self.setScene(scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Mouse Tracking & Magnifier
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._magnifier = MagnifierOverlay(self.viewport())
        self._magnifier.hide()

        self._pixmap_size = (pixmap.width(), pixmap.height())
        self._saved_viewport = saved_viewport
        self._fit_pending    = False
        self._initial_fit_done = saved_viewport is not None
        self._is_fitted      = True

        self._stretch_u16  = None
        self._stretch_lo   = 0.0
        self._stretch_hi   = 1.0
        self._raw_flat     = None
        self._vmin         = 0.0
        self._vmax         = 1.0
        self._auto_vmin    = 0.0
        self._auto_vmax    = 1.0
        self._gamma        = 0.5

        self._bad_label = QLabel("🤮", self.viewport())
        self._bad_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self._bad_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._bad_label.setStyleSheet("color: white; background: transparent;")
        self._bad_label.setVisible(is_bad)

        self._eol_label = QLabel("🛑", self.viewport())
        self._eol_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._eol_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._eol_label.setStyleSheet("color: white; background: transparent;")
        self._eol_label.setVisible(False)

        if stretch is not None:
            self._install_stretch(stretch)

        # Shortcuts (Ctrl→Cmd on macOS)
        QShortcut(QKeySequence("Ctrl+0"), self, self._zoom_one_to_one)
        QShortcut(QKeySequence("Ctrl+9"), self, self._fit)

    def teardown(self):
        """Clean up QGraphicsScene and explicitly release the pixmap memory."""
        if self.scene() is not None:
            self.scene().clear()
        self._item = None
        self._stretch_u16 = None
        self._raw_flat = None

    def _install_stretch(self, stretch: _StretchData):
        """Apply a _StretchData object — called at init time or asynchronously."""
        t0 = time.perf_counter()
        self._stretch_u16 = stretch.u16
        self._stretch_lo  = stretch.lo
        self._stretch_hi  = stretch.hi
        self._raw_flat    = stretch.raw_flat
        sub = _subsample(self._raw_flat)
        self._vmin = float(np.nanpercentile(sub, 1))
        self._vmax = float(np.nanpercentile(sub, 99))
        self._auto_vmin = self._vmin
        self._auto_vmax = self._vmax
        _log.debug("FitsView._install_stretch: stats %.1f ms",
                   (time.perf_counter() - t0) * 1000)
        self.stretch_applied.emit(self._raw_flat, self._vmin, self._vmax, 0.5)
        
        if stretch.mtf_qimage is not None:
            self._item.setPixmap(QPixmap.fromImage(stretch.mtf_qimage))
        else:
            self.apply_mtf_stretch()

    def set_stretch_data(self, stretch: _StretchData):
        """Deliver stretch data that arrived asynchronously after init."""
        self._install_stretch(stretch)

    # ------------------------------------------------------------------
    # Viewport state (preserved across navigation)
    # ------------------------------------------------------------------

    @property
    def pixmap_size(self):
        return self._pixmap_size

    def viewport_state(self) -> tuple | None:
        if self._fit_pending:
            _log.debug("viewport_state: fit_pending is True, returning None")
            return None
        if self._saved_viewport is not None:
            _log.debug("viewport_state: returning cached _saved_viewport")
            return self._saved_viewport
        return (
            self.transform(),
            self.horizontalScrollBar().value(),
            self.verticalScrollBar().value(),
            self._pixmap_size
        )

    def _apply_pending_restore(self):
        if self._saved_viewport is not None:
            # Check length for backwards compatibility with any remaining 3-element tuples in cache
            if len(self._saved_viewport) == 4:
                t, h_val, v_val, psize = self._saved_viewport
                self._saved_viewport = None
                self.setTransform(t)
                
                # Defer setting scroll values until the layout completes
                def restore_scroll():
                    self.horizontalScrollBar().setValue(h_val)
                    self.verticalScrollBar().setValue(v_val)
                QTimer.singleShot(0, restore_scroll)
                
            elif len(self._saved_viewport) == 3:
                t, c, psize = self._saved_viewport
                self._saved_viewport = None
                self.setTransform(t)
                QTimer.singleShot(0, lambda: self.centerOn(c))
                
            self._is_fitted = False

    def _fit(self):
        self._fit_pending = False
        self._is_fitted = True
        self.fitInView(self.scene().itemsBoundingRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def _zoom_one_to_one(self):
        self._fit_pending = False
        self.resetTransform()

    def showEvent(self, event):
        super().showEvent(event)
        if self._saved_viewport is None and not self._initial_fit_done:
            self._initial_fit_done = True
            self._fit_pending = True
            QTimer.singleShot(0, self._fit)
        self._position_overlays()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._saved_viewport is not None:
            # Wait until the view has a non-trivial size from the layout
            if self.width() > 100 and self.height() > 100:
                self._apply_pending_restore()
            return

        if self._initial_fit_done and getattr(self, '_is_fitted', False):
            self._fit()
        self._position_overlays()

    # ------------------------------------------------------------------
    # Overlays position
    # ------------------------------------------------------------------

    def _position_overlays(self):
        vp = self.viewport()
        margin = 12
        
        font = self._bad_label.font()
        font.setPixelSize(max(24, int(vp.width() * 0.10)))
        self._bad_label.setFont(font)
        self._bad_label.adjustSize()
        self._bad_label.move(vp.width() - self._bad_label.width() - margin, margin)
        self._bad_label.raise_()
        
        font_eol = self._eol_label.font()
        font_eol.setPixelSize(max(24, int(vp.width() * 0.10)))
        self._eol_label.setFont(font_eol)
        self._eol_label.adjustSize()
        self._eol_label.move(margin, margin)
        self._eol_label.raise_()
    def set_bad(self, is_bad: bool):
        self._bad_label.setVisible(is_bad)

    def show_eol(self):
        self._eol_label.setVisible(True)
        if hasattr(self, '_eol_anim') and self._eol_anim.state() == QPropertyAnimation.State.Running:
            self._eol_anim.stop()
        self._eol_eff = QGraphicsOpacityEffect(self._eol_label)
        self._eol_label.setGraphicsEffect(self._eol_eff)
        self._eol_anim = QPropertyAnimation(self._eol_eff, b"opacity")
        self._eol_anim.setDuration(1000)
        self._eol_anim.setStartValue(1.0)
        self._eol_anim.setEndValue(0.0)
        self._eol_anim.finished.connect(lambda: self._eol_label.setVisible(False))
        self._eol_anim.start(QPropertyAnimation.DeletionPolicy.KeepWhenStopped)



    # ------------------------------------------------------------------
    # Stretch control
    # ------------------------------------------------------------------

    def _apply_prebuilt_lut(self, lut: np.ndarray):
        """Apply a pre-built 65536-entry uint8 LUT to the u16 data."""
        if self._stretch_u16 is None:
            return
        t0     = time.perf_counter()
        result = lut[self._stretch_u16]   # (H,W) or (H,W,3) uint8 — vectorised
        t_qi   = time.perf_counter()
        qi     = (_gray_to_qimage(result) if result.ndim == 2
                  else _rgb_to_qimage(result))
        t_px   = time.perf_counter()
        self._item.setPixmap(QPixmap.fromImage(qi))
        _log.debug("_apply_prebuilt_lut: index %.1f ms  qimage %.1f ms  setPixmap %.1f ms  total %.1f ms",
                   (t_qi - t0) * 1000, (t_px - t_qi) * 1000,
                   (time.perf_counter() - t_px) * 1000,
                   (time.perf_counter() - t0) * 1000)

    def _apply_lut(self, vmin: float, gamma: float, vmax: float):
        """Build a 65 536-entry LUT and apply it to the pre-quantised u16 data.

        LUT computation is ~65 k ops (negligible); application is a single
        numpy fancy-index op (vectorised memcopy) — fast regardless of image size.
        """
        if self._stretch_u16 is None:
            return
        lo, hi = self._stretch_lo, self._stretch_hi

        # Map vmin/vmax from data space → u16 space [0, 65535]
        scale = hi - lo
        vmin_u = (vmin - lo) / scale * 65535
        vmax_u = (vmax - lo) / scale * 65535
        span   = vmax_u - vmin_u

        idx = np.arange(65536, dtype=np.float64)
        t   = np.clip((idx - vmin_u) / span if span > 0 else idx * 0.0, 0.0, 1.0)
        if abs(gamma - 0.5) > 1e-4:
            g   = float(np.clip(gamma, 1e-6, 1.0 - 1e-6))
            exp = np.log(0.5) / np.log(g)
            t   = np.power(np.maximum(t, 0.0), exp)
        lut = (t * 255.0).astype(np.uint8)
        self._apply_prebuilt_lut(lut)

    def _on_stretch_changed(self, vmin: float, gamma: float, vmax: float):
        self._vmin, self._gamma, self._vmax = vmin, gamma, vmax
        self._apply_lut(vmin, gamma, vmax)

    def auto_stretch(self):
        """Reset to 1–99 percentile auto-stretch (Cmd+1)."""
        if self._stretch_u16 is None:
            return
        self._vmin  = self._auto_vmin
        self._vmax  = self._auto_vmax
        self._gamma = 0.5
        self._apply_lut(self._vmin, self._gamma, self._vmax)
        self._histogram.set_data(self._raw_flat, self._vmin, self._vmax, 0.5)

    def apply_asinh_stretch(self, stretch_factor: float = 5.0):
        """Asinh stretch using current black/white points (Cmd+2)."""
        if self._stretch_u16 is None:
            return
        lut = _compute_asinh_lut(self._stretch_lo, self._stretch_hi,
                                   self._vmin, self._vmax, stretch_factor)
        self._apply_prebuilt_lut(lut)
        self._histogram.set_data(self._raw_flat, self._vmin, self._vmax, self._gamma)

    def apply_zscale_stretch(self):
        """ZScale stretch using astropy ZScaleInterval (Cmd+3)."""
        if self._stretch_u16 is None or self._raw_flat is None:
            return
        lut, z1, z2 = _compute_zscale_lut(self._stretch_lo, self._stretch_hi,
                                            self._raw_flat)
        self._vmin, self._vmax, self._gamma = z1, z2, 0.5
        self._apply_prebuilt_lut(lut)
        self._histogram.set_data(self._raw_flat, self._vmin, self._vmax, self._gamma)

    def apply_mtf_stretch(self):
        """MTF auto-stretch (PixInsight-style) (Cmd+4).

        Grayscale: fast LUT path (statistics in u16 space).
        Colour:    per-channel unlinked background neutralisation +
                   luminance-preserving rational MTF reconstruction.
        """
        if self._stretch_u16 is None:
            return
        if self._stretch_u16.ndim == 2:
            # Grayscale — LUT-based
            lut = _compute_mtf_lut(self._stretch_u16.ravel())
            self._apply_prebuilt_lut(lut)
        else:
            # Colour — per-channel unlinked, luma-preserving
            result = _apply_mtf_color(self._stretch_u16)
            qi = _rgb_to_qimage(result)
            self._item.setPixmap(QPixmap.fromImage(qi))
        if self._raw_flat is not None:
            self._histogram.set_data(self._raw_flat, self._vmin, self._vmax, self._gamma)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        
        scene_rect = self.scene().itemsBoundingRect()
        if scene_rect.width() > 0 and scene_rect.height() > 0:
            vp_rect = self.viewport().rect()
            min_scale = min(vp_rect.width() / scene_rect.width(),
                            vp_rect.height() / scene_rect.height())
            
            current_scale = self.transform().m11()
            new_scale = current_scale * factor
            
            if factor < 1.0 and new_scale <= min_scale:
                # Clamp to exact fit
                self._fit()
                return
                
        self._is_fitted = False
        self.scale(factor, factor)

    # ------------------------------------------------------------------
    # Mouse Interaction (Magnifier)
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        
        if self._stretch_u16 is None or self._item is None:
            return
            
        vp_pos = event.position().toPoint()
        scene_pos = self.mapToScene(vp_pos)
        
        x, y = int(scene_pos.x()), int(scene_pos.y())
        h, w = self._stretch_u16.shape[:2]
        
        if 0 <= x < w and 0 <= y < h:
            self.setCursor(Qt.CursorShape.CrossCursor)
            
            # Reconstruct the absolute raw FITS values by reversing the 16-bit quantization curve
            _lo, _hi = self._stretch_lo, self._stretch_hi
            
            if self._stretch_u16.ndim == 2:
                uval = self._stretch_u16[y, x]
                raw_val = (uval / 65535.0) * (_hi - _lo) + _lo
                text = f"Val: {raw_val:.1f}" if _hi < 100 else f"Val: {int(raw_val)}"
            else:
                ur, ug, ub = self._stretch_u16[y, x]
                rr = (ur / 65535.0) * (_hi - _lo) + _lo
                rg = (ug / 65535.0) * (_hi - _lo) + _lo
                rb = (ub / 65535.0) * (_hi - _lo) + _lo
                if _hi < 100:
                    text = f"R:{rr:.1f} G:{rg:.1f} B:{rb:.1f}"
                else:
                    text = f"R:{int(rr)} G:{int(rg)} B:{int(rb)}"
                
            # Crop tiny 15x15 region (viewport handles bounding limits safely)
            crop_size = 15
            rx = max(0, x - crop_size // 2)
            ry = max(0, y - crop_size // 2)
            rw = min(w - rx, crop_size)
            rh = min(h - ry, crop_size)
            
            crop = self._item.pixmap().copy(rx, ry, rw, rh)
            self._magnifier.set_data(crop, text)
            
            # Position logic (dodge the mouse slightly)
            mag_w = self._magnifier.width()
            mag_h = self._magnifier.height()
            vx = vp_pos.x() + 16
            vy = vp_pos.y() + 16
            
            if vx + mag_w > self.viewport().width():
                vx = vp_pos.x() - mag_w - 16
            if vy + mag_h > self.viewport().height():
                vy = vp_pos.y() - mag_h - 16
                
            self._magnifier.move(vx, vy)
            self._magnifier.show()
            self._magnifier.raise_()
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._magnifier.hide()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if hasattr(self, '_magnifier'):
            self._magnifier.hide()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    # Emitted from worker thread; queued connection ensures delivery on main thread.
    _stretch_ready = Signal(int, str, object)
    _buffer_gauge_trigger = Signal()

    def __init__(self, fits_path: str | None = None):
        super().__init__()
        self.setWindowTitle("FITS Preview")
        self.setGeometry(QApplication.primaryScreen().availableGeometry())
        
        self.main_container = MainContainer()
        self.setCentralWidget(self.main_container)
        
        self._fits_files: list[str] = []
        self._bad_files: set[str] = set()
        self._fwhm_db: dict[str, float] = {}
        self._current_path: str = ""
        self._current_index: int    = 0
        self._nav_direction: int    = 1
        self._current_ahead: int    = 0
        self._current_behind: int   = 0
        self._stretch_generation: int = 0   # incremented on each navigate; stale
                                            # async results are discarded

        # Two-stage pipeline: one I/O worker serialises all disk reads;
        # multiple compute workers handle debayer + quantize + MTF in parallel.
        # _preload_futures stores pipeline Future objects (see _start_pipeline).
        self._preload_futures: dict[str, Future] = {}
        self._io_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fits-io")
        self._compute_executor = ThreadPoolExecutor(
            max_workers=_COMPUTE_WORKERS, thread_name_prefix="fits-compute")

        # Dispatch a tiny warmup matrix to force Numba JIT compilation/deserialization 
        # in the background before the first real megapixel image arrives.
        self._compute_executor.submit(
            compute_backend.apply_mtf_color,
            np.zeros((1, 1, 3), dtype=np.uint16),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            0.5
        )

        # Stretch data cache (u16 + MTF QImage) for both current and preloaded files.
        self._stretch_cache: OrderedDict[str, _StretchData] = OrderedDict()
        self._stretch_pending: int = 0   # in-flight current-file pipelines (approx)
        self._stretch_ready.connect(self._deliver_stretch)
        self._buffer_gauge_trigger.connect(self._update_buffer_gauges)

        # Saved viewport for the in-flight stretch miss; consumed by _deliver_stretch.
        self._pending_saved_viewport = None

        # Window-level shortcuts (fire even when FitsView has focus)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._navigate(+1))
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, lambda: self._navigate(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Down),  self, lambda: self._navigate(-10))
        QShortcut(QKeySequence(Qt.Key.Key_Up),    self, lambda: self._navigate(+10))
        QShortcut(QKeySequence(Qt.Key.Key_Home),  self, lambda: self._navigate(-9999999))
        QShortcut(QKeySequence(Qt.Key.Key_End),   self, lambda: self._navigate(+9999999))
        QShortcut(QKeySequence("Ctrl+1"),          self, self._mtf_stretch)
        QShortcut(QKeySequence("Ctrl+2"),          self, self._asinh_stretch)
        QShortcut(QKeySequence("Ctrl+3"),          self, self._zscale_stretch)
        QShortcut(QKeySequence("Ctrl+4"),          self, self._auto_stretch)
        QShortcut(QKeySequence(Qt.Key.Key_Space),  self, self._toggle_bad)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self._commit_bad_files)
        QShortcut(QKeySequence(Qt.Key.Key_Enter),  self, self._commit_bad_files)
        QShortcut(QKeySequence(Qt.Key.Key_I),      self, self.main_container.toggle_header_state)
        QShortcut(QKeySequence(Qt.Key.Key_H),      self, self.main_container.toggle_histogram)

        if fits_path:
            self._load_fits(fits_path)
        else:
            self._show_placeholder()

    # ------------------------------------------------------------------
    # Queue diagnostics
    # ------------------------------------------------------------------

    def _queue_info(self) -> str:
        return f"preload:{len(self._preload_futures)} stretch:{self._stretch_pending}"

    # ------------------------------------------------------------------
    # Two-stage I/O → compute pipeline
    # ------------------------------------------------------------------

    def _start_pipeline(self, path: str) -> Future:
        """Submit path through the single I/O worker then the compute pool.

        Returns a Future that resolves to _StretchData | None once both
        stages complete.  Call .cancel() to abandon; in-progress I/O cannot
        be interrupted but the compute stage will not start if the future
        is already cancelled when I/O finishes.
        """
        result_future: Future = Future()
        name = os.path.basename(path)

        def _on_io_done(io_future):
            if result_future.cancelled():
                return
            try:
                raw_data, raw_header = io_future.result()
            except Exception as exc:
                _log.exception("[io] failed: %s", name)
                try:
                    result_future.set_exception(exc)
                except Exception:
                    pass
                return
            if raw_data is None:
                try:
                    result_future.set_result(None)
                except Exception:
                    pass
                return
            _log.debug("[io] done → compute  %s", name)
            try:
                compute_future = self._compute_executor.submit(
                    _compute_from_raw, path, raw_data, raw_header)
            except RuntimeError:
                # Executor already shut down (app quitting); discard quietly.
                return
            compute_future.add_done_callback(_on_compute_done)

        def _on_compute_done(compute_future):
            if result_future.cancelled():
                return
            try:
                result_future.set_result(compute_future.result())
            except Exception as exc:
                try:
                    result_future.set_exception(exc)
                except Exception:
                    pass

        _log.debug("[io] queued  [%s]  %s", self._queue_info(), name)
        io_future = self._io_executor.submit(_load_path_raw_data, path)
        io_future.add_done_callback(_on_io_done)
        return result_future

    # ------------------------------------------------------------------
    # Stretch data cache helpers (main-thread only)
    # ------------------------------------------------------------------

    def _stretch_cache_put(self, path: str, sd: _StretchData):
        self._stretch_cache[path] = sd
        self._stretch_cache.move_to_end(path)
        while len(self._stretch_cache) > _config.config["cache_max"]:
            self._stretch_cache.popitem(last=False)

    def _stretch_cache_get(self, path: str) -> _StretchData | None:
        # Harvest any completed preload futures first.
        future = self._preload_futures.get(path)
        if future is not None and future.done():
            self._preload_futures.pop(path)
            try:
                r = future.result()
                if r is not None:
                    self._stretch_cache_put(path, r)
            except Exception:
                pass
        if path in self._stretch_cache:
            self._stretch_cache.move_to_end(path)
            return self._stretch_cache[path]
        return None

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def _trigger_preload(self, ahead: int, behind: int):
        self._current_ahead = ahead
        self._current_behind = behind
        if not self._fits_files:
            return
        n = len(self._fits_files)
        d = self._nav_direction
        wanted: set[str] = set()
        for i in range(1, ahead + 1):
            wanted.add(self._fits_files[(self._current_index + d * i) % n])
        for i in range(1, behind + 1):
            wanted.add(self._fits_files[(self._current_index - d * i) % n])
        for path in list(self._preload_futures):
            if path not in wanted:
                self._preload_futures.pop(path).cancel()
        for path in wanted:
            if path not in self._stretch_cache and path not in self._preload_futures:
                future = self._start_pipeline(path)
                future.add_done_callback(lambda f: self._buffer_gauge_trigger.emit())
                self._preload_futures[path] = future



    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _show_placeholder(self):
        label = QLabel("Open a FITS file to get started.\n\nUsage: fits-preview <file.fits>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_container.set_view(label)

    def _load_fits(self, path: str, saved_viewport=None):
        t0       = time.perf_counter()
        abs_path = os.path.abspath(path)
        name     = os.path.basename(abs_path)
        self.setWindowTitle(f"FITS Preview — {name}")
        self._current_path = abs_path
        self._stretch_generation += 1
        gen = self._stretch_generation
        try:
            # ── Step 1: Check stretch cache (includes harvesting preloads) ─
            stretch = self._stretch_cache_get(abs_path)
            _log.info("_load_fits: stretch %s  [%s]  %s",
                       "HIT" if stretch else "MISS (async)", self._queue_info(), name)

            if stretch is not None:
                # ── HIT: build FitsView with pre-rendered MTF image ───────
                pixmap = QPixmap.fromImage(stretch.mtf_qimage)
                vp_to_restore = None
                if saved_viewport is not None:
                    old_size = saved_viewport[-1]
                    if old_size == (pixmap.width(), pixmap.height()):
                        vp_to_restore = saved_viewport
                        _log.debug("_load_fits (HIT): matched old size %s, propagating vp_to_restore", old_size)
                    else:
                        _log.debug("_load_fits (HIT): size mismatch, discarding saved_viewport")
                is_bad = abs_path in self._bad_files
                view = FitsView(stretch.mtf_qimage, stretch,
                                saved_viewport=vp_to_restore, is_bad=is_bad)
                self.main_container.set_view(view, stretch.raw_header)
                if stretch.fwhm > 0.0:
                    self._fwhm_db[abs_path] = stretch.fwhm
                if self.main_container is not None:
                    self.main_container.update_fwhm_db(self._fwhm_db, self._bad_files, self._current_path)
                    
                _log.debug("_load_fits: displayed in %.1f ms  [%s]  %s",
                           (time.perf_counter() - t0) * 1000, self._queue_info(), name)
            else:
                # ── MISS: keep current view visible; submit stretch worker ─
                # Store the saved viewport so _deliver_stretch can restore it.
                self._pending_saved_viewport = saved_viewport
                if not isinstance(self.main_container.get_view(), FitsView):
                    self.main_container.set_view(LoadingWidget())

                def _on_pipeline_done(future):
                    self._stretch_pending -= 1
                    try:
                        sd = future.result()
                    except Exception:
                        _log.exception("pipeline failed: %s", name)
                        return
                    if sd is not None:
                        self._stretch_ready.emit(gen, abs_path, sd)

                self._stretch_pending += 1
                self._start_pipeline(abs_path).add_done_callback(_on_pipeline_done)

            # ── Step 2: Directory index + preload ─────────────────────────
            self._fits_files = _fits_siblings(abs_path)
            try:
                self._current_index = self._fits_files.index(abs_path)
            except ValueError:
                self._current_index = 0

            self._trigger_preload(
                _config.config["preload_ahead"],
                _config.config["preload_behind"]
            )
            
            # Immediately update buffer gauges to reflect the loaded view's buffer state
            self._update_buffer_gauges()

            _log.debug("_load_fits: TOTAL %.1f ms  [%s]  %s",
                       (time.perf_counter() - t0) * 1000, self._queue_info(), name)

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.main_container.set_view(QLabel(f"Error loading file:\n{e}"))

    def _deliver_stretch(self, gen: int, path: str, sd: _StretchData):
        """Called on the main thread when async stretch build completes."""
        if gen != self._stretch_generation:
            _log.debug("_deliver_stretch: stale gen %d (current %d), discarding [%s]  %s",
                       gen, self._stretch_generation, self._queue_info(),
                       os.path.basename(path))
            return
        self._stretch_cache_put(path, sd)

        # Resolve the saved viewport now that we know the image dimensions.
        saved_vp = self._pending_saved_viewport
        self._pending_saved_viewport = None
        vp_to_restore = None
        if saved_vp is not None:
            old_size = saved_vp[-1]
            if old_size == (sd.mtf_qimage.width(), sd.mtf_qimage.height()):
                vp_to_restore = saved_vp
                _log.debug("_deliver_stretch: matched old size %s, propagating vp_to_restore", old_size)
            else:
                _log.debug("_deliver_stretch: size mismatch, discarding saved_viewport")

        _log.debug("_deliver_stretch: installing [%s]  %s",
                   self._queue_info(), os.path.basename(path))
        is_bad = path in self._bad_files
        view = FitsView(sd.mtf_qimage, sd, saved_viewport=vp_to_restore, is_bad=is_bad)
        self.main_container.set_view(view, sd.raw_header)
        
        if sd.fwhm > 0.0:
            self._fwhm_db[path] = sd.fwhm
        if self.main_container is not None:
            self.main_container.update_fwhm_db(self._fwhm_db, self._bad_files, self._current_path)
            
        self._update_buffer_gauges()

    def _update_buffer_gauges(self):
        if not self._fits_files:
            if self.main_container is not None:
                self.main_container.update_gauges(1.0, 1.0)
            return
            
        def is_buffered(path: str) -> bool:
            if path in self._stretch_cache:
                return True
            f = self._preload_futures.get(path)
            if f is None or not f.done() or f.cancelled():
                return False
            return f.exception() is None
            
        idx = self._current_index
        n = len(self._fits_files)
        
        ahead = self._current_ahead
        behind = self._current_behind
        d = self._nav_direction
        
        back_count = 0
        for i in range(1, behind + 1):
            if is_buffered(self._fits_files[(idx - d * i) % n]):
                back_count += 1
                
        fwd_count = 0
        for i in range(1, ahead + 1):
            if is_buffered(self._fits_files[(idx + d * i) % n]):
                fwd_count += 1
                
        back_pct = back_count / behind if behind > 0 else 1.0
        fwd_pct = fwd_count / ahead if ahead > 0 else 1.0
        _log.info("_update_buffer_gauges: back %d/%d (%.2f)  fwd %d/%d (%.2f)",
                  back_count, behind, back_pct, fwd_count, ahead, fwd_pct)
        if self.main_container is not None:
            self.main_container.update_gauges(back_pct, fwd_pct)

    def _navigate(self, delta: int):
        if not self._fits_files:
            return
            
        new_idx = self._current_index + delta
        if new_idx < 0:
            if self._current_index == 0:
                view = self.main_container.get_view()
                if isinstance(view, FitsView): view.show_eol()
                return
            new_idx = 0
        elif new_idx >= len(self._fits_files):
            if self._current_index == len(self._fits_files) - 1:
                view = self.main_container.get_view()
                if isinstance(view, FitsView): view.show_eol()
                return
            new_idx = len(self._fits_files) - 1
            
        _log.info("key: %s  [%s]", "→" if delta > 0 else "←", self._queue_info())
        t0 = time.perf_counter()
        self._nav_direction = 1 if delta > 0 else -1
        saved_vp = None
        view = self.main_container.get_view()
        if isinstance(view, FitsView):
            saved_vp = view.viewport_state()
            _log.debug("_navigate: captured saved_vp %s", bool(saved_vp))
        self._current_index = new_idx
        self._load_fits(self._fits_files[self._current_index], saved_viewport=saved_vp)
        _log.debug("_navigate: TOTAL %.1f ms  [%s]", (time.perf_counter() - t0) * 1000,
                   self._queue_info())

    def _auto_stretch(self):
        _log.info("key: Cmd+4 auto-stretch  [%s]", self._queue_info())
        view = self.main_container.get_view()
        if isinstance(view, FitsView):
            view.auto_stretch()

    def _asinh_stretch(self):
        _log.info("key: Cmd+2 asinh-stretch  [%s]", self._queue_info())
        view = self.main_container.get_view()
        if isinstance(view, FitsView):
            view.apply_asinh_stretch()

    def _zscale_stretch(self):
        _log.info("key: Cmd+3 zscale-stretch  [%s]", self._queue_info())
        view = self.main_container.get_view()
        if isinstance(view, FitsView):
            view.apply_zscale_stretch()

    def _mtf_stretch(self):
        _log.info("key: Cmd+1 mtf-stretch  [%s]", self._queue_info())
        view = self.main_container.get_view()
        if isinstance(view, FitsView):
            view.apply_mtf_stretch()

    def _toggle_bad(self):
        if not self._fits_files: return
        path = self._fits_files[self._current_index]
        view = self.main_container.get_view()
        if path in self._bad_files:
            self._bad_files.remove(path)
            if isinstance(view, FitsView): view.set_bad(False)
            _log.info("key: Space (unmarked bad)  %s", os.path.basename(path))
        else:
            self._bad_files.add(path)
            if isinstance(view, FitsView): view.set_bad(True)
            _log.info("key: Space (marked bad)  %s", os.path.basename(path))

    def _commit_bad_files(self):
        if not self._bad_files: return
        
        current_path = self._fits_files[self._current_index]
        parent_dir = os.path.dirname(current_path)
        bad_dir = os.path.join(parent_dir, "bad")
        os.makedirs(bad_dir, exist_ok=True)
        
        count = len(self._bad_files)
        _log.info("key: Enter (moving %d bad files to %s)", count, bad_dir)
        
        for bad_file in list(self._bad_files):
            try:
                shutil.move(bad_file, os.path.join(bad_dir, os.path.basename(bad_file)))
            except Exception as e:
                _log.error("Failed to move %s: %s", bad_file, e)
                
        self._bad_files.clear()
        
        # Re-index
        self._fits_files = _fits_siblings(current_path)
        if not self._fits_files:
            self._current_index = 0
            self.setWindowTitle("FITS Preview — Empty")
            
            label = QLabel("u got no imgs bro 🤷‍♂️")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = label.font()
            font.setPixelSize(36)
            label.setFont(font)
            self.main_container.set_view(label, None)
            return
            
        if self._current_index >= len(self._fits_files):
            self._current_index = len(self._fits_files) - 1
            
        self._load_fits(self._fits_files[self._current_index])

    def _shutdown(self):
        """Cancel pending work and stop executor threads immediately on quit."""
        _log.debug("shutdown: cancelling %d preload futures", len(self._preload_futures))
        for f in self._preload_futures.values():
            f.cancel()
        self._preload_futures.clear()
        self._io_executor.shutdown(wait=False, cancel_futures=True)
        self._compute_executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class FitsApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self._pending_file: str | None = None
        self._main_window: MainWindow | None = None

    def set_main_window(self, window: MainWindow):
        self._main_window = window
        if self._pending_file:
            self._main_window._load_fits(self._pending_file)
            self._pending_file = None

    def event(self, event: QEvent) -> bool:
        if isinstance(event, QFileOpenEvent):
            path = event.file()
            if self._main_window:
                self._main_window._load_fits(path)
            else:
                self._pending_file = path
            return True
        return super().event(event)


def _install_stderr_excepthook():
    if sys.stderr is None or not sys.stderr.isatty():
        return
    def hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.excepthook = hook


def run_headless(fits_path: str, out_path: str, stretch_type: str):
    """Executes the pipeline without spawning a PySide6 GUI and saves the output."""
    _log.info(f"Running in headless mode. Input: {fits_path}")
    raw_data, raw_header = _load_path_raw_data(fits_path)
    if raw_data is None:
        _log.error("Failed to load FITS data.")
        sys.exit(1)
        
    sd = _compute_from_raw(fits_path, raw_data, raw_header, headless=True)
    if sd is None:
        _log.error("Compute pipeline failed.")
        sys.exit(1)
        
    rgb = sd.mtf_rgb
    
    if stretch_type != "mtf":
        if stretch_type == "auto":
            sub = _subsample(sd.raw_flat)
            vmin = float(np.nanpercentile(sub, 1))
            vmax = float(np.nanpercentile(sub, 99))
            scale = sd.hi - sd.lo
            if scale == 0: scale = 1.0
            vmin_u = (vmin - sd.lo) / scale * 65535
            vmax_u = (vmax - sd.lo) / scale * 65535
            span = vmax_u - vmin_u
            idx = np.arange(65536, dtype=np.float64)
            t = np.clip((idx - vmin_u) / span if span > 0 else idx * 0.0, 0.0, 1.0)
            lut = (t * 255.0).astype(np.uint8)
            rgb = lut[sd.u16]
        elif stretch_type == "zscale":
            lut, _, _ = _compute_zscale_lut(sd.lo, sd.hi, sd.raw_flat)
            rgb = lut[sd.u16]
        elif stretch_type == "asinh":
            sub = _subsample(sd.raw_flat)
            vmin = float(np.nanpercentile(sub, 1))
            vmax = float(np.nanpercentile(sub, 99))
            lut = _compute_asinh_lut(sd.lo, sd.hi, vmin, vmax, 5.0)
            rgb = lut[sd.u16]
            
    if rgb.ndim == 3:
        bgr = rgb[:, :, ::-1]  # OpenCV expects BGR
    else:
        bgr = rgb
        
    import cv2
    cv2.imwrite(out_path, bgr)
    _log.info(f"Saved {stretch_type} stretch result to {out_path}")


def main():
    _install_stderr_excepthook()
    
    parser = argparse.ArgumentParser(description="FITS Image Preview and Stretch Tool")
    parser.add_argument("file", nargs="?", help="Path to the FITS file to load")
    parser.add_argument("--headless", action="store_true", help="Run without UI, saving the result to disk")
    parser.add_argument("--out", type=str, help="Output path for the stretched image (required if --headless)")
    parser.add_argument("--stretch", choices=["mtf", "auto", "zscale", "asinh"], default="mtf", 
                        help="Stretch algorithm to use (default: mtf)")
    parser.add_argument("--ahead", type=int, help="Number of images to preload ahead of the current view")
    parser.add_argument("--behind", type=int, help="Number of images to keep preloaded behind the current view")
    parser.add_argument("--cache", type=int, help="Maximum number of fully stretched images to keep in RAM")
    parser.add_argument("--log", action="store_true", help="Log debug output to /tmp/py-fits-preview.log")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to console")
    args = parser.parse_args()
    
    if args.ahead is not None: _config.config["preload_ahead"] = args.ahead
    if args.behind is not None: _config.config["preload_behind"] = args.behind
    if args.cache is not None: _config.config["cache_max"] = args.cache

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    
    if args.log:
        file_handler = logging.FileHandler("/tmp/py-fits-preview.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(file_handler)
    
    if args.headless:
        if not args.file:
            parser.error("A FITS file is required in headless mode.")
        if not args.out:
            parser.error("--out argument is required in headless mode.")
        run_headless(args.file, args.out, args.stretch)
        return

    app = FitsApp([sys.argv[0]])
    window = MainWindow(args.file)
    app.aboutToQuit.connect(window._shutdown)
    app.set_main_window(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
