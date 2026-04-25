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
    QVBoxLayout, QProgressBar,
)
from PySide6.QtGui import (
    QImage, QPixmap, QFileOpenEvent, QKeySequence, QShortcut,
    QPainter, QColor, QPen, QBrush, QPolygon,
)
from PySide6.QtCore import Qt, QEvent, QTimer, Signal, QPoint


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
            data, header = hdu.data.copy(), hdu.header
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
            r = float(header[r_key])
            g = float(header[g_key])
            b = float(header[b_key])
            if g > 0:
                return r / g, 1.0, b / g      # normalise to green
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
    sub = _subsample(d.ravel())
    lo = float(np.nanpercentile(sub, 0.01))
    hi = float(np.nanpercentile(sub, 99.99))
    if hi <= lo:
        hi = lo + 1.0

    u16 = np.clip((d - lo) / (hi - lo) * 65535, 0, 65535).astype(np.uint16)
    result = np.ascontiguousarray(np.flipud(u16))
    _log.debug("_build_stretch_data: quantize %.1f ms  total %.1f ms  shape=%s",
               (time.perf_counter() - t_q) * 1000,
               (time.perf_counter() - t0) * 1000, result.shape)
    return result, lo, hi


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
    rng = np.random.default_rng(0)
    
    if N > _SUBSAMPLE_N:
        idx = rng.choice(N, size=_SUBSAMPLE_N, replace=False)
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

_PRELOAD_HIT       = (3, 2)
_PRELOAD_MISS      = (4, 2)
_COMPUTE_WORKERS   = 4          # CPU workers for debayer + quantize + MTF
_STRETCH_CACHE_MAX = 6          # u16 arrays ~121 MB each → ~726 MB max
_SUBSAMPLE_N       = 500_000    # pixels used for median/MAD/histogram stats


def _subsample(arr: np.ndarray, n: int = _SUBSAMPLE_N) -> np.ndarray:
    """Return a flat random subsample of arr, at most n elements.

    Used for all statistical operations (median, MAD, percentile, histogram)
    where the full 60M-element array is overkill.  500 K pixels gives
    statistically equivalent results for sky-background estimation.
    """
    flat = arr.ravel()
    if flat.size <= n:
        return flat
    rng = np.random.default_rng(0)   # fixed seed → reproducible stretch
    return rng.choice(flat, size=n, replace=False)

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
    """Semi-transparent histogram with draggable black / mid / white markers."""

    stretch_changed = Signal(float, float, float)   # vmin, gamma, vmax

    W, H      = 300, 92
    HIST_TOP  = 4
    HIST_BOT  = 64       # bottom of histogram bar area
    SEP_Y     = 66       # separator line y
    TRI_TOP   = SEP_Y    # triangle apex y (touching separator)
    TRI_BOT   = SEP_Y + 20   # triangle base y
    TRI_HW    = 7        # triangle half-width
    MX        = 10       # left/right content margin
    HIT_R     = 10       # click-detection radius (px)

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
        self.setMouseTracking(True)

        self._log_counts: np.ndarray | None = None
        self._data_lo = 0.0
        self._data_hi = 1.0
        self._vmin  = 0.0
        self._vmax  = 1.0
        self._gamma = 0.5
        self._drag: str | None = None   # 'black' | 'mid' | 'white'

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

    def _x_to_data(self, x: float) -> float:
        t = (x - self.MX) / self._content_w()
        return self._data_lo + t * (self._data_hi - self._data_lo)

    def _gamma_x(self) -> float:
        xb = self._data_to_x(self._vmin)
        xw = self._data_to_x(self._vmax)
        return xb + self._gamma * (xw - xb)

    def _x_to_gamma(self, x: float) -> float:
        xb = self._data_to_x(self._vmin)
        xw = self._data_to_x(self._vmax)
        if xw <= xb:
            return 0.5
        return float(np.clip((x - xb) / (xw - xb), 0.01, 0.99))

    def _nearest_marker(self, x: float) -> str | None:
        candidates = [
            ("black", self._data_to_x(self._vmin)),
            ("mid",   self._gamma_x()),
            ("white", self._data_to_x(self._vmax)),
        ]
        candidates.sort(key=lambda t: abs(x - t[1]))
        name, mx = candidates[0]
        return name if abs(x - mx) <= self.HIT_R else None

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

        # Histogram bars
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

        # Separator
        p.setPen(QPen(QColor(75, 75, 75), 1))
        p.drawLine(self.MX, self.SEP_Y, self.W - self.MX, self.SEP_Y)

        # Markers
        marker_xs = {
            "black": self._data_to_x(self._vmin),
            "mid":   self._gamma_x(),
            "white": self._data_to_x(self._vmax),
        }
        for name, mx in marker_xs.items():
            col = self._COLORS[name]
            ix  = int(round(mx))

            # Dashed vertical line through histogram
            pen = QPen(QColor(col.red(), col.green(), col.blue(), 140), 1,
                       Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(ix, self.HIST_TOP, ix, self.HIST_BOT)

            # Upward-pointing triangle (apex at SEP_Y, flat base lower)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            tri = QPolygon([
                QPoint(ix,                self.TRI_TOP),
                QPoint(ix - self.TRI_HW, self.TRI_BOT),
                QPoint(ix + self.TRI_HW, self.TRI_BOT),
            ])
            p.drawPolygon(tri)

        p.end()

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = self._nearest_marker(event.position().x())

    def mouseMoveEvent(self, event):
        x = event.position().x()
        if self._drag is None:
            m = self._nearest_marker(x)
            self.setCursor(Qt.CursorShape.SizeHorCursor
                           if m else Qt.CursorShape.ArrowCursor)
            return

        eps = (self._data_hi - self._data_lo) * 1e-4
        val = float(np.clip(self._x_to_data(x), self._data_lo, self._data_hi))

        if self._drag == "black":
            self._vmin = min(val, self._vmax - eps)
        elif self._drag == "white":
            self._vmax = max(val, self._vmin + eps)
        elif self._drag == "mid":
            self._gamma = self._x_to_gamma(x)

        self.update()
        self.stretch_changed.emit(self._vmin, self._gamma, self._vmax)

    def mouseReleaseEvent(self, _event):
        self._drag = None


# ---------------------------------------------------------------------------
# Stretch data container
# ---------------------------------------------------------------------------

class _StretchData:
    """Holds the pre-quantised u16 array and associated metadata needed for
    interactive stretch.  Kept separate from QImage so the two can be cached
    and delivered independently (QImage for immediate display, u16 for controls).
    """
    __slots__ = ("u16", "lo", "hi", "raw_flat", "mtf_qimage", "mtf_rgb")

    def __init__(self, u16: np.ndarray, lo: float, hi: float,
                 raw_flat: np.ndarray, mtf_qimage: "QImage | None" = None,
                 mtf_rgb: np.ndarray | None = None):
        self.u16        = u16         # (H, W) or (H, W, 3) uint16, flipped
        self.lo         = lo          # data value → u16 == 0
        self.hi         = hi          # data value → u16 == 65535
        self.raw_flat   = raw_flat    # float32 flat of pre-debayer data for histogram
        self.mtf_qimage = mtf_qimage  # pre-rendered MTF QImage for instant cache-hit display
        self.mtf_rgb    = mtf_rgb     # raw uint8 array for headless mode


def _compute_from_raw(path: str, raw_data: np.ndarray,
                      raw_header, headless: bool = False) -> "_StretchData | None":
    """Compute stage: debayer + quantize + MTF render.  Runs in compute pool.

    Receives raw FITS data already loaded from disk by the I/O worker.
    Returns None on any error.
    """
    name = os.path.basename(path)
    t0 = time.perf_counter()
    u16, lo, hi = _build_stretch_data(raw_data, raw_header)
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
    return _StretchData(u16, lo, hi, raw_flat, mtf_qi, mtf_rgb)

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

class FitsView(QGraphicsView):
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
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._pixmap_size = (pixmap.width(), pixmap.height())
        self._saved_viewport = saved_viewport
        self._fit_pending    = False
        self._initial_fit_done = saved_viewport is not None

        self._stretch_u16  = None
        self._stretch_lo   = 0.0
        self._stretch_hi   = 1.0
        self._raw_flat     = None
        self._vmin         = 0.0
        self._vmax         = 1.0
        self._auto_vmin    = 0.0
        self._auto_vmax    = 1.0
        self._gamma        = 0.5

        # Histogram overlay — child of the viewport so it stays fixed on screen
        self._histogram = HistogramOverlay(self.viewport())
        self._histogram.stretch_changed.connect(self._on_stretch_changed)
        
        self._bad_label = QLabel("🤮", self.viewport())
        self._bad_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self._bad_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._bad_label.setStyleSheet("color: white; background: transparent;")
        self._bad_label.setVisible(is_bad)

        if stretch is not None:
            self._install_stretch(stretch)

        # Shortcuts (Ctrl→Cmd on macOS)
        QShortcut(QKeySequence("Ctrl+0"), self, self._zoom_one_to_one)
        QShortcut(QKeySequence("Ctrl+9"), self, self._fit)

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
        t_hist = time.perf_counter()
        self._histogram.set_data(self._raw_flat, self._vmin, self._vmax, 0.5)
        _log.debug("FitsView._install_stretch: histogram %.1f ms",
                   (time.perf_counter() - t_hist) * 1000)
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
            return None
        if self._saved_viewport is not None:
            t, c = self._saved_viewport
            return (t, c, self._pixmap_size)
        center = self.mapToScene(self.viewport().rect().center())
        return (self.transform(), center, self._pixmap_size)

    def _apply_pending_restore(self):
        if self._saved_viewport is not None:
            t, c = self._saved_viewport
            self._saved_viewport = None
            self.setTransform(t)
            self.centerOn(c)

    def _fit(self):
        self._fit_pending = False
        self.fitInView(self.scene().itemsBoundingRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def _zoom_one_to_one(self):
        self._fit_pending = False
        self.resetTransform()

    def showEvent(self, event):
        super().showEvent(event)
        if self._saved_viewport is not None:
            QTimer.singleShot(0, self._apply_pending_restore)
        elif not self._initial_fit_done:
            self._initial_fit_done = True
            self._fit_pending = True
            QTimer.singleShot(0, self._fit)
        self._position_overlays()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlays()

    # ------------------------------------------------------------------
    # Overlays position
    # ------------------------------------------------------------------

    def _position_overlays(self):
        vp = self.viewport()
        margin = 12
        
        h = self._histogram
        h.move(margin, vp.height() - h.height() - margin)
        h.raise_()
        
        font = self._bad_label.font()
        font.setPixelSize(max(24, int(vp.width() * 0.10)))
        self._bad_label.setFont(font)
        self._bad_label.adjustSize()
        self._bad_label.move(vp.width() - self._bad_label.width() - margin, margin)
        self._bad_label.raise_()
        
    def set_bad(self, is_bad: bool):
        self._bad_label.setVisible(is_bad)

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
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    # Emitted from worker thread; queued connection ensures delivery on main thread.
    _stretch_ready = Signal(int, str, object)

    def __init__(self, fits_path: str | None = None):
        super().__init__()
        self.setWindowTitle("FITS Preview")
        self.setGeometry(QApplication.primaryScreen().availableGeometry())
        self._fits_files: list[str] = []
        self._bad_files: set[str] = set()
        self._current_index: int    = 0
        self._nav_direction: int    = 1
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

        # Stretch data cache (u16 + MTF QImage) for both current and preloaded files.
        self._stretch_cache: OrderedDict[str, _StretchData] = OrderedDict()
        self._stretch_pending: int = 0   # in-flight current-file pipelines (approx)
        self._stretch_ready.connect(self._deliver_stretch)

        # Saved viewport for the in-flight stretch miss; consumed by _deliver_stretch.
        self._pending_saved_viewport = None

        # Window-level shortcuts (fire even when FitsView has focus)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._navigate(+1))
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, lambda: self._navigate(-1))
        QShortcut(QKeySequence("Ctrl+1"),          self, self._mtf_stretch)
        QShortcut(QKeySequence("Ctrl+2"),          self, self._asinh_stretch)
        QShortcut(QKeySequence("Ctrl+3"),          self, self._zscale_stretch)
        QShortcut(QKeySequence("Ctrl+4"),          self, self._auto_stretch)
        QShortcut(QKeySequence(Qt.Key.Key_Space),  self, self._toggle_bad)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self._commit_bad_files)
        QShortcut(QKeySequence(Qt.Key.Key_Enter),  self, self._commit_bad_files)

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
        while len(self._stretch_cache) > _STRETCH_CACHE_MAX:
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
                self._preload_futures[path] = self._start_pipeline(path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _show_placeholder(self):
        label = QLabel("Open a FITS file to get started.\n\nUsage: fits-preview <file.fits>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(label)

    def _load_fits(self, path: str, saved_viewport=None):
        t0       = time.perf_counter()
        abs_path = os.path.abspath(path)
        name     = os.path.basename(abs_path)
        self.setWindowTitle(f"FITS Preview — {name}")
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
                    _, _, old_size = saved_viewport
                    if old_size == (pixmap.width(), pixmap.height()):
                        vp_to_restore = (saved_viewport[0], saved_viewport[1])
                is_bad = abs_path in self._bad_files
                view = FitsView(stretch.mtf_qimage, stretch,
                                saved_viewport=vp_to_restore, is_bad=is_bad)
                self.setCentralWidget(view)
                _log.debug("_load_fits: displayed in %.1f ms  [%s]  %s",
                           (time.perf_counter() - t0) * 1000, self._queue_info(), name)
            else:
                # ── MISS: keep current view visible; submit stretch worker ─
                # Store the saved viewport so _deliver_stretch can restore it.
                self._pending_saved_viewport = saved_viewport
                if not isinstance(self.centralWidget(), FitsView):
                    self.setCentralWidget(LoadingWidget())

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

            self._trigger_preload(*(_PRELOAD_HIT if stretch else _PRELOAD_MISS))

            _log.debug("_load_fits: TOTAL %.1f ms  [%s]  %s",
                       (time.perf_counter() - t0) * 1000, self._queue_info(), name)

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.setCentralWidget(QLabel(f"Error loading file:\n{e}"))

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
            _, _, old_size = saved_vp
            if old_size == (sd.mtf_qimage.width(), sd.mtf_qimage.height()):
                vp_to_restore = (saved_vp[0], saved_vp[1])

        _log.debug("_deliver_stretch: installing [%s]  %s",
                   self._queue_info(), os.path.basename(path))
        is_bad = path in self._bad_files
        view = FitsView(sd.mtf_qimage, sd, saved_viewport=vp_to_restore, is_bad=is_bad)
        self.setCentralWidget(view)

    def _navigate(self, delta: int):
        if not self._fits_files:
            return
        _log.info("key: %s  [%s]", "→" if delta > 0 else "←", self._queue_info())
        t0 = time.perf_counter()
        self._nav_direction = 1 if delta > 0 else -1
        saved_vp = None
        view = self.centralWidget()
        if isinstance(view, FitsView):
            saved_vp = view.viewport_state()
        self._current_index = (self._current_index + delta) % len(self._fits_files)
        self._load_fits(self._fits_files[self._current_index], saved_viewport=saved_vp)
        _log.debug("_navigate: TOTAL %.1f ms  [%s]", (time.perf_counter() - t0) * 1000,
                   self._queue_info())

    def _auto_stretch(self):
        _log.info("key: Cmd+4 auto-stretch  [%s]", self._queue_info())
        view = self.centralWidget()
        if isinstance(view, FitsView):
            view.auto_stretch()

    def _asinh_stretch(self):
        _log.info("key: Cmd+2 asinh-stretch  [%s]", self._queue_info())
        view = self.centralWidget()
        if isinstance(view, FitsView):
            view.apply_asinh_stretch()

    def _zscale_stretch(self):
        _log.info("key: Cmd+3 zscale-stretch  [%s]", self._queue_info())
        view = self.centralWidget()
        if isinstance(view, FitsView):
            view.apply_zscale_stretch()

    def _mtf_stretch(self):
        _log.info("key: Cmd+1 mtf-stretch  [%s]", self._queue_info())
        view = self.centralWidget()
        if isinstance(view, FitsView):
            view.apply_mtf_stretch()

    def _toggle_bad(self):
        if not self._fits_files: return
        path = self._fits_files[self._current_index]
        view = self.centralWidget()
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
            self.setCentralWidget(label)
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
    parser.add_argument("--log", action="store_true", help="Log debug output to /tmp/py-fits-preview.log")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging to console")
    args = parser.parse_args()

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
