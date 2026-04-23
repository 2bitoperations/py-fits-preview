import os
import sys
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from astropy.io import fits
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSizePolicy,
)
from PySide6.QtGui import QImage, QPixmap, QFileOpenEvent, QKeySequence, QShortcut, QPainter
from PySide6.QtCore import Qt, QEvent, QTimer


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """1–99 percentile stretch → uint8."""
    a = arr.astype(np.float64)
    vmin, vmax = np.nanpercentile(a, [1, 99])
    if vmax == vmin:
        vmax = vmin + 1.0
    return np.clip((a - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)


def _gray_to_qimage(gray: np.ndarray) -> QImage:
    """(H, W) uint8 → QImage (grayscale). Safe to call from any thread."""
    gray = np.ascontiguousarray(gray)
    h, w = gray.shape
    # .copy() makes QImage own its pixel buffer so the numpy array can be freed.
    return QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """(H, W, 3) uint8 → QImage (RGB888). Safe to call from any thread."""
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


# ---------------------------------------------------------------------------
# Bayer demosaicing
# ---------------------------------------------------------------------------

# How many rows/cols to roll each pattern so R lands at (0, 0) → RGGB.
# np.roll(a, shift) moves a[i] → a[i + shift], so a[0] ← a[-shift].
# We need a[-shift_r, -shift_c] == R pixel, i.e. shift = -R_position.
_BAYER_TO_RGGB_ROLL: dict[str, tuple[int, int]] = {
    # pattern  R is at    roll = -R_pos
    "RGGB": (0, 0),   # R at (0,0) → roll (0, 0)
    "BGGR": (-1, -1), # R at (1,1) → roll (-1,-1)
    "GRBG": (0, -1),  # R at (0,1) → roll (0,-1)
    "GBRG": (-1, 0),  # R at (1,0) → roll (-1, 0)
}


def _debayer_rggb(raw: np.ndarray) -> np.ndarray:
    """
    Bilinear demosaicing of a 2-D array that is already in RGGB layout:
        R G R G …   (even rows)
        G B G B …   (odd rows)

    Returns (H, W, 3) float64 with the same value range as the input.
    """
    h, w = raw.shape
    d = raw.astype(np.float64)

    # Pad by 2 with reflection so edge arithmetic never goes out of bounds.
    p = np.pad(d, 2, mode="reflect")

    def s(dr: int, dc: int) -> np.ndarray:
        """Return the original-sized slice of p shifted by (dr, dc)."""
        return p[2 + dr: 2 + dr + h, 2 + dc: 2 + dc + w]

    # Boolean masks for each tile position
    rr, cc = np.mgrid[0:h, 0:w]
    at_R  = (rr % 2 == 0) & (cc % 2 == 0)  # R
    at_Gr = (rr % 2 == 0) & (cc % 2 == 1)  # G on R-row
    at_Gb = (rr % 2 == 1) & (cc % 2 == 0)  # G on B-row
    at_B  = (rr % 2 == 1) & (cc % 2 == 1)  # B

    # R channel -----------------------------------------------------------
    # known at R; horiz-avg at Gr; vert-avg at Gb; diag-avg at B
    R = np.where(at_R,  d,
        np.where(at_Gr, (s(0, -1) + s(0, +1)) / 2,
        np.where(at_Gb, (s(-1, 0) + s(+1, 0)) / 2,
                        (s(-1, -1) + s(-1, +1) + s(+1, -1) + s(+1, +1)) / 4)))

    # G channel -----------------------------------------------------------
    # known at Gr and Gb; cross-avg at R and B
    G_interp = (s(0, -1) + s(0, +1) + s(-1, 0) + s(+1, 0)) / 4
    G = np.where(at_Gr | at_Gb, d, G_interp)

    # B channel -----------------------------------------------------------
    # known at B; horiz-avg at Gb; vert-avg at Gr; diag-avg at R
    B = np.where(at_B,  d,
        np.where(at_Gb, (s(0, -1) + s(0, +1)) / 2,
        np.where(at_Gr, (s(-1, 0) + s(+1, 0)) / 2,
                        (s(-1, -1) + s(-1, +1) + s(+1, -1) + s(+1, +1)) / 4)))

    return np.stack([R, G, B], axis=-1)  # (H, W, 3)


def _debayer(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Debayer any supported Bayer pattern → (H, W, 3) float64."""
    pat = pattern.upper().strip()
    if pat not in _BAYER_TO_RGGB_ROLL:
        raise ValueError(f"Unsupported Bayer pattern: {pattern!r}")
    dr, dc = _BAYER_TO_RGGB_ROLL[pat]
    data = np.roll(raw, (dr, dc), axis=(0, 1))
    return _debayer_rggb(data)


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def fits_data_to_qimage(data: np.ndarray, header=None) -> QImage:
    """
    Convert FITS image data to a QImage. Safe to call from any thread.

    Handles:
      • 2-D grayscale
      • 2-D Bayer OSC  (BAYERPAT / COLORTYP keyword in header)
      • 3-D RGB cube   (3, H, W) or (H, W, 3)
    """
    # Squeeze length-1 axes (e.g. (1, H, W) → (H, W))
    data = np.squeeze(data)

    # --- 3-D: already colour ---
    if data.ndim == 3:
        if data.shape[0] == 3:
            rgb = np.stack([_normalize(data[i]) for i in range(3)], axis=-1)
        elif data.shape[2] == 3:
            rgb = np.stack([_normalize(data[:, :, i]) for i in range(3)], axis=-1)
        else:
            return fits_data_to_qimage(data[0], header)
        return _rgb_to_qimage(np.flipud(rgb))

    # --- 2-D ---
    if data.ndim == 2:
        bayer = None
        if header is not None:
            bayer = (
                header.get("BAYERPAT")
                or header.get("COLORTYP")
                or header.get("BAYER")
            )

        if bayer and str(bayer).upper().strip() in _BAYER_TO_RGGB_ROLL:
            rgb_f = _debayer(data, str(bayer))
            rgb = np.stack([_normalize(rgb_f[:, :, i]) for i in range(3)], axis=-1)
            return _rgb_to_qimage(np.flipud(rgb))

        return _gray_to_qimage(np.flipud(_normalize(data)))

    raise ValueError(f"Cannot display data with shape {data.shape}")


def fits_data_to_pixmap(data: np.ndarray, header=None) -> QPixmap:
    """Convenience wrapper — must be called from the main thread."""
    return QPixmap.fromImage(fits_data_to_qimage(data, header))


def _load_path_as_qimage(path: str) -> QImage | None:
    """
    Load a FITS file and return a QImage. Designed to run in a worker thread.
    Copies array data before the file handle closes so mmap backing is safe.
    """
    try:
        with fits.open(path) as hdul:
            hdu = next(
                (h for h in hdul if h.data is not None and h.data.ndim >= 2),
                None,
            )
            if hdu is None:
                return None
            data   = hdu.data.copy()   # copy before mmap closes
            header = hdu.header
        return fits_data_to_qimage(data, header)
    except Exception:
        return None


# (images_ahead, images_behind) relative to the direction of travel.
# MISS is used when the user jumped past our preloads — assume they'll keep going.
_PRELOAD_HIT  = (3, 2)
_PRELOAD_MISS = (8, 2)
_POOL_WORKERS = 4           # concurrent background loads
_CACHE_MAX    = 30          # max QImages held in RAM (~50–300 MB depending on size)

# ---------------------------------------------------------------------------
# Directory navigation helpers
# ---------------------------------------------------------------------------

_FITS_EXTENSIONS = frozenset({".fits", ".fit", ".fts"})


def _fits_siblings(path: str) -> list[str]:
    """Sorted list of FITS files in the same directory as *path*."""
    directory = os.path.dirname(os.path.abspath(path))
    siblings = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in _FITS_EXTENSIONS
    ]
    return sorted(siblings)


# ---------------------------------------------------------------------------
# Qt widgets
# ---------------------------------------------------------------------------

class FitsView(QGraphicsView):
    def __init__(self, pixmap: QPixmap, saved_viewport=None, parent=None):
        """
        saved_viewport: (QTransform, QPointF) to restore, or None to auto-fit.
        """
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem(pixmap)
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        scene.addItem(self._item)
        self.setScene(scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap_size = (pixmap.width(), pixmap.height())
        self._saved_viewport = saved_viewport   # (QTransform, QPointF) or None
        self._initial_fit_done = saved_viewport is not None

        # Cmd+0 (Qt maps Ctrl→Cmd on macOS) → 1:1 zoom
        QShortcut(QKeySequence("Ctrl+0"), self, self._zoom_one_to_one)

    @property
    def pixmap_size(self) -> tuple[int, int]:
        return self._pixmap_size

    def viewport_state(self) -> tuple:
        """Return (QTransform, center_QPointF, pixmap_size) for later restoration."""
        center = self.mapToScene(self.viewport().rect().center())
        return (self.transform(), center, self._pixmap_size)

    def _fit(self):
        self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _zoom_one_to_one(self):
        self.resetTransform()

    def showEvent(self, event):
        super().showEvent(event)
        if self._saved_viewport is not None:
            transform, center = self._saved_viewport
            self._saved_viewport = None
            QTimer.singleShot(0, lambda: (self.setTransform(transform), self.centerOn(center)))
        elif not self._initial_fit_done:
            self._initial_fit_done = True
            QTimer.singleShot(0, self._fit)

    def wheelEvent(self, event):
        # angleDelta is in eighths of a degree; a mouse wheel notch is 120 units.
        # Scale continuously so trackpad (many tiny deltas) feels smooth while
        # a mouse wheel notch still gives a reasonable jump (~15% per notch).
        delta = event.angleDelta().y()
        factor = 1.15 ** (delta / 120.0)
        self.scale(factor, factor)


class MainWindow(QMainWindow):
    def __init__(self, fits_path: str | None = None):
        super().__init__()
        self.setWindowTitle("FITS Preview")
        self.setGeometry(QApplication.primaryScreen().availableGeometry())
        self._fits_files: list[str] = []
        self._current_index: int = 0
        self._nav_direction: int = 1   # +1 = forward, -1 = backward

        # QImage cache: path → QImage, capped at _CACHE_MAX entries (LRU order).
        self._cache: OrderedDict[str, QImage] = OrderedDict()
        # In-flight preload futures: path → Future[QImage | None]
        self._preload_futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(max_workers=_POOL_WORKERS,
                                            thread_name_prefix="fits-preload")

        # Left/Right navigate siblings; window-level so they fire even when
        # FitsView has keyboard focus.
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._navigate(+1))
        QShortcut(QKeySequence(Qt.Key.Key_Left),  self, lambda: self._navigate(-1))

        if fits_path:
            self._load_fits(fits_path)
        else:
            self._show_placeholder()

    # ------------------------------------------------------------------
    # Cache helpers (main-thread only)
    # ------------------------------------------------------------------

    def _cache_put(self, path: str, image: QImage):
        self._cache[path] = image
        self._cache.move_to_end(path)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)

    def _cache_get(self, path: str) -> QImage | None:
        """Return cached QImage for *path*, draining a completed future if needed."""
        # Drain completed future into cache
        future = self._preload_futures.get(path)
        if future is not None and future.done():
            self._preload_futures.pop(path)
            try:
                result = future.result()
                if result is not None:
                    self._cache_put(path, result)
            except Exception:
                pass

        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        return None

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def _trigger_preload(self, ahead: int, behind: int):
        """
        Submit background loads for neighbors in the direction of travel.

        *ahead* images are loaded in ``_nav_direction``; *behind* in the
        opposite direction.  Any in-flight future that is no longer inside
        the desired window is cancelled so workers stay focused on useful work.
        """
        if not self._fits_files:
            return
        n = len(self._fits_files)
        d = self._nav_direction

        wanted: set[str] = set()
        for i in range(1, ahead + 1):
            wanted.add(self._fits_files[(self._current_index + d * i) % n])
        for i in range(1, behind + 1):
            wanted.add(self._fits_files[(self._current_index - d * i) % n])

        # Cancel futures that fell outside the new window (stale direction /
        # the user slowed down).  cancel() is a no-op if the task is already
        # running, but it prevents queued tasks from starting.
        for path in list(self._preload_futures):
            if path not in wanted:
                self._preload_futures.pop(path).cancel()

        for path in wanted:
            if path not in self._cache and path not in self._preload_futures:
                self._preload_futures[path] = self._executor.submit(
                    _load_path_as_qimage, path
                )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _show_placeholder(self):
        label = QLabel("Open a FITS file to get started.\n\nUsage: fits-preview <file.fits>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(label)

    def _load_fits(self, path: str, saved_viewport=None):
        abs_path = os.path.abspath(path)
        self.setWindowTitle(f"FITS Preview — {os.path.basename(abs_path)}")
        try:
            # --- Check cache / in-flight future ---
            qimage = self._cache_get(abs_path)
            cache_hit = qimage is not None

            if qimage is None:
                # Cache miss: cancel any queued (not-yet-running) future for
                # this path and load synchronously right now.
                future = self._preload_futures.pop(abs_path, None)
                if future is not None:
                    future.cancel()
                qimage = _load_path_as_qimage(abs_path)
                if qimage is None:
                    label = QLabel(f"No image data found in:\n{abs_path}")
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.setCentralWidget(label)
                    return

            self._cache_put(abs_path, qimage)
            pixmap = QPixmap.fromImage(qimage)

            # Decide whether to restore the previous viewport (same image size)
            # or let FitsView auto-fit the new image.
            vp_to_restore = None
            if saved_viewport is not None:
                _, _, old_size = saved_viewport
                if old_size == (pixmap.width(), pixmap.height()):
                    vp_to_restore = (saved_viewport[0], saved_viewport[1])

            view = FitsView(pixmap, saved_viewport=vp_to_restore)
            self.setCentralWidget(view)

            # Update the sibling file list and current position.
            self._fits_files = _fits_siblings(abs_path)
            try:
                self._current_index = self._fits_files.index(abs_path)
            except ValueError:
                self._current_index = 0

            ahead, behind = _PRELOAD_HIT if cache_hit else _PRELOAD_MISS
            self._trigger_preload(ahead, behind)

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            label = QLabel(f"Error loading file:\n{e}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setCentralWidget(label)

    def _navigate(self, delta: int):
        if not self._fits_files:
            return

        # Snapshot the current viewport before replacing it.
        saved_vp = None
        view = self.centralWidget()
        if isinstance(view, FitsView):
            saved_vp = view.viewport_state()

        self._current_index = (self._current_index + delta) % len(self._fits_files)
        self._load_fits(self._fits_files[self._current_index], saved_viewport=saved_vp)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class FitsApp(QApplication):
    """QApplication subclass that handles macOS QFileOpenEvent (Apple Events)."""

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
    """Print full tracebacks to stderr when running from a terminal."""
    if sys.stderr is None or not sys.stderr.isatty():
        return

    def hook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)

    sys.excepthook = hook


def main():
    _install_stderr_excepthook()
    app = FitsApp(sys.argv)
    fits_path = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(fits_path)
    app.set_main_window(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
