import threading
import numpy as np
import logging

_log = logging.getLogger(__name__)

# Try importing backends
try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# The Numba "workqueue" threading layer (default on Apple Silicon) is not thread-safe 
# when invoked concurrently from multiple Python threads. Since Numba's parallel=True 
# already saturates all CPU cores, we serialize access to it to prevent both crashes 
# and massive CPU oversubscription.
_numba_lock = threading.Lock()

if HAS_NUMBA:
    _log.info("Using Numba backend for MTF stretch.")
    
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _apply_mtf_color_numba(u16: np.ndarray, black: np.ndarray, span: np.ndarray, M_luma: float) -> np.ndarray:
        """
        Numba-accelerated color MTF stretch.
        Processes the array in-place via parallel threads, bypassing massive memory allocations.
        """
        H, W, C = u16.shape
        out = np.empty((H, W, 3), dtype=np.uint8)
        
        # black and span are expected to be 1D float32 arrays of size 3
        for i in nb.prange(H):
            for j in range(W):
                # 1. Normalize linear input to [0, 1]
                r_lin = u16[i, j, 0] / 65535.0
                g_lin = u16[i, j, 1] / 65535.0
                b_lin = u16[i, j, 2] / 65535.0
                
                # 2. Apply per-channel black point and span
                r_sub = (r_lin - black[0]) / span[0]
                if r_sub < 0.0: r_sub = 0.0
                elif r_sub > 1.0: r_sub = 1.0
                
                g_sub = (g_lin - black[1]) / span[1]
                if g_sub < 0.0: g_sub = 0.0
                elif g_sub > 1.0: g_sub = 1.0
                
                b_sub = (b_lin - black[2]) / span[2]
                if b_sub < 0.0: b_sub = 0.0
                elif b_sub > 1.0: b_sub = 1.0
                
                # 3. Compute linear luminance (Rec.709)
                luma_lin = 0.2126 * r_sub + 0.7152 * g_sub + 0.0722 * b_sub
                
                # 4. Apply PixInsight MTF rational function to luminance
                if luma_lin == 0.0:
                    luma_str = 0.0
                else:
                    denom = (2.0 * M_luma - 1.0) * luma_lin - M_luma
                    if abs(denom) < 1e-12:
                        denom = 1.0
                    luma_str = (M_luma - 1.0) * luma_lin / denom
                    if luma_str < 0.0: luma_str = 0.0
                    elif luma_str > 1.0: luma_str = 1.0
                    
                # 5. Compute chroma preservation ratio
                if luma_lin > 1e-6:
                    ratio = luma_str / luma_lin
                else:
                    ratio = 0.0
                    
                # 6. Apply ratio and quantize to 8-bit
                r_out = r_sub * ratio
                if r_out < 0.0: r_out = 0.0
                elif r_out > 1.0: r_out = 1.0
                
                g_out = g_sub * ratio
                if g_out < 0.0: g_out = 0.0
                elif g_out > 1.0: g_out = 1.0
                
                b_out = b_sub * ratio
                if b_out < 0.0: b_out = 0.0
                elif b_out > 1.0: b_out = 1.0
                
                out[i, j, 0] = int(r_out * 255.0)
                out[i, j, 1] = int(g_out * 255.0)
                out[i, j, 2] = int(b_out * 255.0)
                
        return out
        
    def apply_mtf_color(u16: np.ndarray, black: np.ndarray, span: np.ndarray, M_luma: float) -> np.ndarray:
        with _numba_lock:
            return _apply_mtf_color_numba(u16, black, span, M_luma)

else:
    _log.warning("Numba not found. Using slow pure-NumPy fallback for MTF stretch.")
    
    def apply_mtf_color(u16: np.ndarray, black: np.ndarray, span: np.ndarray, M_luma: float) -> np.ndarray:
        """
        Pure NumPy fallback. Highly memory intensive due to intermediate array allocations.
        """
        lin = u16.astype(np.float32) / 65535.0
        bg_sub = np.clip((lin - black[np.newaxis, np.newaxis, :]) / span[np.newaxis, np.newaxis, :], 0.0, 1.0)
        luma_lin = (0.2126 * bg_sub[:, :, 0] + 0.7152 * bg_sub[:, :, 1] + 0.0722 * bg_sub[:, :, 2])
        
        denom = (2.0 * M_luma - 1.0) * luma_lin - M_luma
        safe = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        result_luma = (M_luma - 1.0) * luma_lin / safe
        luma_str = np.where(luma_lin == 0.0, 0.0, np.clip(result_luma, 0.0, 1.0))
        
        safe_luma = np.where(luma_lin > 1e-6, luma_lin, 1.0)
        ratio = np.where(luma_lin > 1e-6, luma_str / safe_luma, 0.0)
        result = np.clip(bg_sub * ratio[:, :, np.newaxis], 0.0, 1.0)
        return (result * 255.0).astype(np.uint8)
