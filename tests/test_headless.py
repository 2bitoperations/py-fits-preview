import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
from astropy.io import fits
import cv2

# Add parent directory to sys.path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import run_headless

@pytest.fixture
def dummy_fits_file():
    """Generates a tiny dummy FITS file with a BAYERPAT header for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        fits_path = Path(temp_dir) / "test_image.fit"
        
        # Create a small 100x100 dummy sensor data array with a gradient
        data = np.linspace(0, 10000, 100*100).reshape((100, 100)).astype(np.uint16)
        
        # Add some noise to prevent degenerate statistics
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 500, size=(100, 100), dtype=np.uint16)
        data = data + noise
        
        # Build the Primary HDU
        hdu = fits.PrimaryHDU(data)
        
        # Add headers needed by the pipeline
        hdu.header['BAYERPAT'] = 'RGGB'
        hdu.header['BZERO'] = 0.0
        hdu.header['BSCALE'] = 1.0
        
        # Save to disk
        hdu.writeto(fits_path)
        
        yield fits_path

def test_headless_mtf_stretch(dummy_fits_file):
    """Test that the MTF stretch runs headless and outputs a valid image file."""
    out_path = dummy_fits_file.with_name("output_mtf.png")
    
    # Run the pipeline
    run_headless(str(dummy_fits_file), str(out_path), stretch_type="mtf")
    
    # Assert output exists
    assert out_path.exists(), "Output image was not created!"
    
    # Read output and verify it's a valid RGB image
    img = cv2.imread(str(out_path))
    assert img is not None, "Failed to read the output image with OpenCV."
    
    # 100x100 raw Bayer array demosaics into 100x100 RGB using OpenCV
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] == 100
    assert img.shape[1] == 100

def test_headless_auto_stretch(dummy_fits_file):
    """Test that an alternative stretch (auto) works properly."""
    out_path = dummy_fits_file.with_name("output_auto.png")
    
    # Run the pipeline with auto stretch
    run_headless(str(dummy_fits_file), str(out_path), stretch_type="auto")
    
    assert out_path.exists()
    img = cv2.imread(str(out_path))
    assert img is not None
    assert img.ndim == 3

def test_mtf_math_correctness():
    """Directly test the mathematical correctness of the MTF backend function."""
    import compute_backend
    
    # Create a simple 2x2x3 array
    u16 = np.zeros((2, 2, 3), dtype=np.uint16)
    
    # Set a pixel to a known value: e.g. R=65535, G=65535, B=65535
    u16[0, 0] = [65535, 65535, 65535] # white, [1.0, 1.0, 1.0]
    
    # Midpoint exactly at 32767
    u16[0, 1] = [32767, 32767, 32767] # mid, ~[0.5, 0.5, 0.5]
    
    # Black at 0
    u16[1, 0] = [0, 0, 0]             # black, [0.0, 0.0, 0.0]
    
    # Pure red
    u16[1, 1] = [65535, 0, 0] 
    
    black = np.zeros(3, dtype=np.float32)
    span = np.ones(3, dtype=np.float32)
    M_luma = 0.25 # standard midtone balance
    
    out = compute_backend.apply_mtf_color(u16, black, span, M_luma)
    
    # White should remain white
    assert list(out[0, 0]) == [255, 255, 255], f"Expected white, got {out[0, 0]}"
    
    # Black should remain black
    assert list(out[1, 0]) == [0, 0, 0], f"Expected black, got {out[1, 0]}"
    
    # Mid point [0.5, 0.5, 0.5]:
    # luma_lin = 0.5
    # denom = (2 * 0.25 - 1) * 0.5 - 0.25 = -0.5 * 0.5 - 0.25 = -0.5
    # luma_str = (0.25 - 1) * 0.5 / -0.5 = -0.75 * 0.5 / -0.5 = 0.75
    # ratio = 0.75 / 0.5 = 1.5
    # out = 0.5 * 1.5 = 0.75 -> 0.75 * 255 = 191
    np.testing.assert_allclose(out[0, 1], [191, 191, 191], atol=1)
    
    # Pure red [1.0, 0.0, 0.0]
    # luma_lin = 0.2126 * 1.0 = 0.2126
    # denom = (2 * 0.25 - 1) * 0.2126 - 0.25 = -0.5 * 0.2126 - 0.25 = -0.3563
    # luma_str = -0.75 * 0.2126 / -0.3563 = 0.44748...
    # ratio = 0.44748 / 0.2126 = 2.1048...
    # r_out = 1.0 * 2.1048 = 2.1048 -> clipped to 1.0 -> 255
    np.testing.assert_allclose(out[1, 1], [255, 0, 0], atol=1)

def test_headless_stretch_black_white_points(dummy_fits_file):
    """Test that the auto stretch properly clips to black and white."""
    out_path = dummy_fits_file.with_name("output_auto_bw.png")
    run_headless(str(dummy_fits_file), str(out_path), stretch_type="auto")
    
    img = cv2.imread(str(out_path))
    
    # The gradient in dummy_fits_file guarantees that the 1st percentile 
    # maps to 0 and the 99th percentile maps to 255.
    # Therefore, the image MUST contain absolute black (0) and absolute white (255) pixels.
    assert np.min(img) == 0, "Auto stretch failed to clip shadows to absolute black (0)"
    assert np.max(img) == 255, "Auto stretch failed to clip highlights to absolute white (255)"
