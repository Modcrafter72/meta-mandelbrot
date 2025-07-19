"""
Meta-Mandelbrot Visualization (Panel-Based)

Generates a grid of Mandelbrot variants by varying the initial value z₀ in the iteration:
    zₙ₊₁ = zₙ² + c
Each pixel in the final image represents a unique Mandelbrot set starting from a distinct z₀.

Author: Mika Grauel
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================
# Configuration
# =========================================

GRID_N = 100               # Number of z₀ values per row/column (GRID_N x GRID_N)
PANEL_SIZE = 30            # Resolution of each individual Mandelbrot (in pixels)
MAX_ITER = 50              # Max iterations for escape test
ESCAPE_RADIUS = 2.0        # Escape radius for Mandelbrot computation
Z0_RADIUS = 2.0            # z₀ values span the square [-Z0_RADIUS, Z0_RADIUS]
OUTPUT_FILE = "meta_mandelbrot_axes.png"
CMAP = "plasma"            # Colormap: 'plasma', 'inferno', 'turbo', etc.

# =========================================
# Prepare c-grid for inner Mandelbrot panels
# =========================================

RE_MIN, RE_MAX = -2.0, 1.0
IM_MIN, IM_MAX = -1.5, 1.5

re = np.linspace(RE_MIN, RE_MAX, PANEL_SIZE)
im = np.linspace(IM_MIN, IM_MAX, PANEL_SIZE)
c_grid = re[np.newaxis, :] + 1j * im[:, np.newaxis]

# =========================================
# Generate list of z₀ values across complex plane
# =========================================

z0_real = np.linspace(-Z0_RADIUS, Z0_RADIUS, GRID_N)
z0_imag = np.linspace(-Z0_RADIUS, Z0_RADIUS, GRID_N)
z0_values = [x + 1j * y for y in z0_imag for x in z0_real]

# =========================================
# Mandelbrot variant: compute escape time image
# =========================================

def mandelbrot_variant(z0, max_iter=MAX_ITER):
    """
    Computes a Mandelbrot image starting from z₀ instead of 0.
    Returns a 2D array of iteration counts.
    """
    z = np.full_like(c_grid, z0, dtype=np.complex128)
    mask = np.ones(c_grid.shape, dtype=bool)
    img = np.zeros(c_grid.shape, dtype=np.int32)

    for i in range(max_iter):
        z[mask] = z[mask]**2 + c_grid[mask]
        escaped = np.abs(z) > ESCAPE_RADIUS
        img[mask & escaped] = i
        mask &= ~escaped
        if not mask.any():
            break

    img[mask] = max_iter
    return img

# =========================================
# Compose meta-image of all Mandelbrot variants
# =========================================

canvas_size = GRID_N * PANEL_SIZE
meta_img = np.zeros((canvas_size, canvas_size), dtype=np.int32)

print(f"Computing {GRID_N * GRID_N} Mandelbrot variants...")
for idx, z0 in enumerate(tqdm(z0_values)):
    img = mandelbrot_variant(z0)
    row = idx // GRID_N
    col = idx % GRID_N
    meta_img[row*PANEL_SIZE:(row+1)*PANEL_SIZE,
             col*PANEL_SIZE:(col+1)*PANEL_SIZE] = img

# =========================================
# Plot with coordinate axes in z₀-plane
# =========================================

plt.figure(figsize=(10, 10))
extent = [-Z0_RADIUS, Z0_RADIUS, -Z0_RADIUS, Z0_RADIUS]
plt.imshow(meta_img, cmap=CMAP, origin='lower', extent=extent)
plt.title(f"Meta-Mandelbrot (Grid {GRID_N}×{GRID_N}, each with z₀ ≠ 0)", fontsize=13)
plt.xlabel("Re(z₀)", fontsize=12)
plt.ylabel("Im(z₀)", fontsize=12)
plt.colorbar(label="Escape iteration count")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Image saved as {OUTPUT_FILE}")
