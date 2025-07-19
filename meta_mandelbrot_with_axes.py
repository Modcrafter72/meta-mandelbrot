"""
Meta-Mandelbrot Scoremap with Axes

This script computes a 2D map showing how the Mandelbrot set changes
when the initial value z₀ in the iteration

    zₙ₊₁ = zₙ² + c

is varied across the complex plane. Each pixel represents a specific z₀
and encodes how many c-values lead to bounded orbits.

Author: Mika Grauel
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================
# Configuration
# =========================================

GRID_N = 100                # Number of z₀ values per row/column (image size)
MAX_ITER = 100              # Max iterations for escape test
ESCAPE_RADIUS = 2.0         # Orbit considered diverged if |z| > ESCAPE_RADIUS
Z0_RADIUS = 2.0             # Real/imag range of z₀ from -Z0_RADIUS to +Z0_RADIUS
OUTPUT_FILE = "meta_mandelbrot_with_axes.png"
CMAP = "turbo"              # Good perceptual colormap for scalar fields

# =========================================
# Mandelbrot domain (c-values)
# =========================================

RE_MIN, RE_MAX = -2.0, 1.0
IM_MIN, IM_MAX = -1.5, 1.5

PANEL_SIZE = 50             # Resolution of internal Mandelbrot used for scoring
re = np.linspace(RE_MIN, RE_MAX, PANEL_SIZE)
im = np.linspace(IM_MIN, IM_MAX, PANEL_SIZE)
c_grid = re[np.newaxis, :] + 1j * im[:, np.newaxis]

# =========================================
# z₀ sampling grid
# =========================================

z0_real = np.linspace(-Z0_RADIUS, Z0_RADIUS, GRID_N)
z0_imag = np.linspace(-Z0_RADIUS, Z0_RADIUS, GRID_N)
z0_values = [x + 1j*y for y in z0_imag for x in z0_real]

# =========================================
# Score function per z₀
# =========================================

def mandelbrot_score(z0, max_iter=MAX_ITER):
    """
    For a given z₀, computes how many c-values in the grid lead to bounded orbits.
    Returns a score ∈ [0, 1].
    """
    z = np.full_like(c_grid, z0, dtype=np.complex128)
    mask = np.ones_like(c_grid, dtype=bool)
    escape_map = np.zeros_like(c_grid, dtype=np.int32)

    for i in range(max_iter):
        z[mask] = z[mask]**2 + c_grid[mask]
        escaped = np.abs(z) > ESCAPE_RADIUS
        escape_map[mask & escaped] = i
        mask &= ~escaped
        if not mask.any():
            break

    escape_map[mask] = max_iter
    return np.sum(escape_map == max_iter) / escape_map.size

# =========================================
# Compute meta-image
# =========================================

meta_img = np.zeros((GRID_N, GRID_N), dtype=np.float32)

print(f"Computing Meta-Mandelbrot ({GRID_N} × {GRID_N})...")
for idx, z0 in enumerate(tqdm(z0_values)):
    row = idx // GRID_N
    col = idx % GRID_N
    meta_img[row, col] = mandelbrot_score(z0)

# =========================================
# Visualization
# =========================================

plt.figure(figsize=(10, 10))
extent = [-Z0_RADIUS, Z0_RADIUS, -Z0_RADIUS, Z0_RADIUS]
plt.imshow(meta_img, cmap=CMAP, origin='lower', extent=extent)
plt.title("Meta-Mandelbrot: Variation of z₀ in f_c(z) = z² + c", fontsize=14)
plt.xlabel("Re(z₀)", fontsize=12)
plt.ylabel("Im(z₀)", fontsize=12)
plt.colorbar(label='Stability Score')
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Image saved as {OUTPUT_FILE}")
