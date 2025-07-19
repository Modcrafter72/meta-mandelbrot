"""
Mandelbrot Variant Grid (z₀ ≠ 0)

This script visualizes how the Mandelbrot set changes when the iteration

    zₙ₊₁ = zₙ² + c

starts with different initial values z₀ ≠ 0. Each panel displays the Mandelbrot set
for a specific z₀ on a regular grid over the complex plane.

Note: This does NOT compute a single meta-image like the meta-Mandelbrot. Instead, 
it shows 2D arrays of full Mandelbrot images to illustrate structural differences 
as z₀ varies.

Author: Mika Grauel
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================
# Configuration
# =========================================
GRID_N = 10                  # Grid size: GRID_N × GRID_N panels
PANEL_SIZE = 300             # Resolution of each Mandelbrot variant
MAX_ITER = 100               # Max iterations for escape test
ESCAPE_RADIUS = 2.0          # Orbit considered divergent if |z| > ESCAPE_RADIUS
Z0_RADIUS = 2.0              # z₀ sampled in [-Z0_RADIUS, Z0_RADIUS] (real and imag)
CMAP = 'turbo'               # Good high-contrast colormap
OUTPUT_FILE = "mandelbrot_variants_grid.png"

# =========================================
# Domain for c-values (fixed across all panels)
# =========================================
RE_MIN, RE_MAX = -2.0, 1.0
IM_MIN, IM_MAX = -1.5, 1.5
re = np.linspace(RE_MIN, RE_MAX, PANEL_SIZE)
im = np.linspace(IM_MIN, IM_MAX, PANEL_SIZE)
c_grid = re[np.newaxis, :] + 1j * im[:, np.newaxis]

# =========================================
# z₀ values across meta-grid
# =========================================
z0_real = np.linspace(-Z0_RADIUS, Z0_RADIUS, GRID_N)
z0_imag = np.linspace(-Z0_RADIUS, Z0_RADIUS, GRID_N)
z0_values = [x + 1j*y for y in z0_imag for x in z0_real]

# =========================================
# Mandelbrot variant for a given z₀
# =========================================
def mandelbrot_variant(z0, max_iter=MAX_ITER):
    """
    Computes the Mandelbrot set for a fixed z₀ over all c-values in the domain.
    Returns an image representing iteration counts.
    """
    z = np.full_like(c_grid, z0, dtype=np.complex128)
    img = np.zeros_like(z, dtype=np.float32)
    mask = np.ones(z.shape, dtype=bool)

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
# Prepare full canvas image
# =========================================
canvas_size = GRID_N * PANEL_SIZE
canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)

print(f"Generating {GRID_N}×{GRID_N} Mandelbrot variants...")
for idx, z0 in enumerate(tqdm(z0_values)):
    row = idx // GRID_N
    col = idx % GRID_N
    mandel_img = mandelbrot_variant(z0)
    canvas[
        row * PANEL_SIZE:(row + 1) * PANEL_SIZE,
        col * PANEL_SIZE:(col + 1) * PANEL_SIZE
    ] = mandel_img

# =========================================
# Display and save final image
# =========================================
plt.figure(figsize=(10, 10))
plt.imshow(canvas, cmap=CMAP, origin='lower')
plt.axis('off')
plt.title("Mandelbrot Variants Grid: z₀ ≠ 0", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.show()

print(f"Image saved as: {OUTPUT_FILE}")
