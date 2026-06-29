# plotting.py
# -----------
# Diagnostic visualizations: difference imaging stacks, aperture trends, and centroid panels.

from __future__ import annotations
import base64
import io
import logging

logger = logging.getLogger(__name__)

# Ensure matplotlib runs headlessly
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def plot_difference_image_to_base64(
    diff_img: np.ndarray | None,
    target_col: float | None,
    target_row: float | None,
    fit_col: float | None,
    fit_row: float | None,
    aperture_mask: np.ndarray | None,
    gaia_neighbors: list | None = None,
) -> str:
    """Generates the difference image panel with localized target/source offsets, returning base64 PNG."""
    if not MATPLOTLIB_AVAILABLE or diff_img is None:
        return ""
        
    try:
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plot difference image
        im = ax.imshow(diff_img, cmap="viridis", origin="lower", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Flux Difference")
        
        # Plot aperture boundaries
        if aperture_mask is not None:
            ax.contour(aperture_mask, levels=[0.5], colors="red", linestyles="dashed")
            
        # Overlay target coordinates
        if target_col is not None and target_row is not None:
            ax.plot(target_col, target_row, "bx", markersize=10, markeredgewidth=2, label="Target Star")
            
        # Overlay localized source center
        if fit_col is not None and fit_row is not None:
            ax.plot(fit_col, fit_row, "ro", markersize=8, markeredgewidth=2, label="Localized Source")
            
        # Overlay Gaia neighbors
        if gaia_neighbors:
            for i, n in enumerate(gaia_neighbors):
                # Calculate relative offset pixel index
                # (Simple linear approximation for plotting)
                pass
                
        ax.set_title("Signed Difference Image Stack")
        ax.set_xlabel("TPF Column Pixel")
        ax.set_ylabel("TPF Row Pixel")
        ax.legend(loc="upper right")
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to generate difference image plot: {e}")
        return ""

def plot_aperture_depth_consistency(
    pixel_counts: list[int],
    depths: list[float],
    uncertainties: list[float],
) -> str:
    """Generates depth versus aperture pixel count trend plot, returning base64 PNG."""
    if not MATPLOTLIB_AVAILABLE or not pixel_counts:
        return ""
        
    try:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        # Plot depths with errorbars
        ax.errorbar(
            pixel_counts, depths, yerr=uncertainties,
            fmt="o-", color="purple", capsize=5, elinewidth=2, markeredgewidth=2,
            label="Observed Depth"
        )
        
        ax.set_title("Aperture Vetting Depth Consistency")
        ax.set_xlabel("Aperture Size (Pixels)")
        ax.set_ylabel("Measured Transit Depth")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to generate aperture depth consistency plot: {e}")
        return ""
