# plots.py
# --------
# Plotting routines for Phase 2 validation and threshold sensitivity analysis.

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

# Safe headless matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def generate_evaluation_plots(
    df_features: pd.DataFrame,
    dest_dir: Path,
) -> None:
    """Generates PR curves and feature distribution plots for vetting documentation."""
    if not MATPLOTLIB_AVAILABLE or df_features.empty:
        return
        
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Plot 1: V-shape vs Depth scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        if "depth" in df_features.columns and "v_shape_score" in df_features.columns:
            for cls in df_features["label"].unique():
                cls_df = df_features[df_features["label"] == cls]
                ax.scatter(cls_df["depth"], cls_df["v_shape_score"], label=cls, alpha=0.6)
            ax.set_title("V-shape Score vs Transit Depth")
            ax.set_xlabel("Transit Depth")
            ax.set_ylabel("V-shape Score")
            ax.legend()
            fig.savefig(dest_dir / "v_shape_vs_depth.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        
        # Plot 2: Centroid shift vs Significance
        fig, ax = plt.subplots(figsize=(6, 5))
        if "centroid_shift_arcsec" in df_features.columns and "centroid_shift_significance" in df_features.columns:
            for cls in df_features["label"].unique():
                cls_df = df_features[df_features["label"] == cls]
                ax.scatter(cls_df["centroid_shift_significance"], cls_df["centroid_shift_arcsec"], label=cls, alpha=0.6)
            ax.set_title("Centroid Shift vs Significance")
            ax.set_xlabel("Significance (Sigma)")
            ax.set_ylabel("Shift (Arcsec)")
            ax.legend()
            fig.savefig(dest_dir / "centroid_shift_vs_significance.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        
    except Exception as e:
        logger.warning(f"Failed to generate evaluation plots: {e}")
