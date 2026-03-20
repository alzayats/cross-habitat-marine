"""Shared plotting utilities and journal-quality style configuration.

Defines consistent colors, line styles, and figure formatting for all
paper figures. Uses colorblind-friendly palettes and journal-standard
dimensions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# --- Journal formatting constants ---

# Figure widths in inches (converted from mm)
SINGLE_COLUMN_WIDTH = 89 / 25.4   # 89mm -> ~3.5 inches
DOUBLE_COLUMN_WIDTH = 183 / 25.4  # 183mm -> ~7.2 inches
DPI = 600

# Font sizes
FONT_SIZE_TITLE = 11
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 7

# --- Colorblind-friendly model color palette (Okabe-Ito inspired) ---

MODEL_COLORS: Dict[str, str] = {
    "dinov2_base": "#0072B2",       # Blue
    "dinov2_large": "#003F5C",      # Dark blue
    "dinov3_large": "#009E73",      # Green
    "clip_base": "#E69F00",         # Orange
    "mae_base": "#CC79A7",          # Purple/pink
    "resnet50_imagenet": "#999999", # Gray
    "resnet50_random": "#CCCCCC",   # Light gray
    "efficientnet_b4": "#666666",   # Dark gray
}

# Short display names for legends
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "dinov2_base": "DINOv2-B",
    "dinov2_large": "DINOv2-L",
    "dinov3_large": "DINOv3-L",
    "clip_base": "CLIP-B",
    "mae_base": "MAE-B",
    "resnet50_imagenet": "RN50-IN",
    "resnet50_random": "RN50-Rand",
    "efficientnet_b4": "EffNet-B4",
}

# --- Adaptation strategy line styles ---

ADAPTATION_LINESTYLES: Dict[str, str] = {
    "linear_probe": "dotted",
    "lora_r4": "dashed",
    "lora_r16": "solid",
    "vpt_deep": "dashdot",
    "full_finetune": "solid",
}

ADAPTATION_LINEWIDTHS: Dict[str, float] = {
    "linear_probe": 1.5,
    "lora_r4": 1.5,
    "lora_r16": 1.5,
    "vpt_deep": 1.5,
    "full_finetune": 2.5,
}

ADAPTATION_MARKERS: Dict[str, str] = {
    "linear_probe": "o",
    "lora_r4": "s",
    "lora_r16": "D",
    "vpt_deep": "^",
    "full_finetune": "v",
}

ADAPTATION_DISPLAY_NAMES: Dict[str, str] = {
    "linear_probe": "Linear Probe",
    "lora_r4": "LoRA (r=4)",
    "lora_r16": "LoRA (r=16)",
    "vpt_deep": "VPT-Deep",
    "full_finetune": "Full Fine-tune",
}


def set_journal_style() -> None:
    """Apply journal-quality matplotlib style settings."""
    plt.rcParams.update(
        {
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_SIZE_TICK,
            # Axes
            "axes.labelsize": FONT_SIZE_LABEL,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.which": "major",
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "grid.color": "#E0E0E0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            # Ticks
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Legend
            "legend.fontsize": FONT_SIZE_LEGEND,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": True,
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # PDF/PS: editable text (Type 42 fonts)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
        }
    )


def get_model_color(model_name: str) -> str:
    """Get the color for a model.

    Args:
        model_name: Model config name.

    Returns:
        Hex color string.
    """
    return MODEL_COLORS.get(model_name, "#333333")


def get_model_display_name(model_name: str) -> str:
    """Get the display name for a model.

    Args:
        model_name: Model config name.

    Returns:
        Display name string.
    """
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def get_adaptation_style(adaptation_name: str) -> Dict[str, any]:
    """Get line style properties for an adaptation strategy.

    Args:
        adaptation_name: Adaptation config name.

    Returns:
        Dict with 'linestyle', 'linewidth', 'marker' keys.
    """
    return {
        "linestyle": ADAPTATION_LINESTYLES.get(adaptation_name, "solid"),
        "linewidth": ADAPTATION_LINEWIDTHS.get(adaptation_name, 1.5),
        "marker": ADAPTATION_MARKERS.get(adaptation_name, "o"),
    }


def create_figure(
    width: str = "single",
    height_ratio: float = 0.75,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> Tuple[plt.Figure, any]:
    """Create a figure with journal-standard dimensions.

    Args:
        width: 'single' (89mm) or 'double' (183mm) column.
        height_ratio: Height as fraction of width.
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        **kwargs: Additional arguments for plt.subplots.

    Returns:
        Tuple of (figure, axes).
    """
    set_journal_style()

    fig_width = SINGLE_COLUMN_WIDTH if width == "single" else DOUBLE_COLUMN_WIDTH
    fig_height = fig_width * height_ratio

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        **kwargs,
    )

    return fig, axes


def save_figure(
    fig: plt.Figure,
    output_path: str,
    formats: Optional[List[str]] = None,
) -> None:
    """Save figure in multiple formats (PDF for paper, PNG for presentations).

    Args:
        fig: matplotlib Figure.
        output_path: Base output path (extension will be replaced).
        formats: List of formats to save. Default: ['pdf', 'png'].
    """
    from pathlib import Path

    if formats is None:
        formats = ["pdf", "png"]

    base = Path(output_path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = str(base) + f".{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        logger.info("Figure saved: %s", path)


def format_param_count(n: int) -> str:
    """Format parameter count for display (e.g., 1.5K, 86.6M).

    Args:
        n: Number of parameters.

    Returns:
        Human-readable string.
    """
    if n is None:
        return "N/A"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.08,
    y: float = 1.05,
    fontsize: int = 12,
) -> None:
    """Add a bold panel label (a), (b), etc. to an axes.

    Args:
        ax: matplotlib Axes.
        label: Label text, e.g. "(a)" or "a".
        x: X position in axes fraction.
        y: Y position in axes fraction.
        fontsize: Font size.
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def add_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.02,
) -> None:
    """Add a statistical significance bracket between two bars/points.

    Args:
        ax: matplotlib Axes.
        x1: Start x position.
        x2: End x position.
        y: Y position for bracket.
        p_value: P-value for annotation.
        height: Bracket height.
    """
    stars = ""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."

    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], "k-", linewidth=0.8)
    ax.text(
        (x1 + x2) / 2,
        y + height,
        stars,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE_ANNOTATION,
    )
