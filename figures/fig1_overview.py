"""Figure 1: Experimental overview — dataset summary, models, pipeline, protocols."""

import logging
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from figures.plot_utils import (
    save_figure, set_journal_style, DOUBLE_COLUMN_WIDTH,
    add_panel_label,
)

logger = logging.getLogger(__name__)


def generate_fig1(output_path: Optional[str] = None) -> plt.Figure:
    """Generate the experimental overview figure.

    Three-row layout:
      (a) Dataset table  |  (b) Models & adaptations table
      (c) Pipeline flow: Source -> Backbone -> Train -> Evaluate
      (d) Three experimental protocols
    """
    set_journal_style()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 5.8))

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 0.55, 1.0],
        width_ratios=[1.2, 0.8],
        hspace=0.25, wspace=0.20,
        left=0.03, right=0.97, top=0.97, bottom=0.02,
    )

    # (a) Dataset table — top-left
    ax_data = fig.add_subplot(gs[0, 0])
    _draw_dataset_table(ax_data)
    add_panel_label(ax_data, "(a)", x=-0.02, y=1.02, fontsize=11)

    # (b) Models + adaptations — top-right
    ax_models = fig.add_subplot(gs[0, 1])
    _draw_model_adaptation_tables(ax_models)
    add_panel_label(ax_models, "(b)", x=-0.04, y=1.02, fontsize=11)

    # (c) Pipeline — middle, spans both columns
    ax_pipe = fig.add_subplot(gs[1, :])
    _draw_pipeline(ax_pipe)
    add_panel_label(ax_pipe, "(c)", x=-0.01, y=1.10, fontsize=11)

    # (d) Protocols — bottom, spans both columns
    ax_proto = fig.add_subplot(gs[2, :])
    _draw_protocols(ax_proto)
    add_panel_label(ax_proto, "(d)", x=-0.01, y=1.05, fontsize=11)

    if output_path:
        save_figure(fig, output_path)

    return fig


def _draw_dataset_table(ax: plt.Axes) -> None:
    """Compact dataset summary table with alternating row colors."""
    ax.axis("off")

    col_labels = ["Dataset", "Domain", "Classes", "Images", "Task"]
    data = [
        ("DeepFish",      "20 habitats", "2",  "4 162\u20136 532", "Classification"),
        ("AQUA20",        "20 species",  "20", "\u223c1 000/cls",  "Classification"),
        ("Moorea Corals", "Coral reef",  "9",  "\u223c2 300",      "Classification"),
        ("Coralscapes",   "Coral reef",  "14", "\u223c4 500",      "Seg.\u2192Cls"),
        ("Brackish",      "Brackish",    "6",  "\u223c14 500",     "Det.\u2192Cls"),
    ]

    table = ax.table(
        cellText=data, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.4)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        cell.set_edgecolor("#0D47A1")
        cell.set_height(cell.get_height() * 1.1)

    for i in range(1, len(data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor("#EEF4FB" if i % 2 == 0 else "white")
            cell.set_edgecolor("#D0D0D0")
            cell.set_text_props(fontsize=7)
            if j == 0:
                cell.set_text_props(fontsize=7, fontweight="bold")

    ax.set_title("Datasets", fontsize=9, fontweight="bold",
                 pad=4, color="#1565C0")


def _draw_model_adaptation_tables(ax: plt.Axes) -> None:
    """Two compact tables: models and adaptations."""
    ax.axis("off")

    # --- Models table (top half) ---
    model_cols = ["Backbone", "Architecture", "Params"]
    model_data = [
        ("DINOv2-B",  "ViT-B/14",     "86M"),
        ("CLIP-B",    "ViT-B/16",     "86M"),
        ("RN50-IN",   "ResNet-50",    "25M"),
        ("EffNet-B4", "EfficientNet", "19M"),
    ]
    model_colors = ["#0072B2", "#E69F00", "#777777", "#555555"]

    tbl_m = ax.table(
        cellText=model_data, colLabels=model_cols,
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.52, 1.0, 0.48],
    )
    tbl_m.auto_set_font_size(False)
    tbl_m.set_fontsize(6.5)

    for j in range(len(model_cols)):
        cell = tbl_m[0, j]
        cell.set_facecolor("#2E7D32")
        cell.set_text_props(color="white", fontweight="bold", fontsize=6.5)
        cell.set_edgecolor("#1B5E20")

    for i in range(1, len(model_data) + 1):
        for j in range(len(model_cols)):
            cell = tbl_m[i, j]
            cell.set_facecolor("white")
            cell.set_edgecolor("#D0D0D0")
            if j == 0:
                cell.set_text_props(
                    fontsize=6.5, fontweight="bold",
                    color=model_colors[i - 1],
                )
            else:
                cell.set_text_props(fontsize=6.5, color="#444")

    ax.text(0.5, 1.02, "Models", fontsize=8, fontweight="bold",
            ha="center", va="bottom", transform=ax.transAxes,
            color="#2E7D32")

    # --- Adaptations table (bottom half) ---
    adapt_cols = ["Strategy", "Scope", "Applies to"]
    adapt_data = [
        ("Linear Probe",   "Head only",     "All"),
        ("LoRA (r=4)",     "QKV adapters",  "ViTs"),
        ("VPT-Deep",       "Prompt tokens", "ViTs"),
        ("Full Fine-tune", "All params",    "CNNs"),
    ]

    tbl_a = ax.table(
        cellText=adapt_data, colLabels=adapt_cols,
        loc="lower center", cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.48],
    )
    tbl_a.auto_set_font_size(False)
    tbl_a.set_fontsize(6.5)

    for j in range(len(adapt_cols)):
        cell = tbl_a[0, j]
        cell.set_facecolor("#E65100")
        cell.set_text_props(color="white", fontweight="bold", fontsize=6.5)
        cell.set_edgecolor("#BF360C")

    for i in range(1, len(adapt_data) + 1):
        for j in range(len(adapt_cols)):
            cell = tbl_a[i, j]
            cell.set_facecolor("white")
            cell.set_edgecolor("#D0D0D0")
            if j == 0:
                cell.set_text_props(fontsize=6.5, fontweight="bold", color="#333")
            else:
                cell.set_text_props(fontsize=6.5, color="#444")

    ax.text(0.5, 0.50, "Adaptations", fontsize=8, fontweight="bold",
            ha="center", va="bottom", transform=ax.transAxes,
            color="#E65100")


def _draw_pipeline(ax: plt.Axes) -> None:
    """Draw the pipeline flow: Source -> Backbone -> Train -> Evaluate."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.4)
    ax.axis("off")

    stages = [
        (0.15, 0.25, 2.0, 0.9, "Source\nHabitat(s)",     "#E3F2FD", "#1565C0"),
        (2.65, 0.25, 2.0, 0.9, "Backbone\n+ Adaptation",  "#E8F5E9", "#2E7D32"),
        (5.15, 0.25, 2.0, 0.9, "Train on\nSource Data",   "#FFF3E0", "#E65100"),
        (7.65, 0.25, 2.2, 0.9, "Evaluate on\nTarget Habitat", "#FCE4EC", "#C62828"),
    ]

    for x, y, w, h, label, fc, ec in stages:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.10", fc=fc, ec=ec, lw=1.5,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=ec)

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="#444", lw=1.8, mutation_scale=14)
    for x1, x2 in [(2.15, 2.65), (4.65, 5.15), (7.15, 7.65)]:
        ax.annotate("", xy=(x2, 0.7), xytext=(x1, 0.7), arrowprops=arrow_kw)

    ax.text(5.0, 1.35, "Experimental Pipeline", fontsize=9,
            fontweight="bold", ha="center", va="top", color="#333")


def _draw_protocols(ax: plt.Axes) -> None:
    """Draw the three protocol boxes side by side."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    protocols = [
        (0.05, "Protocol A", "Within-Habitat Transfer",
         "Train on 19 habitats, test on 1\n"
         "20 leave-one-out splits\n"
         "3 seeds per config",
         "#F3E5F5", "#6A1B9A"),
        (3.38, "Protocol B", "Cross-Dataset Transfer",
         "DeepFish \u2192 {AQUA20, Moorea,\n"
         "Coralscapes, Brackish}\n"
         "Open-set label mapping",
         "#E3F2FD", "#1565C0"),
        (6.72, "Protocol C", "Few-Shot Curves",
         "k = {0, 1, 2, 5, 10, 20, 50}\n"
         "3 target datasets, 5 trials each\n"
         "Label-efficiency analysis",
         "#FFFDE7", "#F57F17"),
    ]

    for x, title, subtitle, desc, fc, ec in protocols:
        box = FancyBboxPatch(
            (x, 0.05), 3.18, 2.45,
            boxstyle="round,pad=0.15", fc=fc, ec=ec, lw=1.3,
        )
        ax.add_patch(box)
        ax.text(x + 1.59, 2.15, title,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=ec)
        ax.text(x + 1.59, 1.80, subtitle,
                ha="center", va="center", fontsize=6.5,
                color=ec, fontstyle="italic")

        # Thin separator line
        ax.plot([x + 0.3, x + 2.88], [1.58, 1.58],
                color=ec, linewidth=0.5, alpha=0.4)

        ax.text(x + 1.59, 1.0, desc,
                ha="center", va="center", fontsize=6.5, color="#333",
                linespacing=1.5)


if __name__ == "__main__":
    generate_fig1("outputs/figures/fig1_overview")
