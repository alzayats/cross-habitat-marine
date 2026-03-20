"""Figure 7: Class Imbalance Analysis.

(a) Empty F1 vs Valid F1 scatter (all runs excl. habitat 7268).
(b) Per-habitat difficulty bar chart sorted worst -> best.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from figures.data_utils import load_results_df
from figures.plot_utils import (
    ADAPTATION_MARKERS,
    DOUBLE_COLUMN_WIDTH,
    FONT_SIZE_ANNOTATION,
    add_panel_label,
    get_model_color,
    save_figure,
    set_journal_style,
)

logger = logging.getLogger(__name__)

DEGENERATE_HABITAT = "7268"


def generate_fig7(
    results_dir: str = "./outputs/within_habitat",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 7: class imbalance analysis."""
    set_journal_style()
    df = load_results_df(results_dir)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 5.0))
    gs = gridspec.GridSpec(
        1, 2, wspace=0.40, width_ratios=[1.0, 1.0],
        left=0.08, right=0.95, top=0.92, bottom=0.10,
    )

    ax_scatter = fig.add_subplot(gs[0, 0])
    _draw_class_scatter(df, ax_scatter)
    add_panel_label(ax_scatter, "(a)", x=-0.10)

    ax_bars = fig.add_subplot(gs[0, 1])
    _draw_difficulty_bars(df, ax_bars)
    add_panel_label(ax_bars, "(b)", x=-0.02)

    if output_path:
        save_figure(fig, output_path)

    return fig


def _draw_class_scatter(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Empty F1 vs Valid F1 scatter, colored by model, shaped by adaptation."""
    sub = df[df["target_habitat"] != DEGENERATE_HABITAT].dropna(
        subset=["empty_f1", "valid_f1"]
    )

    if sub.empty:
        ax.text(0.5, 0.5, "No per-class data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Plot individual runs — small, semi-transparent
    for _, row in sub.iterrows():
        color = get_model_color(row["model"])
        marker = ADAPTATION_MARKERS.get(row["adaptation"], "o")
        ax.scatter(
            row["valid_f1"], row["empty_f1"],
            c=color, marker=marker, s=12,
            alpha=0.3, edgecolors="none",
        )

    # Model centroids as stars — larger, with labels offset to avoid collision
    centroids = sub.groupby("model_display").agg(
        valid_f1=("valid_f1", "mean"),
        empty_f1=("empty_f1", "mean"),
        model=("model", "first"),
    ).reset_index()

    # Model-specific label offsets to avoid collisions
    label_offsets = {
        "RN50-IN":   (10, -18),
        "CLIP-B":    (-65, -18),
        "EffNet-B4": (-70, 10),
        "DINOv2-B":  (10, 10),
    }

    for _, cent in centroids.iterrows():
        color = get_model_color(cent["model"])
        ax.scatter(
            cent["valid_f1"], cent["empty_f1"],
            c=color, marker="*", s=180,
            edgecolors="black", linewidths=0.6, zorder=10,
        )
        dx, dy = label_offsets.get(cent["model_display"], (10, 8))
        ax.annotate(
            cent["model_display"],
            (cent["valid_f1"], cent["empty_f1"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5, alpha=0.5),
        )

    # Diagonal line (equal performance)
    ax.plot([0, 1], [0, 1], "k-", linewidth=0.6, alpha=0.25, zorder=0)

    # Label diagonal — position in data space, rotated
    ax.text(0.82, 0.88, "y = x", fontsize=6, color="#AAA",
            rotation=38, ha="center", va="center",
            transform=ax.transAxes)

    # Shade the region below diagonal to show bias
    ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.04, color="red", zorder=0)
    ax.text(0.6, 0.15, '"Empty" harder',
            fontsize=6.5, color="#C62828", alpha=0.6,
            transform=ax.transAxes, fontstyle="italic")

    ax.set_xlabel("Valid class F1", fontsize=8)
    ax.set_ylabel("Empty class F1", fontsize=8)
    ax.set_title("Per-Class Performance", fontsize=9,
                 fontweight="bold", pad=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.2, linewidth=0.4)


def _draw_difficulty_bars(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Horizontal bars sorted worst->best, colored by macro F1."""
    hab = df.groupby("target_habitat").agg(
        mean_f1=("macro_f1", "mean"),
        n_samples=("n_samples", "first"),
        empty_ratio=("empty_ratio", "mean"),
    ).reset_index().sort_values("mean_f1", ascending=True)

    from matplotlib.cm import RdYlGn
    norm = plt.Normalize(vmin=0, vmax=1)
    colors = [RdYlGn(norm(f)) for f in hab["mean_f1"]]

    y_pos = np.arange(len(hab))

    bars = ax.barh(y_pos, hab["mean_f1"].values, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.7)

    # Annotate: n_samples and empty ratio on right of each bar
    for i, (_, row) in enumerate(hab.iterrows()):
        # Put n= and empty% after the bar
        label = f"n={int(row['n_samples'])}"
        if row["empty_ratio"] > 0:
            label += f"  ({row['empty_ratio']:.0%} empty)"
        ax.text(
            max(row["mean_f1"] + 0.02, 0.08), i,
            label,
            va="center", fontsize=5.5, color="#444",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(hab["target_habitat"].values, fontsize=6.5)
    ax.set_xlabel("Mean Macro F1", fontsize=8)
    ax.set_title("Habitat Difficulty", fontsize=9, fontweight="bold", pad=8)
    ax.set_xlim(0, 1.35)  # room for annotations

    # Mark degenerate habitat in red
    if DEGENERATE_HABITAT in hab["target_habitat"].values:
        idx = list(hab["target_habitat"].values).index(DEGENERATE_HABITAT)
        ax.get_yticklabels()[idx].set_color("#C62828")
        ax.get_yticklabels()[idx].set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.4)
    ax.grid(axis="y", visible=False)


if __name__ == "__main__":
    generate_fig7(output_path="outputs/figures/fig5_class_imbalance")
