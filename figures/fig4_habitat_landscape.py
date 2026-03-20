"""Figure 4: Habitat Difficulty Landscape.

(a) Top-5 combos x 20 habitats heatmap (sorted by difficulty).
(b) Test set size vs mean F1, colored by class imbalance.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from figures.data_utils import load_results_df
from figures.plot_utils import (
    DOUBLE_COLUMN_WIDTH,
    FONT_SIZE_ANNOTATION,
    add_panel_label,
    save_figure,
    set_journal_style,
)

logger = logging.getLogger(__name__)

# Short names for heatmap y-axis
COMBO_SHORT = {
    "DINOv2-B + Linear Probe": "DINOv2-B LP",
    "EffNet-B4 + Full Fine-tune": "EffNet-B4 FT",
    "DINOv2-B + VPT-Deep": "DINOv2-B VPT",
    "DINOv2-B + LoRA (r=4)": "DINOv2-B LoRA",
    "RN50-IN + Full Fine-tune": "RN50-IN FT",
    "CLIP-B + Linear Probe": "CLIP-B LP",
    "CLIP-B + VPT-Deep": "CLIP-B VPT",
    "CLIP-B + LoRA (r=4)": "CLIP-B LoRA",
    "RN50-IN + Linear Probe": "RN50-IN LP",
}


def generate_fig4(
    results_dir: str = "./outputs/within_habitat",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 4: habitat difficulty landscape."""
    set_journal_style()
    df = load_results_df(results_dir)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 5.5))
    gs = gridspec.GridSpec(
        2, 1, height_ratios=[1.3, 1.0], hspace=0.45,
        left=0.10, right=0.92, top=0.95, bottom=0.08,
    )

    ax_heat = fig.add_subplot(gs[0, 0])
    _draw_habitat_heatmap(df, ax_heat)
    add_panel_label(ax_heat, "(a)", x=-0.08)

    ax_scatter = fig.add_subplot(gs[1, 0])
    _draw_size_vs_f1(df, ax_scatter)
    add_panel_label(ax_scatter, "(b)", x=-0.08)

    if output_path:
        save_figure(fig, output_path)

    return fig


def _draw_habitat_heatmap(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Heatmap: top 5 combos x 20 habitats, sorted by difficulty."""
    # Top 5 combos by mean F1
    combo_means = df.groupby("combo")["macro_f1"].mean().sort_values(ascending=False)
    top5 = combo_means.head(5).index.tolist()

    sub = df[df["combo"].isin(top5)]

    pivot = sub.pivot_table(
        values="macro_f1",
        index="combo",
        columns="target_habitat",
        aggfunc="mean",
    )

    # Sort habitats by difficulty (hardest left)
    habitat_order = pivot.mean(axis=0).sort_values().index
    pivot = pivot[habitat_order]

    # Sort combos by overall mean (best on top)
    combo_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[combo_order]

    # Shorten combo names
    pivot.index = [COMBO_SHORT.get(c, c) for c in pivot.index]

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        annot_kws={"size": 6},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Macro F1", "pad": 0.02},
    )

    # Annotate habitat 7268 if present
    if "7268" in pivot.columns:
        col_idx = list(pivot.columns).index("7268")
        # Place annotation above the heatmap to avoid x-tick overlap
        ax.annotate(
            "degenerate\n(0 empty)",
            xy=(col_idx + 0.5, 0),
            xytext=(col_idx + 2.5, -0.8),
            fontsize=6, color="#C62828", fontstyle="italic",
            ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8),
        )

    ax.set_title("Macro F1 by Habitat (top 5 configs)",
                 fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel("Target Habitat (hardest \u2192 easiest)", fontsize=7)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=6.5)
    ax.tick_params(axis="y", rotation=0, labelsize=7)


def _draw_size_vs_f1(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter: test set size vs mean F1, colored by class imbalance."""
    hab = df.groupby("target_habitat").agg(
        mean_f1=("macro_f1", "mean"),
        n_samples=("n_samples", "first"),
        empty_ratio=("empty_ratio", "mean"),
    ).reset_index()

    # Color by mean F1 using same cmap as panel (a) for consistency
    sc = ax.scatter(
        hab["n_samples"],
        hab["mean_f1"],
        c=hab["mean_f1"],
        cmap="RdYlGn",
        s=55,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
        vmin=0.0, vmax=1.0,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Macro F1", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Spearman correlation
    rho, pval = stats.spearmanr(hab["n_samples"], hab["mean_f1"])
    ax.text(
        0.97, 0.05,
        f"Spearman \u03c1 = {rho:.2f} (p = {pval:.2f})",
        transform=ax.transAxes,
        fontsize=7, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDD", alpha=0.9),
    )

    # Annotate only the most notable habitats
    # Find: worst F1, best F1, degenerate (7268)
    notable = set()
    notable.add(hab.loc[hab["mean_f1"].idxmin(), "target_habitat"])
    notable.add(hab.loc[hab["mean_f1"].idxmax(), "target_habitat"])
    notable.add(hab.loc[hab["n_samples"].idxmax(), "target_habitat"])
    if "7268" in hab["target_habitat"].values:
        notable.add("7268")

    for _, row in hab.iterrows():
        if row["target_habitat"] in notable:
            # Offset to avoid overlap
            yoff = -10 if row["mean_f1"] > 0.5 else 8
            ax.annotate(
                row["target_habitat"],
                (row["n_samples"], row["mean_f1"]),
                textcoords="offset points",
                xytext=(6, yoff),
                fontsize=6.5,
                fontweight="bold",
                color="#333",
                arrowprops=dict(arrowstyle="-", color="#999", lw=0.5),
            )

    ax.set_xlabel("Test set size (n samples)", fontsize=8)
    ax.set_ylabel("Mean Macro F1", fontsize=8)
    ax.set_title("Sample Size vs. Performance", fontsize=9,
                 fontweight="bold", pad=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linewidth=0.5)


if __name__ == "__main__":
    generate_fig4(output_path="outputs/figures/fig4_habitat_landscape")
