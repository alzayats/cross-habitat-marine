"""Figure 3: Main Results — Heatmap + Forest Plot.

(a) Model x Adaptation heatmap of mean macro F1 across habitats.
(b) Forest plot: all combos ranked by mean F1 with error bars and param counts.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from figures.data_utils import load_results_df
from figures.plot_utils import (
    ADAPTATION_MARKERS,
    DOUBLE_COLUMN_WIDTH,
    FONT_SIZE_ANNOTATION,
    add_panel_label,
    format_param_count,
    get_model_color,
    save_figure,
    set_journal_style,
)

logger = logging.getLogger(__name__)

# Display ordering
MODEL_ORDER = ["DINOv2-B", "CLIP-B", "RN50-IN", "EffNet-B4"]
ADAPT_ORDER = ["Linear Probe", "LoRA (r=4)", "VPT-Deep", "Full Fine-tune"]


def generate_fig3(
    results_dir: str = "./outputs/within_habitat",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 3: heatmap + forest plot."""
    set_journal_style()
    df = load_results_df(results_dir)

    if df.empty:
        return _create_placeholder(output_path)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 3.2))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[0.85, 1.15], wspace=0.45,
        left=0.01, right=0.88, top=0.90, bottom=0.12,
    )

    # --- Panel (a): Heatmap ---
    ax_heat = fig.add_subplot(gs[0, 0])
    _draw_heatmap(df, ax_heat, fig)
    add_panel_label(ax_heat, "(a)", x=-0.15)

    # --- Panel (b): Forest plot ---
    ax_forest = fig.add_subplot(gs[0, 1])
    _draw_forest_plot(df, ax_forest)
    add_panel_label(ax_forest, "(b)", x=-0.02)

    if output_path:
        save_figure(fig, output_path)

    return fig


def _draw_heatmap(df: pd.DataFrame, ax: plt.Axes, fig: plt.Figure) -> None:
    """Draw model x adaptation heatmap of mean macro F1."""
    pivot = df.pivot_table(
        values="macro_f1",
        index="model_display",
        columns="adapt_display",
        aggfunc="mean",
    )

    pivot = pivot.reindex(
        index=[m for m in MODEL_ORDER if m in pivot.index],
        columns=[a for a in ADAPT_ORDER if a in pivot.columns],
    )

    mask = pivot.isna()

    # Shorten column names for less clutter
    col_short = {"Linear Probe": "LP", "LoRA (r=4)": "LoRA",
                 "VPT-Deep": "VPT", "Full Fine-tune": "Full FT"}
    pivot_display = pivot.rename(columns=col_short)
    mask_display = mask.rename(columns=col_short)

    hm = sns.heatmap(
        pivot_display,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.40,
        vmax=0.80,
        square=False,
        ax=ax,
        annot_kws={"size": 9, "fontweight": "bold"},
        linewidths=1.0,
        linecolor="white",
        cbar=False,
        mask=mask_display,
    )

    # Add colorbar manually — compact horizontal bar below
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0.40, 0.80))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal", shrink=0.6,
        pad=0.15, aspect=20,
    )
    cbar.set_label("Macro F1", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Gray out N/A cells
    for i in range(pivot_display.shape[0]):
        for j in range(pivot_display.shape[1]):
            if mask_display.iloc[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=True, color="#E8E8E8", zorder=0,
                ))
                ax.text(
                    j + 0.5, i + 0.5, "N/A",
                    ha="center", va="center",
                    fontsize=7, color="#AAA", fontstyle="italic",
                )

    ax.set_title("Mean Macro F1", fontsize=9, fontweight="bold", pad=10)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def _draw_forest_plot(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Draw forest plot: combos ranked by mean F1 with SD error bars."""
    # Short label map for combo names
    adapt_short = {
        "Linear Probe": "LP", "LoRA (r=4)": "LoRA",
        "VPT-Deep": "VPT", "Full Fine-tune": "Full FT",
    }

    combo_stats = (
        df.groupby(["model", "adaptation", "model_display", "adapt_display"])
        .agg(
            mean_f1=("macro_f1", "mean"),
            std_f1=("macro_f1", "std"),
            trainable=("trainable_params", "first"),
            trainable_pct=("trainable_pct", "first"),
        )
        .reset_index()
        .sort_values("mean_f1", ascending=True)
    )

    # Build short combo labels
    combo_stats["short_label"] = (
        combo_stats["model_display"] + " " +
        combo_stats["adapt_display"].map(lambda a: adapt_short.get(a, a))
    )

    y_positions = np.arange(len(combo_stats))

    for idx, (_, row) in enumerate(combo_stats.iterrows()):
        color = get_model_color(row["model"])
        marker = ADAPTATION_MARKERS.get(row["adaptation"], "o")

        ax.errorbar(
            row["mean_f1"], idx,
            xerr=row["std_f1"],
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            capthick=1.0,
            alpha=0.7,
        )
        ax.scatter(
            row["mean_f1"], idx,
            color=color,
            marker=marker,
            s=60,
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )

    # Annotate trainable params as a right-aligned column
    x_right = combo_stats["mean_f1"].max() + combo_stats["std_f1"].max() + 0.04
    for idx, (_, row) in enumerate(combo_stats.iterrows()):
        if pd.notna(row["trainable"]):
            param_str = format_param_count(int(row["trainable"]))
            pct = row["trainable_pct"]
            if pct < 0.01:
                pct_str = f"{pct:.3f}%"
            elif pct < 1:
                pct_str = f"{pct:.2f}%"
            else:
                pct_str = f"{pct:.0f}%"
            ax.text(
                x_right, idx,
                f"{param_str} ({pct_str})",
                fontsize=6.5,
                va="center", ha="left",
                color="#555",
                family="monospace",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(combo_stats["short_label"].values, fontsize=7)
    ax.set_ylim(-0.8, len(combo_stats) - 0.2)
    ax.set_xlabel("Mean Macro F1", fontsize=8)
    ax.set_title("Ranked by Performance", fontsize=9, fontweight="bold", pad=10)

    # Reference line at best
    ax.axvline(x=combo_stats["mean_f1"].max(), color="#DDD", linestyle=":",
               linewidth=0.8, zorder=0)

    # Set x limits with room for annotations
    xmin = max(0, combo_stats["mean_f1"].min() - combo_stats["std_f1"].max() - 0.03)
    ax.set_xlim(xmin, x_right + 0.18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.grid(axis="y", visible=False)


def _create_placeholder(output_path: Optional[str] = None) -> plt.Figure:
    """Create placeholder when no data available."""
    set_journal_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 5.0))
    ax.text(0.5, 0.5, "No results data available",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.axis("off")
    if output_path:
        save_figure(fig, output_path)
    return fig


if __name__ == "__main__":
    generate_fig3(output_path="outputs/figures/fig3_heatmap_within")
