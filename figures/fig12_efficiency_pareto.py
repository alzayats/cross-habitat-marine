"""Figure 5: Efficiency-Performance Tradeoff (Pareto frontier).

Scatter: log(trainable params) vs mean macro F1.
Bubble size = training time. Pareto frontier highlighted.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Manual label offsets to avoid overlap: (dx, dy) in points
LABEL_OFFSETS = {
    "DINOv2-B LP": (10, 8),
    "DINOv2-B LoRA": (10, -12),
    "DINOv2-B VPT": (10, 8),
    "EffNet-B4 FT": (10, 8),
    "RN50-IN FT": (10, -12),
    "CLIP-B LP": (10, -16),
    "CLIP-B VPT": (10, -2),
    "CLIP-B LoRA": (10, -16),
    "RN50-IN LP": (10, 8),
    "EffNet-B4 LP": (10, -8),
}

ADAPT_SHORT = {
    "Linear Probe": "LP", "LoRA (r=4)": "LoRA",
    "VPT-Deep": "VPT", "Full Fine-tune": "FT",
}


def _pareto_front(xs, ys):
    """Return indices of Pareto-optimal points (minimize x, maximize y)."""
    points = sorted(zip(xs, ys, range(len(xs))), key=lambda p: p[0])
    pareto = []
    best_y = -np.inf
    for x, y, idx in points:
        if y > best_y:
            pareto.append(idx)
            best_y = y
    return pareto


def generate_fig5(
    results_dir: str = "./outputs/within_habitat",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 5: efficiency-performance Pareto plot."""
    set_journal_style()
    df = load_results_df(results_dir)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 4.5))

    combo_stats = (
        df.groupby(["model", "adaptation", "model_display", "adapt_display"])
        .agg(
            mean_f1=("macro_f1", "mean"),
            std_f1=("macro_f1", "std"),
            trainable=("trainable_params", "first"),
            mean_time=("training_time", "mean"),
        )
        .reset_index()
        .dropna(subset=["trainable", "mean_f1"])
    )

    # Build short labels
    combo_stats["short_label"] = (
        combo_stats["model_display"] + " " +
        combo_stats["adapt_display"].map(lambda a: ADAPT_SHORT.get(a, a))
    )

    # Normalize bubble sizes
    time_min = combo_stats["mean_time"].min()
    time_max = combo_stats["mean_time"].max()
    size_norm = (combo_stats["mean_time"] - time_min) / max(time_max - time_min, 1)
    bubble_sizes = 100 + 250 * size_norm

    # Plot each combo
    for i, row in combo_stats.iterrows():
        color = get_model_color(row["model"])
        marker = ADAPTATION_MARKERS.get(row["adaptation"], "o")
        ax.scatter(
            row["trainable"], row["mean_f1"],
            c=color, marker=marker,
            s=bubble_sizes.loc[i],
            edgecolors="white", linewidths=0.8,
            zorder=5, alpha=0.85,
        )

        # Label with manual offset to avoid overlap
        label = row["short_label"]
        dx, dy = LABEL_OFFSETS.get(label, (10, 4))
        ax.annotate(
            label,
            (row["trainable"], row["mean_f1"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.5,
            color="#333",
            arrowprops=dict(arrowstyle="-", color="#BBB", lw=0.5),
        )

    # Pareto frontier
    xs = combo_stats["trainable"].values
    ys = combo_stats["mean_f1"].values
    pareto_idx = _pareto_front(xs, ys)

    if len(pareto_idx) > 1:
        pareto_df = combo_stats.iloc[pareto_idx].sort_values("trainable")
        ax.plot(
            pareto_df["trainable"], pareto_df["mean_f1"],
            "k--", linewidth=0.8, alpha=0.4, zorder=2,
        )

    # Highlight DINOv2 LP
    dino_lp = combo_stats[
        (combo_stats["model"] == "dinov2_base")
        & (combo_stats["adaptation"] == "linear_probe")
    ]
    if not dino_lp.empty:
        row = dino_lp.iloc[0]
        ax.scatter(
            row["trainable"], row["mean_f1"],
            facecolors="none", edgecolors="#C62828",
            s=300, linewidths=2.0, zorder=10,
        )
        ax.annotate(
            f"Best: {format_param_count(int(row['trainable']))} params\n"
            f"F1 = {row['mean_f1']:.2f}",
            (row["trainable"], row["mean_f1"]),
            textcoords="offset points",
            xytext=(14, -30),
            fontsize=7, color="#C62828", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C62828",
                      alpha=0.9),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=8)
    ax.set_ylabel("Mean Macro F1", fontsize=8)
    ax.set_title("Efficiency vs. Performance", fontsize=10,
                 fontweight="bold", pad=10)

    # Y-axis: give some breathing room
    ymin = combo_stats["mean_f1"].min() - 0.03
    ymax = combo_stats["mean_f1"].max() + 0.03
    ax.set_ylim(ymin, ymax)

    # Size legend
    for t_label, t_frac in [("Short", 0.0), ("Long", 1.0)]:
        s = 100 + 250 * t_frac
        ax.scatter([], [], c="gray", alpha=0.5, s=s,
                   label=f"{t_label} training", edgecolors="white")
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9,
              edgecolor="#DDD")

    ax.grid(alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


if __name__ == "__main__":
    generate_fig5(output_path="outputs/figures/fig12_efficiency_pareto")
