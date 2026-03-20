"""Figure 6: Training Dynamics.

(a) Convergence curves: val F1 over epochs for selected combos on a
    representative habitat.
(b) Best-epoch distribution: strip plot across all runs per combo.
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

REPRESENTATIVE_HABITAT = None  # None = average across all habitats

# Give each combo a distinct color — don't rely on model color alone
# since multiple combos share a model. Use model color + linestyle.
COMBO_STYLES = {
    "DINOv2-B + Linear Probe": {"color": "#0072B2", "ls": "-", "lw": 2.0},
    "DINOv2-B + LoRA (r=4)":  {"color": "#0072B2", "ls": "--", "lw": 1.5},
    "DINOv2-B + VPT-Deep":    {"color": "#0072B2", "ls": ":", "lw": 1.5},
    "CLIP-B + Linear Probe":  {"color": "#E69F00", "ls": "-", "lw": 2.0},
    "CLIP-B + LoRA (r=4)":    {"color": "#E69F00", "ls": "--", "lw": 1.5},
    "CLIP-B + VPT-Deep":      {"color": "#E69F00", "ls": ":", "lw": 1.5},
    "RN50-IN + Full Fine-tune": {"color": "#555555", "ls": "-", "lw": 1.8},
    "RN50-IN + Linear Probe":   {"color": "#555555", "ls": "--", "lw": 1.5},
    "EffNet-B4 + Full Fine-tune": {"color": "#CC79A7", "ls": "-", "lw": 1.8},
}

ADAPT_SHORT = {
    "Linear Probe": "LP", "LoRA (r=4)": "LoRA",
    "VPT-Deep": "VPT", "Full Fine-tune": "FT",
}


def generate_fig6(
    results_dir: str = "./outputs/within_habitat",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 6: training dynamics."""
    set_journal_style()
    df = load_results_df(results_dir)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 4.5))
    gs = gridspec.GridSpec(
        1, 2, wspace=0.35,
        left=0.08, right=0.97, top=0.90, bottom=0.13,
    )

    ax_conv = fig.add_subplot(gs[0, 0])
    _draw_convergence(df, ax_conv)
    add_panel_label(ax_conv, "(a)", x=-0.10)

    ax_best = fig.add_subplot(gs[0, 1])
    _draw_best_epoch_distribution(df, ax_best)
    add_panel_label(ax_best, "(b)", x=-0.02)

    if output_path:
        save_figure(fig, output_path)

    return fig


def _draw_convergence(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Mean source val F1 over epochs (all habitats) + mean target test F1."""
    # Pick top 6 combos by overall performance
    overall_rank = (
        df.groupby("combo")["macro_f1"].mean().sort_values(ascending=False)
    )
    selected = overall_rank.head(6).index.tolist()

    max_epoch = 0
    combo_test_f1 = {}

    for combo_name in selected:
        combo_runs = df[df["combo"] == combo_name]

        # Gather all val_f1 curves and average them
        all_curves = {}
        for _, run in combo_runs.iterrows():
            history = run["history"]
            if not history:
                continue
            for h in history:
                ep = h["epoch"]
                f1 = h.get("val_f1", 0)
                all_curves.setdefault(ep, []).append(f1)

        if not all_curves:
            continue

        epochs = sorted(all_curves.keys())
        mean_f1 = [np.mean(all_curves[e]) for e in epochs]
        std_f1 = [np.std(all_curves[e]) for e in epochs]
        max_epoch = max(max_epoch, max(epochs))

        style = COMBO_STYLES.get(combo_name,
                                  {"color": "#888", "ls": "-", "lw": 1.2})
        short = combo_name.split(" + ")
        short = (short[0] + " " +
                 ADAPT_SHORT.get(short[1], short[1]) if len(short) > 1
                 else combo_name)

        ax.plot(epochs, mean_f1,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                label=short, alpha=0.9)

        # Shaded ±1 SD band
        lo = [m - s for m, s in zip(mean_f1, std_f1)]
        hi = [m + s for m, s in zip(mean_f1, std_f1)]
        ax.fill_between(epochs, lo, hi,
                         color=style["color"], alpha=0.08)

        # Mean target test F1 across all habitats
        mean_test = combo_runs["macro_f1"].mean()
        combo_test_f1[combo_name] = (mean_test, style["color"], short)

    # Right-side diamond markers: mean target test F1
    marker_x = max_epoch + 4
    for combo_name, (test_f1, color, short) in combo_test_f1.items():
        ax.scatter(marker_x, test_f1, marker="D", c=color,
                   s=30, zorder=7, edgecolors="white", linewidths=0.5)

    ax.axvline(marker_x - 1.5, color="#CCC", ls=":", lw=0.5, zorder=1)
    highest_f1 = max(v[0] for v in combo_test_f1.values()) if combo_test_f1 else 0.8
    ax.text(marker_x, highest_f1 + 0.02,
            "Mean\ntarget F1", fontsize=5.5, ha="center", va="bottom",
            color="#777", style="italic")
    ax.set_xlim(right=marker_x + 4)

    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Mean Val F1 (source, ±1 SD)", fontsize=8)
    ax.set_title("Convergence (all habitats)", fontsize=9,
                 fontweight="bold", pad=8)
    ax.legend(fontsize=6, loc="lower left", framealpha=0.9,
              edgecolor="#DDD", ncol=1, handlelength=2.5)
    ax.grid(alpha=0.3, linewidth=0.5)


def _draw_best_epoch_distribution(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Strip plot of best_epoch across all runs per combo."""
    valid = df.dropna(subset=["best_epoch"]).copy()
    if valid.empty:
        ax.text(0.5, 0.5, "No epoch data", ha="center", va="center",
                transform=ax.transAxes)
        return

    # Short labels
    valid["short_label"] = (
        valid["model_display"] + " " +
        valid["adapt_display"].map(lambda a: ADAPT_SHORT.get(a, a))
    )

    # Order by median best_epoch
    combo_medians = valid.groupby("short_label")["best_epoch"].median().sort_values()
    combo_order = combo_medians.index.tolist()

    y_pos = {c: i for i, c in enumerate(combo_order)}
    rng = np.random.default_rng(42)

    for combo_name in combo_order:
        sub = valid[valid["short_label"] == combo_name]
        model = sub["model"].iloc[0]
        combo_full = sub["combo"].iloc[0]
        style = COMBO_STYLES.get(combo_full,
                                  {"color": get_model_color(model)})
        color = style["color"]
        y = y_pos[combo_name]

        jitter = rng.uniform(-0.2, 0.2, len(sub))
        ax.scatter(
            sub["best_epoch"], y + jitter,
            c=color, s=14, alpha=0.5, edgecolors="none", zorder=4,
        )

        # Median line — thick and clear
        med = sub["best_epoch"].median()
        ax.plot([med, med], [y - 0.35, y + 0.35],
                color=color, linewidth=2.5, zorder=5)

    ax.set_yticks(range(len(combo_order)))
    ax.set_yticklabels(combo_order, fontsize=6.5)
    ax.set_xlabel("Best Epoch", fontsize=8)
    ax.set_title("Best Epoch Distribution", fontsize=9,
                 fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.grid(axis="y", visible=False)

    # Set y limits with padding
    ax.set_ylim(-0.6, len(combo_order) - 0.4)


if __name__ == "__main__":
    generate_fig6(output_path="outputs/figures/fig11_training_dynamics")
