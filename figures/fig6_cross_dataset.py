"""Figure 9: Cross-Dataset Transfer (Protocol B).

(a) Grouped bar chart: macro F1 by transfer pair x model combo.
    Pairs ordered by difficulty (easy -> hardest).
(b) Domain gap scatter: source val F1 vs target test F1,
    diagonal = no gap. Points colored by model, shaped by adaptation.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

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

# Transfer pairs in difficulty order (matching run_protocol_b.sh)
PAIR_ORDER = [
    ("deepfish", "aqua20"),
    ("deepfish", "brackish"),
    ("deepfish", "moorea"),
    ("deepfish", "coralscapes"),
]

PAIR_LABELS = {
    ("deepfish", "aqua20"): "DF \u2192 AQ20",
    ("deepfish", "coralscapes"): "DF \u2192 Coral",
    ("deepfish", "moorea"): "DF \u2192 Moorea",
    ("deepfish", "brackish"): "DF \u2192 Brack",
}

PAIR_DIFFICULTY = {
    ("deepfish", "aqua20"): "Easiest",
    ("deepfish", "brackish"): "Medium",
    ("deepfish", "moorea"): "Hard",
    ("deepfish", "coralscapes"): "Hardest",
}

ADAPT_SHORT = {
    "linear_probe": "LP",
    "lora_r4": "LoRA",
    "vpt_deep": "VPT",
    "full_finetune": "FT",
}

MODEL_SHORT = {
    "dinov2_base": "DINOv2",
    "clip_base": "CLIP",
    "resnet50_imagenet": "RN50",
    "efficientnet_b4": "EffNet",
}


def load_cross_dataset_df(results_dir: str) -> pd.DataFrame:
    """Load Protocol B results into a DataFrame."""
    rows = []
    for path in sorted(Path(results_dir).rglob("results.json")):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            continue

        test = r.get("test_metrics", {})
        train = r.get("train_results", {})

        # Get best val F1 from training history
        history = train.get("history", [])
        best_val_f1 = max((h.get("val_f1", 0) for h in history), default=0)

        rows.append({
            "model": r.get("model", ""),
            "adaptation": r.get("adaptation", ""),
            "source": r.get("source", ""),
            "target": r.get("target", ""),
            "strategy": r.get("strategy", ""),
            "seed": r.get("seed", 42),
            "macro_f1": test.get("macro_f1", None),
            "balanced_accuracy": test.get("balanced_accuracy", None),
            "top1_accuracy": test.get("top1_accuracy", None),
            "best_val_f1": best_val_f1,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No Protocol B results found in %s", results_dir)
    return df


def generate_fig9(
    results_dir: str = "./outputs/cross_dataset",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 9: cross-dataset transfer results."""
    set_journal_style()
    df = load_cross_dataset_df(results_dir)

    if df.empty or len(df) < 5:
        logger.warning("Insufficient Protocol B data for Fig 9")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 3.8))
    gs = gridspec.GridSpec(
        1, 2, wspace=0.35, width_ratios=[1.4, 1.0],
        left=0.08, right=0.97, top=0.90, bottom=0.15,
    )

    ax_bars = fig.add_subplot(gs[0, 0])
    _draw_grouped_bars(df, ax_bars)
    add_panel_label(ax_bars, "(a)", x=-0.10)

    ax_gap = fig.add_subplot(gs[0, 1])
    _draw_domain_gap(df, ax_gap)
    add_panel_label(ax_gap, "(b)", x=-0.08)

    if output_path:
        save_figure(fig, output_path)
    return fig


def _draw_grouped_bars(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Grouped bar chart: F1 by transfer pair x model combos."""
    # Get all combos present in data, sorted by mean F1
    combo_means = (
        df.groupby(["model", "adaptation"])["macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )
    all_combos = combo_means.index.tolist()

    pairs = [p for p in PAIR_ORDER
             if ((df["source"] == p[0]) & (df["target"] == p[1])).any()]
    if not pairs:
        pairs = PAIR_ORDER

    n_pairs = len(pairs)
    n_combos = len(all_combos)
    x = np.arange(n_pairs)
    width = 0.85 / n_combos

    for i, (model, adapt) in enumerate(all_combos):
        vals = []
        for src, tgt in pairs:
            sub = df[(df["model"] == model) & (df["adaptation"] == adapt)
                     & (df["source"] == src) & (df["target"] == tgt)]
            vals.append(sub["macro_f1"].mean() if not sub.empty else 0)

        color = get_model_color(model)
        short = f"{MODEL_SHORT.get(model, model)} {ADAPT_SHORT.get(adapt, adapt)}"
        # Darken/lighten slightly for different adaptations of same model
        adapt_alpha = {"LP": 0.6, "LoRA": 0.8, "VPT": 0.9, "FT": 1.0}
        alpha = adapt_alpha.get(ADAPT_SHORT.get(adapt, ""), 0.85)

        ax.bar(x + i * width - 0.425 + width / 2, vals, width * 0.9,
               color=color, alpha=alpha, label=short,
               edgecolor="white", linewidth=0.4)

    # X-axis: pair labels with difficulty underneath
    pair_labels = []
    for p in pairs:
        diff = PAIR_DIFFICULTY.get(p, "")
        label = PAIR_LABELS.get(p, f"{p[0]}\u2192{p[1]}")
        pair_labels.append(f"{label}\n({diff})" if diff else label)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=7)
    ax.set_ylabel("Macro F1", fontsize=8)
    ax.set_title("Transfer Performance by Pair", fontsize=9,
                 fontweight="bold", pad=8)
    ax.legend(fontsize=5.5, ncol=2, loc="upper right", framealpha=0.9,
              edgecolor="#DDD", handlelength=1.0, handletextpad=0.4,
              columnspacing=0.8)
    ax.set_ylim(0, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def _draw_domain_gap(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Scatter: source val F1 vs target test F1."""
    # Diagonal line = no gap
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", lw=0.8, alpha=0.5,
            zorder=1)
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#fee", alpha=0.15,
                    zorder=0)

    # Plot per transfer pair (not aggregated) for more points
    for _, row in df.iterrows():
        color = get_model_color(row["model"])
        marker = ADAPTATION_MARKERS.get(row["adaptation"], "o")
        ax.scatter(
            row["best_val_f1"], row["macro_f1"],
            c=color, marker=marker, s=45,
            edgecolors="white", linewidths=0.4, alpha=0.8, zorder=5,
        )

    # Build proper legend with model colors + adaptation markers
    legend_handles = []

    # Model colors
    for model_key, model_name in MODEL_SHORT.items():
        color = get_model_color(model_key)
        h = ax.scatter([], [], c=color, s=30, label=model_name,
                       edgecolors="white", linewidths=0.4)
        legend_handles.append(h)

    # Separator
    from matplotlib.lines import Line2D
    sep = Line2D([], [], color="none", label="")
    legend_handles.append(sep)

    # Adaptation markers
    adapt_labels = {"linear_probe": "LP", "lora_r4": "LoRA",
                    "vpt_deep": "VPT", "full_finetune": "FT"}
    for adapt_key, adapt_name in adapt_labels.items():
        marker = ADAPTATION_MARKERS.get(adapt_key, "o")
        h = ax.scatter([], [], c="#666", marker=marker, s=30,
                       label=adapt_name, edgecolors="white", linewidths=0.4)
        legend_handles.append(h)

    ax.legend(handles=legend_handles, fontsize=5.5, loc="upper left",
              framealpha=0.9, edgecolor="#DDD", ncol=2,
              handletextpad=0.3, columnspacing=0.6)

    # "No gap" / "Gap" labels
    ax.text(0.15, 0.85, "No gap", fontsize=6.5, color="#8a8",
            ha="center", va="center", alpha=0.6, style="italic",
            transform=ax.transAxes)
    ax.text(0.80, 0.15, "Gap", fontsize=6.5, color="#c88",
            ha="center", va="center", alpha=0.6, style="italic",
            transform=ax.transAxes)

    ax.set_xlabel("Source Val F1", fontsize=8)
    ax.set_ylabel("Target Test F1", fontsize=8)
    ax.set_title("Domain Gap", fontsize=9, fontweight="bold", pad=8)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.02, 0.90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.5)


if __name__ == "__main__":
    generate_fig9(output_path="outputs/figures/fig6_cross_dataset")
