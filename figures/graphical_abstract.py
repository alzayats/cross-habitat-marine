"""Single-panel graphical abstract for the cross-habitat marine transfer study.

Journals typically require a high-resolution raster image (JPEG or TIFF) with a short
caption. It is used in the table of contents, for promotion, and is considered for the
journal cover, so it must read at thumbnail size: one message, large type, no clutter.

The message chosen is the paper's practical claim: recognition accuracy rises steeply
with the first few labelled images per class and plateaus early, so a small annotation
budget buys most of the achievable performance.

Numbers are read from the Protocol C curves rather than hardcoded.

Usage:
    python -m figures.graphical_abstract
"""

from __future__ import annotations

import glob
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

logger = logging.getLogger(__name__)

K_VALUES = [1, 2, 5, 10, 20, 50, 100]
TARGET_LABEL = {"aqua20": "AQUA20", "moorea": "Moorea", "brackish": "Brackish"}
TARGET_COLOR = {"aqua20": "#0072B2", "moorea": "#D55E00", "brackish": "#009E73"}

OUT_STEM = "outputs/figures/graphical_abstract"


def load_curves(root: str = "outputs/fewshot_curve") -> dict:
    """Mean balanced accuracy per target per k, across all model-adaptation combos."""
    agg: dict = defaultdict(list)
    for f in glob.glob(f"{root}/*/k*_trial*/results.json"):
        m = re.search(r"deepfish_to_(\w+?)_combined_seed42/k(\d+)_trial", f)
        if not m:
            continue
        try:
            agg[(m.group(1), int(m.group(2)))].append(
                json.load(open(f))["balanced_accuracy"])
        except Exception:
            continue
    return agg


def generate(output_stem: str = OUT_STEM) -> None:
    agg = load_curves()
    if not agg:
        raise SystemExit("No Protocol C results found under outputs/fewshot_curve")

    fig, ax = plt.subplots(figsize=(7.2, 4.05), dpi=300)   # 16:9, ~2160x1215 px

    for tgt in ["aqua20", "moorea", "brackish"]:
        ks = [k for k in K_VALUES if agg.get((tgt, k))]
        ys = [float(np.mean(agg[(tgt, k)])) for k in ks]
        ax.plot(ks, ys, marker="o", markersize=5.5, linewidth=2.6,
                color=TARGET_COLOR[tgt], label=TARGET_LABEL[tgt],
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)

    # Shade the recommended budget.
    ax.axvspan(10, 20, color="#FFD24D", alpha=0.30, zorder=0, lw=0)
    ax.text(14, 0.055, "10–20 labels\nper class", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#7A5A00", zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=10)
    ax.set_xlim(0.85, 118)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="y", labelsize=10)

    ax.set_xlabel("Labelled images per class at the new site  ($k$)",
                  fontsize=11.5, labelpad=6)
    ax.set_ylabel("Balanced accuracy", fontsize=11.5, labelpad=6)

    ax.set_title("How many labels do you need at a new monitoring site?",
                 fontsize=13.5, fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=9.5, loc="lower right", frameon=True, framealpha=0.92,
              edgecolor="#DDD", title="Transfer target", title_fontsize=9.5)

    # Diminishing returns hold clearly for the easier transfers; the hardest is still
    # improving at k=100, so the annotation is scoped rather than absolute.
    ax.annotate("diminishing returns\nafter ~20 labels",
                xy=(46, 0.868), xytext=(4.6, 0.985),
                fontsize=9.5, color="#333", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2,
                                connectionstyle="arc3,rad=-0.18"))
    ax.annotate("harder transfers\nkeep improving",
                xy=(97, 0.545), xytext=(21, 0.185),
                fontsize=9.5, color="#00734F", ha="left",
                arrowprops=dict(arrowstyle="->", color="#00734F", lw=1.2,
                                connectionstyle="arc3,rad=-0.28"))

    fig.tight_layout()
    Path(output_stem).parent.mkdir(parents=True, exist_ok=True)
    # ESE accepts JPEG, GIF or TIFF. Supply both a JPEG for upload and a TIFF.
    fig.savefig(f"{output_stem}.jpg", dpi=300, format="jpg",
                bbox_inches="tight", pil_kwargs={"quality": 95})
    fig.savefig(f"{output_stem}.tiff", dpi=300, format="tiff",
                bbox_inches="tight")
    fig.savefig(f"{output_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_stem}.jpg / .tiff / .png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate()
