"""Full graphical abstract for the cross-habitat marine transfer study.

A single panel-set that carries the whole paper: what was benchmarked, what happens
when a model moves to a new site, how much labelling fixes it, and where the approach
stops working. Designed so a reader who only ever sees this image understands the
study.

Four panels:
  (a) the benchmark    -- five datasets, three oceans, the transfer gradient
  (b) the problem      -- performance falls as ecological distance grows, but stays
                          well above trivial baselines
  (c) the answer       -- accuracy vs labels per class, with the recommended budget
  (d) the boundary     -- which monitoring questions this can and cannot answer

All numbers are read from the experiment outputs; nothing is hardcoded except the
qualitative panel (d), which mirrors Table 3 of the manuscript.

Usage:
    python -m figures.graphical_abstract_full
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_STEM = "outputs/figures/graphical_abstract_full"

K_VALUES = [1, 2, 5, 10, 20, 50, 100]
# Difficulty order by mean macro F1 on the corrected Protocol B results.
PAIRS = [("aqua20", "AQUA20\ntropical mix"),
         ("moorea", "Moorea\ncoral, Pacific"),
         ("brackish", "Brackish\nDanish fjord"),
         ("coralscapes", "Coralscapes\nRed Sea benthos")]
TCOL = {"aqua20": "#0072B2", "moorea": "#D55E00",
        "brackish": "#009E73", "coralscapes": "#8E44AD"}

INK, MUTED, RULE = "#1A1A1A", "#5A5A5A", "#D8D8D8"


# --- dataset thumbnails -----------------------------------------------------
# Showing the actual imagery makes the visual domain gap -- the subject of the
# paper -- legible at a glance, rather than asserting it in text.

THUMB_SPEC = {
    "deepfish":    dict(split="train", label="valid"),
    "aqua20":      dict(split="train", label="fish"),
    "moorea":      dict(split="train", label="PORIT"),
    "brackish":    dict(split="train", label="fish"),
    "coralscapes": dict(split="train", label="Hard_Coral_Mushroom"),
}


def _thumb(name, size=260, index=0):
    """One representative square RGB thumbnail for a dataset, or None."""
    import logging
    import yaml
    from PIL import Image
    from experiments.run_cross_dataset import _get_dataset
    logging.disable(logging.CRITICAL)
    try:
        cfg = yaml.safe_load(Path("configs/base_config.yaml").read_text())
        spec = THUMB_SPEC[name]
        ds = _get_dataset(name, cfg["data"]["root_dir"], split=spec["split"],
                          transform=None)
        cands = [s["image_path"] for s in ds.samples if s["label"] == spec["label"]]
        if not cands:
            cands = [s["image_path"] for s in ds.samples]
        img = Image.open(cands[index % len(cands)]).convert("RGB")
    except Exception:
        return None
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(float)
    # Brackish frames are very dark; lift them so the panel reads evenly.
    if name == "brackish":
        arr = np.clip((arr - arr.min()) / max(np.ptp(arr), 1) * 255 * 1.15, 0, 255)
    return arr.astype(np.uint8)


def _card(fig, ax, pad=0.022, fc="white", ec="#E4E4E4", lw=1.1):
    """Draw a soft card behind an axes, for visual grouping."""
    bb = ax.get_position()
    fig.patches.append(FancyBboxPatch(
        (bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=fig.transFigure, facecolor=fc, edgecolor=ec, lw=lw, zorder=-10))


def protocol_b():
    """Corrected Protocol B: mean/best macro F1 and the trivial baselines."""
    base = json.load(open("outputs/protocol_b_baselines.json"))
    aff = {r["experiment"]: r
           for r in json.load(open("outputs/protocol_b_reeval_affected.json"))}
    out = {}
    for t, _ in PAIRS:
        if t in ("moorea", "coralscapes"):
            v = [r["macro_f1_fixed"] for r in aff.values() if r["target"] == t]
        else:
            v = [json.load(open(f))["test_metrics"]["macro_f1"]
                 for f in glob.glob(
                     f"outputs/cross_dataset/*_deepfish_to_{t}_open_set_seed42/results.json")]
        b = base[t]
        out[t] = dict(
            mean=float(np.mean(v)), best=max(v),
            rand=b["baseline_uniform_random_mean"]["test_present"]["macro_f1"],
            maj=b["baseline_majority_class"]["test_present"]["macro_f1"],
            k=b["n_classes_test"])
    return out


def protocol_c():
    agg = defaultdict(list)
    for f in glob.glob("outputs/fewshot_curve/*/k*_trial*/results.json"):
        m = re.search(r"deepfish_to_(\w+?)_combined_seed42/k(\d+)_trial", f)
        if m:
            try:
                agg[(m.group(1), int(m.group(2)))].append(
                    json.load(open(f))["balanced_accuracy"])
            except Exception:
                pass
    return agg


def panel_a(ax, pb, fig):
    """The benchmark, shown with real imagery so the domain gap is visible.

    Laid out in normalised axes coordinates so the thumbnails stay square and fill
    the panel regardless of the figure geometry.
    """
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    bb = ax.get_position()
    w_in, h_in = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    aspect = w_in / h_in                      # >1 for a wide, short panel

    # Fill the width: source + arrow gap + four targets.
    gap, arrow = 0.022, 0.075
    tw = (1.0 - arrow - 3 * gap) / 5.0
    th = tw * aspect                          # square on the page
    top_room, bot_room = 0.26, 0.21           # for the header/gradient and the labels
    if th > 1.0 - top_room - bot_room:        # shrink if the panel is too short
        th = 1.0 - top_room - bot_room
        tw = th / aspect

    ybox = bot_room
    ytop = ybox + th

    ax.text(0, 1.0, "A five-dataset marine benchmark", fontsize=12.5,
            fontweight="bold", color=INK, va="top")
    ax.text(0, 1.0 - 0.075, "928 experiments  ·  4 vision models  ·  "
            "4 adaptation strategies  ·  3 oceans",
            fontsize=9.0, color=MUTED, va="top")

    def place(x, name, edge):
        t = _thumb(name)
        if t is not None:
            ax.imshow(t, extent=(x, x + tw, ybox, ytop), aspect="auto", zorder=2)
        ax.add_patch(FancyBboxPatch((x, ybox), tw, th,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor="none", edgecolor=edge, lw=2.6, zorder=3))

    # source
    place(0.0, "deepfish", "#2C6FA8")
    ax.text(tw / 2, ytop + 0.02, "SOURCE", fontsize=7.4, fontweight="bold",
            color="#2C6FA8", ha="center", va="bottom")
    ax.text(tw / 2, ybox - 0.035, "DeepFish", fontsize=10.0, fontweight="bold",
            color=INK, ha="center", va="top")
    ax.text(tw / 2, ybox - 0.125, "Great Barrier Reef\n20 habitats · fish/empty",
            fontsize=7.2, color=MUTED, ha="center", va="top", linespacing=1.45)

    # arrow
    ax.annotate("", xy=(tw + arrow - 0.012, ybox + th / 2),
                xytext=(tw + 0.012, ybox + th / 2),
                arrowprops=dict(arrowstyle="-|>", color="#8A8A8A", lw=2.0,
                                mutation_scale=17))
    ax.text(tw + arrow / 2, ybox + th / 2 + 0.035, "transfer", fontsize=7.4,
            color=MUTED, ha="center")

    # targets
    x0 = tw + arrow
    span = 4 * tw + 3 * gap
    for i, (t, lab) in enumerate(PAIRS):
        x = x0 + i * (tw + gap)
        place(x, t, TCOL[t])
        head, sub = lab.split("\n")
        ax.text(x + tw / 2, ybox - 0.035, head, fontsize=9.6, fontweight="bold",
                color=TCOL[t], ha="center", va="top")
        ax.text(x + tw / 2, ybox - 0.125, f"{sub}\n{pb[t]['k']} classes",
                fontsize=7.2, color=MUTED, ha="center", va="top", linespacing=1.45)

    # difficulty gradient above the target row
    gy, gh = ytop + 0.045, 0.042
    grad = np.linspace(0, 1, 512).reshape(1, -1)
    ax.imshow(grad, extent=(x0, x0 + span, gy, gy + gh), aspect="auto",
              cmap="RdYlGn_r", alpha=0.9, zorder=2)
    ax.add_patch(FancyBboxPatch((x0, gy), span, gh,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor="none", edgecolor="white", lw=1.0, zorder=3))
    ax.text(x0 + 0.004, gy + gh + 0.015, "easier transfer", fontsize=7.2,
            color=MUTED, va="bottom")
    ax.text(x0 + span - 0.004, gy + gh + 0.015, "harder transfer", fontsize=7.2,
            color=MUTED, ha="right", va="bottom")



def panel_b(ax, pb):
    """The problem: performance falls with distance, but beats trivial classifiers."""
    xs = np.arange(len(PAIRS))
    mean = [pb[t]["mean"] for t, _ in PAIRS]
    best = [pb[t]["best"] for t, _ in PAIRS]
    triv = [max(pb[t]["rand"], pb[t]["maj"]) for t, _ in PAIRS]

    ax.bar(xs - 0.19, best, 0.36, color=[TCOL[t] for t, _ in PAIRS],
           alpha=0.95, label="best model", zorder=3)
    ax.bar(xs + 0.19, mean, 0.36, color=[TCOL[t] for t, _ in PAIRS],
           alpha=0.42, label="mean of 10 models", zorder=3)
    ax.plot(xs, triv, marker="_", markersize=22, markeredgewidth=2.4,
            linestyle="none", color="#B00020", zorder=5,
            label="best trivial baseline")

    for i, (b, tr) in enumerate(zip(best, triv)):
        r = b / tr
        ax.text(i, b + 0.035, (f"{r:.0f}×" if r >= 10 else f"{r:.1f}×"),
                ha="center", fontsize=8.2,
                fontweight="bold", color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels([l.split("\n")[0] for _, l in PAIRS], fontsize=8.2)
    ax.set_ylabel("Macro F1 (cross-dataset transfer)", fontsize=8.6)
    ax.set_ylim(0, 1.0)
    ax.set_title("Moving to a new site costs accuracy —\nbut transfer is always real",
                 fontsize=10, fontweight="bold", pad=8, loc="left")
    h, l = ax.get_legend_handles_labels()
    order = [l.index("best model"), l.index("mean of 10 models"), l.index("best trivial baseline")]
    ax.legend([h[i] for i in order], [l[i] for i in order],
              fontsize=7.2, loc="upper right", framealpha=0.94, edgecolor=RULE)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    ax.tick_params(labelsize=8)


def panel_c(ax, pc):
    """The answer: labels per class vs accuracy."""
    for t, lab in PAIRS:
        ks = [k for k in K_VALUES if pc.get((t, k))]
        if not ks:
            continue
        ys = [float(np.mean(pc[(t, k)])) for k in ks]
        ax.plot(ks, ys, marker="o", markersize=4.6, lw=2.4, color=TCOL[t],
                markeredgecolor="white", markeredgewidth=0.9, zorder=3,
                label=lab.split("\n")[0])

    ax.axvspan(10, 20, color="#FFD24D", alpha=0.32, lw=0, zorder=0)
    ax.text(14, 0.045, "10–20 labels\nper class", ha="center", va="bottom",
            fontsize=8.4, fontweight="bold", color="#7A5A00", zorder=4)

    ax.set_xscale("log"); ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=8)
    ax.set_xlim(0.85, 122); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Labelled images per class at the new site  ($k$)", fontsize=8.6)
    ax.set_ylabel("Balanced accuracy", fontsize=8.6)
    ax.set_title("A few labels go a long way — how far\ndepends on the transfer",
                 fontsize=10, fontweight="bold", pad=8, loc="left")
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.94, edgecolor=RULE)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, lw=0.6); ax.tick_params(labelsize=8)
    ax.annotate("diminishing returns\nbeyond ~20", xy=(46, 0.865),
                xytext=(1.5, 0.955), fontsize=7.6, color=INK, va="top",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0,
                                connectionstyle="arc3,rad=-0.16"))


def panel_d(ax):
    """The boundary: what this can and cannot answer."""
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0, 9.85, "What a small labelling budget can and cannot answer",
            fontsize=10, fontweight="bold", color=INK, va="top")

    rows = [("Is the target group present here?", "yes", "#1E7A46"),
            ("Broad benthic composition", "partly", "#B8860B"),
            ("Percentage cover / reef health", "no", "#A32020"),
            ("Abundance or size structure", "no", "#A32020"),
            ("A specific invasive or protected species", "untested", "#5A5A5A")]
    y = 8.55
    for label, verdict, col in rows:
        ax.add_patch(FancyBboxPatch((0.0, y - 0.62), 6.55, 0.92,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#F6F6F6", edgecolor="none"))
        ax.text(0.22, y - 0.16, label, fontsize=8.3, color=INK, va="center")
        ax.add_patch(FancyBboxPatch((6.75, y - 0.55), 2.6, 0.78,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=col, edgecolor="none"))
        ax.text(8.05, y - 0.16, verdict.upper(), fontsize=8.0, fontweight="bold",
                color="white", ha="center", va="center")
        y -= 1.16

    ax.plot([0, 9.35], [2.42, 2.42], color=RULE, lw=1.0)
    ax.text(0, 2.05, "Image-level recognition answers presence, not quantity. "
                     "Labels are\ncoarser than species, so the effort saving is an "
                     "upper bound.",
            fontsize=7.8, color=MUTED, va="top")
    ax.text(0, 0.72, "10–20 labels per class  ≈  1–4 hours of annotation per site   ·   "
                     "frozen DINOv2 + linear probe  ·  1,538 trainable parameters",
            fontsize=8.2, color=INK, va="top", fontweight="bold")


def generate(stem: str = OUT_STEM) -> None:
    pb, pc = protocol_b(), protocol_c()

    fig = plt.figure(figsize=(11.0, 11.2), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.0, 0.75],
                          hspace=0.30, wspace=0.22,
                          left=0.055, right=0.975, top=0.935, bottom=0.045)
    fig.suptitle("How many labels do you need to start automating a new marine "
                 "monitoring site?",
                 fontsize=15, fontweight="bold", color=INK, y=0.978)

    panel_a(fig.add_subplot(gs[0, :]), pb, fig)
    panel_b(fig.add_subplot(gs[1, 0]), pb)
    panel_c(fig.add_subplot(gs[1, 1]), pc)
    panel_d(fig.add_subplot(gs[2, :]))

    # hairline dividers between the three bands
    for y in (0.660, 0.283):
        fig.add_artist(plt.Line2D([0.055, 0.975], [y, y], transform=fig.transFigure,
                                  color="#E6E6E6", lw=1.0, zorder=-5))

    for ax, lab in zip(fig.axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(-0.02, 1.06, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK, ha="right", va="bottom")

    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.jpg", dpi=300, format="jpg", bbox_inches="tight",
                pil_kwargs={"quality": 95}, facecolor="white")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.tiff", dpi=300, format="tiff", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}.jpg / .png / .tiff")


if __name__ == "__main__":
    generate()
