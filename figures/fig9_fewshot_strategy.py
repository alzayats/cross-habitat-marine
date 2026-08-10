"""Figure 12: Few-Shot Strategy Comparison (Protocol C).

(a) Combined vs Sequential strategy comparison (bar chart).
(b) Samples-to-threshold: k needed to reach X% of full-data performance.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from figures.plot_utils import (
    DOUBLE_COLUMN_WIDTH,
    FONT_SIZE_ANNOTATION,
    add_panel_label,
    get_model_color,
    save_figure,
    set_journal_style,
)

logger = logging.getLogger(__name__)

# Same pairs/combos as fig11
FEWSHOT_PAIRS = [
    ("deepfish", "aqua20"),
    ("deepfish", "moorea"),
    ("deepfish", "brackish"),
]

PAIR_LABELS = {
    ("deepfish", "aqua20"): "DF → AQ20",
    ("deepfish", "moorea"): "DF → Moorea",
    ("deepfish", "brackish"): "DF → Brack",
}

FEWSHOT_COMBOS = [
    ("dinov2_base", "linear_probe"),
    ("dinov2_base", "lora_r4"),
    ("clip_base", "linear_probe"),
    ("clip_base", "lora_r4"),
    ("resnet50_imagenet", "linear_probe"),
    ("efficientnet_b4", "linear_probe"),
]

MODEL_SHORT = {
    "dinov2_base": "DINOv2-B", "clip_base": "CLIP-B",
    "resnet50_imagenet": "RN50-IN", "efficientnet_b4": "EffNet-B4",
}
ADAPT_SHORT = {
    "linear_probe": "LP", "lora_r4": "LoRA",
}

# Threshold for "samples to threshold" panel
THRESHOLD_PCT = 0.80  # 80% of full-data performance

K_VALUES = [0, 1, 2, 5, 10, 20, 50, 100]


def load_fewshot_results(results_dir: str) -> Dict[str, Dict]:
    """Load Protocol C fewshot_curve.json files."""
    results = {}
    for path in sorted(Path(results_dir).rglob("fewshot_curve.json")):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            continue
        key = r.get("experiment", path.parent.name)
        results[key] = r
    return results


def generate_fig12(
    results_dir: str = "./outputs/fewshot_curve",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate Figure 12: few-shot strategy comparison."""
    set_journal_style()
    raw_results = load_fewshot_results(results_dir)

    use_placeholder = len(raw_results) < 2
    if use_placeholder:
        logger.info("Insufficient data — using placeholder for Fig 12")
        curve_data = _make_placeholder_data()
    else:
        curve_data = _extract_curve_data(raw_results)

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 3.5))
    gs = gridspec.GridSpec(
        1, 2, wspace=0.35, width_ratios=[1.2, 1.0],
        left=0.10, right=0.96, top=0.88, bottom=0.18,
    )

    ax_bars = fig.add_subplot(gs[0, 0])
    _draw_kshot_bars(curve_data, ax_bars)
    add_panel_label(ax_bars, "(a)", x=-0.12)

    ax_thresh = fig.add_subplot(gs[0, 1])
    _draw_samples_to_threshold(curve_data, ax_thresh)
    add_panel_label(ax_thresh, "(b)", x=-0.10)

    if output_path:
        save_figure(fig, output_path)
    return fig


def _draw_kshot_bars(curve_data: Dict, ax: plt.Axes) -> None:
    """Grouped bars: F1 at k=10 and k=100 per combo, averaged across pairs."""
    k_show = [10, 100]
    combos = FEWSHOT_COMBOS
    n_combos = len(combos)
    n_ks = len(k_show)
    x = np.arange(n_combos)
    width = 0.8 / n_ks

    bar_tops: list = []
    for ki, k in enumerate(k_show):
        vals = []
        errs = []
        for model, adapt in combos:
            combo_key = f"{model}_{adapt}"
            f1s = []
            for pair_key, pair_data in curve_data.items():
                data = pair_data.get(combo_key)
                if data is None:
                    continue
                ks = data["k_values"]
                if k in ks:
                    idx = ks.index(k)
                    f1s.append(data["means"][idx])
            vals.append(np.mean(f1s) if f1s else 0)
            errs.append(np.std(f1s) if len(f1s) > 1 else 0)

        color = "#0072B2" if ki == 0 else "#E69F00"
        ax.bar(x + ki * width - 0.4 + width / 2, vals, width * 0.9,
               yerr=errs, color=color, alpha=0.85, label=f"k={k}",
               edgecolor="white", linewidth=0.5,
               capsize=1.5, error_kw={"linewidth": 0.6, "alpha": 0.6})
        # The y-limit was previously hardcoded at 0.85, which clipped the upper
        # error bars. Track the extent so it can be derived from the data.
        bar_tops.extend(v + e for v, e in zip(vals, errs))

    combo_labels = [
        f"{MODEL_SHORT.get(m, m)} {ADAPT_SHORT.get(a, a)}"
        for m, a in combos
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, fontsize=6.5, rotation=25, ha="right")
    ax.set_ylabel("Balanced Acc. (mean across pairs)", fontsize=7)
    ax.set_title("Performance at k=10 vs k=100", fontsize=9,
                 fontweight="bold", pad=6)
    ax.set_ylim(0, max(bar_tops) * 1.10 if bar_tops else 0.85)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9, edgecolor="#DDD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def _draw_samples_to_threshold(curve_data: Dict, ax: plt.Axes) -> None:
    """Horizontal bars: k needed to reach THRESHOLD_PCT of max F1."""
    combos = FEWSHOT_COMBOS
    combo_labels = []
    k_needed = []
    reached = []
    colors = []

    for model, adapt in combos:
        combo_key = f"{model}_{adapt}"
        thresholds = []
        did_reach = []
        for pair_key, pair_data in curve_data.items():
            data = pair_data.get(combo_key)
            if data is None:
                continue
            ks = data["k_values"]
            means = data["means"]
            if not ks or not means:
                continue
            max_f1 = max(means)
            target = max_f1 * THRESHOLD_PCT
            found_k = ks[-1]
            hit = False
            for k_val, f1 in zip(ks, means):
                if f1 >= target:
                    found_k = k_val
                    hit = True
                    break
            thresholds.append(found_k)
            did_reach.append(hit)

        combo_labels.append(
            f"{MODEL_SHORT.get(model, model)} {ADAPT_SHORT.get(adapt, adapt)}"
        )
        k_needed.append(np.mean(thresholds) if thresholds else 100)
        reached.append(all(did_reach) if did_reach else False)
        colors.append(get_model_color(model))

    if not k_needed:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    y = np.arange(len(k_needed))
    bars = ax.barh(y, k_needed, color=colors, alpha=0.8,
                   edgecolor="white", height=0.55)

    for i, (k, hit) in enumerate(zip(k_needed, reached)):
        prefix = "" if hit else "\u2265"
        ax.text(k + 1, i, f"{prefix}k={k:.0f}", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(combo_labels, fontsize=6.5)
    ax.set_xlabel(f"k needed for {THRESHOLD_PCT:.0%} of max bal. acc.", fontsize=7)
    ax.set_title("Sample Efficiency", fontsize=9,
                 fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)


def _extract_curve_data(raw_results: Dict) -> Dict:
    """Extract mean/std per k from raw fewshot_curve.json results."""
    curve_data = {}
    for exp_name, result in raw_results.items():
        source = result.get("source", "")
        target = result.get("target", "")
        model = result.get("model", "")
        adapt = result.get("adaptation", "")
        pair_key = f"{source}_to_{target}"
        combo_key = f"{model}_{adapt}"

        agg = result.get("aggregated", {})
        k_values = []
        means = []
        stds = []

        for k_str, stats in sorted(agg.items(), key=lambda x: int(x[0])):
            k_values.append(int(k_str))
            means.append(stats.get("balanced_accuracy_mean", 0))
            stds.append(stats.get("balanced_accuracy_std", 0))

        if k_values:
            curve_data.setdefault(pair_key, {})[combo_key] = {
                "k_values": k_values,
                "means": means,
                "stds": stds,
            }

    return curve_data


def _make_placeholder_data() -> Dict:
    """Generate realistic placeholder few-shot curves."""
    rng = np.random.default_rng(42)
    curve_data = {}

    pair_difficulty = {
        "deepfish_to_aqua20": 0.0,
        "deepfish_to_moorea": -0.15,
        "deepfish_to_brackish": -0.25,
    }

    combo_params = {
        "dinov2_base_linear_probe":       {"zero_shot": 0.35, "plateau": 0.82, "rate": 0.4},
        "dinov2_base_lora_r4":            {"zero_shot": 0.30, "plateau": 0.78, "rate": 0.5},
        "clip_base_linear_probe":         {"zero_shot": 0.40, "plateau": 0.75, "rate": 0.35},
        "clip_base_lora_r4":              {"zero_shot": 0.32, "plateau": 0.72, "rate": 0.45},
        "resnet50_imagenet_linear_probe": {"zero_shot": 0.15, "plateau": 0.65, "rate": 0.3},
    }

    for pair_key, diff in pair_difficulty.items():
        pair_data = {}
        for combo_key, params in combo_params.items():
            ks = K_VALUES
            means = []
            stds = []
            for k in ks:
                if k == 0:
                    f1 = params["zero_shot"] + diff
                else:
                    progress = 1 - np.exp(-params["rate"] * k / 10)
                    f1 = params["zero_shot"] + (params["plateau"] - params["zero_shot"]) * progress + diff
                f1 = np.clip(f1 + rng.normal(0, 0.01), 0.05, 0.95)
                means.append(float(f1))
                stds.append(float(max(0.08 - 0.001 * k, 0.02)))

            pair_data[combo_key] = {
                "k_values": ks,
                "means": means,
                "stds": stds,
            }
        curve_data[pair_key] = pair_data

    return curve_data


if __name__ == "__main__":
    generate_fig12(output_path="outputs/figures/fig9_fewshot_strategy")
