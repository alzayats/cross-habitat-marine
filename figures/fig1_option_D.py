"""Figure 1 Option D: Geographic Story + Compact Experimental Footer.

Map-dominant square design:
  (a) Real world map (cartopy Blue Marble) with dataset pins — takes ~80%
  Footer ribbon: models × adaptations → 3 protocols as inline compact strip
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from figures.plot_utils import (
    save_figure, set_journal_style, DOUBLE_COLUMN_WIDTH, add_panel_label,
)

logger = logging.getLogger(__name__)

DATASETS = [
    {
        "name": "DeepFish",
        "lon": 140.0, "lat": -22.0,
        "path": "./datasets/DeepFish/Classification/7117/valid/7117_Caranx_sexfasciatus_juvenile_f000000.jpg",
        "color": "#1565C0",
        "info": "2 cls | 40K",
        "role": "Source",
        "label_offset": (0, -32),
    },
    {
        "name": "Moorea",
        "lon": -140.0, "lat": -17.5,
        "path": "./datasets/Moorea_Labeled_Corals/knb-lter-mcr.5006.3/MCR_LTER_ComputerVision_LabeledCorals_2009_20120523/2009/mcr_lter4_out10m_pole3-4_qu2_20090407.jpg",
        "color": "#6A1B9A",
        "info": "9 cls | 2.3K",
        "role": "Target",
        "label_offset": (0, -32),
    },
    {
        "name": "Coralscapes",
        "lon": 38.0, "lat": 22.0,
        "path": "./datasets/Coralscapes_dataset/train/images/001193.jpg",
        "color": "#E65100",
        "info": "14 cls | 4.5K",
        "role": "Target",
        "label_offset": (0, -32),
    },
    {
        "name": "AQUA20",
        "lon": 80.0, "lat": 5.0,
        "path": "./datasets/AQUA20_dataset/train_images/coral/image_1000.png",
        "color": "#2E7D32",
        "info": "20 cls | 20K",
        "role": "Target",
        "label_offset": (0, -32),
    },
    {
        "name": "Brackish",
        "lon": 8.7, "lat": 56.6,
        "path": "./datasets/Brackish_dataset/frames/2019-03-07_08-57-30to2019-03-07_08-57-40_1-0060.png",
        "color": "#00695C",
        "info": "6 cls | 14K",
        "role": "Target",
        "label_offset": (0, -35),
    },
]


def _load_thumb(path: str, size: int = 120) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
    return np.array(img)


def _draw_real_map(ax):
    """Draw a real world map using cartopy with NASA Blue Marble background."""
    ax.set_extent([-170, 170, -55, 78], crs=ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "coastline", "50m"),
        linewidth=0.3, edgecolor="#444444", facecolor="none", zorder=2,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", "110m"),
        linewidth=0.15, edgecolor="#777777", facecolor="none", zorder=2,
    )
    ax.gridlines(draw_labels=False, linewidth=0.2, color="white",
                 alpha=0.3, linestyle="-", zorder=1)
    ax.plot([-180, 180], [23.44, 23.44], color="#FFD54F", lw=0.7,
            ls="--", alpha=0.6, transform=ccrs.PlateCarree(), zorder=1)
    ax.plot([-180, 180], [-23.44, -23.44], color="#FFD54F", lw=0.7,
            ls="--", alpha=0.6, transform=ccrs.PlateCarree(), zorder=1)
    ax.spines["geo"].set_linewidth(0.5)
    ax.spines["geo"].set_edgecolor("#555555")


def _draw_footer(ax):
    """Draw compact footer with models × adaptations → protocols."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.0)
    ax.axis("off")

    # --- (b) label ---
    ax.text(0.0, 0.95, "(b)", fontsize=10, fontweight="bold", color="#333",
            va="top", ha="right")

    # --- Models (pill badges) ---
    models = [
        ("DINOv2-B", "#0072B2"),
        ("CLIP-B", "#E69F00"),
        ("RN50-IN", "#999999"),
        ("EffNet-B4", "#666666"),
    ]
    for i, (name, color) in enumerate(models):
        y = 0.70 - i * 0.22
        pill = FancyBboxPatch(
            (0.10, y - 0.06), 1.25, 0.17,
            boxstyle="round,pad=0.03", fc="white", ec=color, lw=0.8,
        )
        ax.add_patch(pill)
        ax.text(0.72, y + 0.025, name, fontsize=5.5, fontweight="bold",
                color=color, ha="center", va="center")

    # --- Cross symbol ---
    ax.text(1.60, 0.42, "\u00d7", fontsize=14, ha="center", va="center",
            color="#888", fontweight="bold")

    # --- Adaptations (pill badges) ---
    adapts = ["Linear Probe", "LoRA (r=4)", "VPT Deep", "Full FT"]
    for i, name in enumerate(adapts):
        y = 0.70 - i * 0.22
        pill = FancyBboxPatch(
            (1.85, y - 0.06), 1.25, 0.17,
            boxstyle="round,pad=0.03", fc="#FFF8E1", ec="#E65100", lw=0.8,
        )
        ax.add_patch(pill)
        ax.text(2.47, y + 0.025, name, fontsize=5.5, fontweight="bold",
                color="#E65100", ha="center", va="center")

    # --- Arrow to protocols ---
    ax.annotate("", xy=(3.60, 0.42), xytext=(3.30, 0.42),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))

    # --- (c) label ---
    ax.text(3.70, 0.95, "(c)", fontsize=10, fontweight="bold", color="#333",
            va="top", ha="right")

    # --- 3 Protocol cards ---
    protocols = [
        ("A", "Within-Habitat", "#6A1B9A", "#F3E5F5",
         "LOHO | 20 splits\n9 combos \u00d7 3 seeds"),
        ("B", "Cross-Dataset", "#1565C0", "#E3F2FD",
         "4 targets | open-set\ndomain gap analysis"),
        ("C", "Few-Shot", "#F57F17", "#FFFDE7",
         "k\u2208{0..100}\n3 targets \u00d7 5 trials"),
    ]

    card_w = 1.90
    gap = 0.12
    x_start = 3.78
    for i, (letter, title, ec, fc, desc) in enumerate(protocols):
        x = x_start + i * (card_w + gap)
        box = FancyBboxPatch(
            (x, 0.0), card_w, 0.82,
            boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.0,
        )
        ax.add_patch(box)
        ax.text(x + card_w / 2, 0.66, f"{letter}: {title}",
                fontsize=5.5, fontweight="bold", color=ec, ha="center", va="center")
        ax.plot([x + 0.08, x + card_w - 0.08], [0.54, 0.54],
                color=ec, lw=0.4, alpha=0.5)
        ax.text(x + card_w / 2, 0.28, desc, fontsize=4.8, color="#444",
                ha="center", va="center", linespacing=1.4)


def generate_fig1_D(output_path: Optional[str] = None) -> plt.Figure:
    """Option D: Map-dominant square figure with compact footer."""
    set_journal_style()

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 4.2))

    # Map takes nearly full figure
    ax_map = fig.add_axes([0.01, 0.08, 0.98, 0.91],
                          projection=ccrs.PlateCarree())
    _draw_real_map(ax_map)

    transform = ccrs.PlateCarree()
    for ds in DATASETS:
        img = _load_thumb(ds["path"], size=110)
        im = OffsetImage(img, zoom=0.40)
        border_width = 3.0 if ds["role"] == "Source" else 2.0
        ab = AnnotationBbox(
            im, (ds["lon"], ds["lat"]),
            xycoords=transform._as_mpl_transform(ax_map),
            frameon=True,
            bboxprops=dict(
                edgecolor=ds["color"], lw=border_width,
                facecolor="white", boxstyle="round,pad=0.08",
            ),
            zorder=5,
        )
        ax_map.add_artist(ab)

        ax_map.plot(ds["lon"], ds["lat"], "o", color=ds["color"],
                    ms=3, zorder=6, mec="white", mew=0.5,
                    transform=transform)

        offset_x, offset_y = ds["label_offset"]
        role_tag = " (Source)" if ds["role"] == "Source" else ""
        va = "top" if offset_y < 0 else "bottom"
        ax_map.annotate(
            f"{ds['name']}{role_tag}\n{ds['info']}",
            xy=(ds["lon"], ds["lat"]),
            xycoords=transform._as_mpl_transform(ax_map),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=6, fontweight="bold", color=ds["color"],
            ha="center", va=va, zorder=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                      alpha=0.85),
        )

    # Transfer arrows
    src = DATASETS[0]
    for tgt in DATASETS[1:]:
        ax_map.annotate(
            "", xy=(tgt["lon"], tgt["lat"]),
            xycoords=transform._as_mpl_transform(ax_map),
            xytext=(src["lon"], src["lat"]),
            textcoords=transform._as_mpl_transform(ax_map),
            arrowprops=dict(
                arrowstyle="-|>", color=tgt["color"],
                lw=1.5, alpha=0.7,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=2,
        )

    ax_map.text(0.50, 0.02, "Tropical \u2192 Temperate transfer",
                transform=ax_map.transAxes, ha="center", va="bottom",
                fontsize=6, fontstyle="italic", color="#555",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7))

    add_panel_label(ax_map, "(a)", x=0.01, y=1.01, fontsize=10)

    # === Footer overlaps into Robinson's curved bottom edge ===
    ax_footer = fig.add_axes([0.01, -0.02, 0.98, 0.18])
    _draw_footer(ax_footer)

    if output_path:
        from pathlib import Path
        base = Path(output_path).with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        for fmt in ["pdf", "png"]:
            fig.savefig(str(base) + f".{fmt}", dpi=600,
                        bbox_inches="tight", pad_inches=0.02)
    return fig


if __name__ == "__main__":
    generate_fig1_D("outputs/figures/fig1_option_D")
