"""Figure 2: Dataset sample grid showing diverse images from each dataset."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageEnhance

from figures.plot_utils import save_figure, set_journal_style, DOUBLE_COLUMN_WIDTH

logger = logging.getLogger(__name__)

# Row accent colors (left border + label background tint)
ROW_COLORS = {
    "DeepFish":      "#1565C0",
    "AQUA20":        "#2E7D32",
    "Moorea Corals": "#E65100",
    "Coralscapes":   "#6A1B9A",
    "Brackish":      "#37474F",
}



def _pretty_label(raw: str) -> str:
    """Render a dataset class code as a short human-readable label."""
    MOOREA = {
        "CCA": "CCA", "SAND": "Sand", "TURF": "Turf algae",
        "PORIT": "Porites", "MACRO": "Macroalgae", "POCILL": "Pocillopora",
        "MONTI": "Montipora", "PAVON": "Pavona",
    }
    if raw in MOOREA:
        return MOOREA[raw]
    return raw.replace("_", " ").replace("-", " ")


def _sample_diverse_images(
    dir_path: Path,
    n_samples: int,
    strategy: str = "diverse_subdirs",
    seed: int = 42,
) -> List[Tuple[Union[Path, Image.Image], str]]:
    """Sample diverse images from a dataset directory."""
    rng = random.Random(seed)

    if strategy == "deepfish_habitats":
        habitat_dirs = sorted([
            d for d in dir_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        chosen = rng.sample(habitat_dirs, min(n_samples, len(habitat_dirs)))
        results = []
        for hab_dir in chosen:
            valid_dir = hab_dir / "valid"
            if valid_dir.exists():
                imgs = list(valid_dir.glob("*.jpg"))
            else:
                imgs = list(hab_dir.rglob("*.jpg"))
            if imgs:
                results.append((rng.choice(imgs), f"Habitat {hab_dir.name} (fish)"))
        return results

    elif strategy == "class_subdirs":
        class_dirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
        if not class_dirs:
            imgs = list(dir_path.rglob("*.jpg")) + list(dir_path.rglob("*.png"))
            chosen = rng.sample(imgs, min(n_samples, len(imgs)))
            return [(img, _pretty_label(img.parent.name)) for img in chosen]
        chosen = rng.sample(class_dirs, min(n_samples, len(class_dirs)))
        results = []
        for cls_dir in chosen:
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            if imgs:
                results.append((rng.choice(imgs), _pretty_label(cls_dir.name)))
        return results

    elif strategy == "video_extract":
        species_dirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
        chosen = rng.sample(species_dirs, min(n_samples, len(species_dirs)))
        results = []
        for sp_dir in chosen:
            videos = list(sp_dir.glob("*.avi"))
            if not videos:
                continue
            video_path = rng.choice(videos)
            cap = cv2.VideoCapture(str(video_path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid = max(total // 2, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    results.append((img, _pretty_label(sp_dir.name)))
            finally:
                cap.release()
        return results

    elif strategy in ("moorea_classes", "coralscapes_classes"):
        # Labels for these two datasets are derived (dominant annotation point /
        # dominant mask class) rather than encoded in the directory tree, so we use
        # the project loaders to recover them. One image per class, most frequent
        # classes first, so every row of the figure carries class information
        # so the figure is consistent across rows.
        import yaml
        from experiments.run_cross_dataset import _get_dataset
        name = "moorea" if strategy == "moorea_classes" else "coralscapes"
        cfg = yaml.safe_load(Path("configs/base_config.yaml").read_text())
        ds = _get_dataset(name, cfg["data"]["root_dir"], split="train", transform=None)
        by_class: dict = {}
        for smp in ds.samples:
            by_class.setdefault(smp["label"], []).append(smp["image_path"])
        ordered = sorted(by_class.items(), key=lambda kv: -len(kv[1]))[:n_samples]
        return [(rng.choice(paths), _pretty_label(lbl)) for lbl, paths in ordered]

    elif strategy == "flat_random":
        imgs = list(dir_path.rglob("*.jpg")) + list(dir_path.rglob("*.png"))
        chosen = rng.sample(imgs, min(n_samples, len(imgs)))
        return [(img, "") for img in chosen]

    return []


DATASET_CONFIGS = [
    {
        "name": "DeepFish",
        "subdir": "Classification",
        "strategy": "deepfish_habitats",
        "description": "20 habitats, fish/empty",
    },
    {
        "name": "AQUA20",
        "subdir": "train_images",
        "strategy": "class_subdirs",
        "description": "20 coarse categories",
    },
    {
        "name": "Moorea Corals",
        "subdir": "knb-lter-mcr.5006.3/MCR_LTER_ComputerVision_LabeledCorals_2009_20120523/2009",
        "strategy": "moorea_classes",
        "description": "8 benthic classes",
    },
    {
        "name": "Coralscapes",
        "subdir": "train/images",
        "strategy": "coralscapes_classes",
        "description": "21 benthic classes",
    },
    {
        "name": "Brackish",
        "subdir": "dataset/videos",
        "strategy": "video_extract",
        "description": "6 coarse categories",
    },
]


def _load_and_prepare_image(img_or_path, enhance_dark: bool = False):
    """Load, center-crop to square, resize, optionally brighten dark images."""
    if isinstance(img_or_path, Image.Image):
        img = img_or_path.convert("RGB")
    else:
        img = Image.open(img_or_path).convert("RGB")

    # Center crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((224, 224), Image.LANCZOS)

    # Brighten dark images (e.g., Brackish underwater footage)
    if enhance_dark:
        arr = np.array(img)
        mean_brightness = arr.mean()
        if mean_brightness < 80:
            factor = min(80 / max(mean_brightness, 1), 2.0)
            img = ImageEnhance.Brightness(img).enhance(factor)

    return np.array(img)


def generate_fig2(
    dataset_dirs: Dict[str, str],
    n_samples: int = 5,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Generate dataset sample grid."""
    set_journal_style()

    configs = []
    for cfg in DATASET_CONFIGS:
        root = dataset_dirs.get(cfg["name"])
        if root is None:
            continue
        full_path = Path(root) / cfg["subdir"]
        if not full_path.exists():
            logger.warning("Path not found: %s", full_path)
            continue
        configs.append({**cfg, "full_path": full_path})

    n_datasets = len(configs)
    if n_datasets == 0:
        logger.error("No valid dataset paths found")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No datasets found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        return fig

    row_height = 1.45
    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, row_height * n_datasets + 0.3))

    gs = gridspec.GridSpec(
        n_datasets, n_samples + 1,
        width_ratios=[0.32] + [1.0] * n_samples,
        hspace=0.12, wspace=0.06,
        left=0.01, right=0.99, top=0.995, bottom=0.01,
    )

    for row, cfg in enumerate(configs):
        samples = _sample_diverse_images(
            cfg["full_path"], n_samples, cfg["strategy"]
        )

        accent = ROW_COLORS.get(cfg["name"], "#555")
        is_dark_dataset = cfg["name"] == "Brackish"

        # --- Row label ---
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis("off")

        # Colored left accent bar (drawn as a thin rectangle in axes coords)
        from matplotlib.patches import Rectangle
        bar = Rectangle(
            (0.90, 0.08), 0.04, 0.84,
            transform=ax_label.transAxes, clip_on=False,
            facecolor=accent, edgecolor="none",
        )
        ax_label.add_patch(bar)

        ax_label.text(
            0.82, 0.55, cfg["name"],
            ha="right", va="center",
            fontsize=9, fontweight="bold", color=accent,
            transform=ax_label.transAxes,
        )
        ax_label.text(
            0.82, 0.30, cfg["description"],
            ha="right", va="center",
            fontsize=7, color="#555", fontstyle="italic",
            transform=ax_label.transAxes,
        )

        # --- Image cells ---
        for col in range(n_samples):
            ax = fig.add_subplot(gs[row, col + 1])

            if col < len(samples):
                img_or_path, label = samples[col]
                try:
                    arr = _load_and_prepare_image(
                        img_or_path, enhance_dark=is_dark_dataset)
                    ax.imshow(arr)

                    if label:
                        # Label below image with background
                        ax.text(
                            0.5, -0.01, label,
                            ha="center", va="top",
                            fontsize=6.5, color="#333",
                            transform=ax.transAxes,
                            bbox=dict(boxstyle="square,pad=0.1",
                                      fc="white", ec="none", alpha=0.7),
                        )
                except Exception as e:
                    logger.warning("Failed to load %s: %s", img_or_path, e)
                    ax.text(0.5, 0.5, "Error", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="red")

            ax.axis("off")

            # Visible thin border around each image
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#999")
                spine.set_linewidth(0.6)


    if output_path:
        save_figure(fig, output_path)

    return fig


if __name__ == "__main__":
    import yaml
    root = Path(yaml.safe_load(
        Path("configs/base_config.yaml").read_text())["data"]["root_dir"])
    datasets = {
        "DeepFish": str(root / "DeepFish"),
        "AQUA20": str(root / "AQUA20_dataset"),
        "Moorea Corals": str(root / "Moorea_Labeled_Corals"),
        "Coralscapes": str(root / "Coralscapes_dataset"),
        "Brackish": str(root / "Brackish_dataset"),
    }
    generate_fig2(datasets, output_path="outputs/figures/fig2_dataset_samples")
