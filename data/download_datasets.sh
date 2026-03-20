#!/usr/bin/env bash
# =============================================================================
# Download and verify datasets for cross-habitat marine species recognition
# Usage: bash data/download_datasets.sh [dataset_name|all|verify] [output_dir]
#
# Datasets:
#   1. DeepFish        - Fish classification/detection (20 habitats, GBR)
#   2. AQUA20          - Marine species classification (20 classes)
#   3. COTS GBR        - Crown-of-Thorns Starfish crops (binary)
#   4. Moorea          - Benthic point annotations (French Polynesia)
#   5. Coralscapes     - Benthic segmentation (Red Sea)
#   6. Brackish        - Underwater detection (temperate Denmark)
# =============================================================================

set -euo pipefail

OUTPUT_DIR="${2:-./datasets}"
DATASET="${1:-all}"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# DeepFish (already downloaded)
# =============================================================================
download_deepfish() {
    echo "=========================================="
    echo "DeepFish dataset"
    echo "=========================================="
    local dest="$OUTPUT_DIR/DeepFish"
    if [ -d "$dest/Classification" ]; then
        echo "DeepFish already present at $dest"
        echo "Verifying..."
        local n_habitats
        n_habitats=$(find "$dest/Classification" -maxdepth 1 -type d | wc -l)
        n_habitats=$((n_habitats - 1))
        echo "  Habitats found: $n_habitats (expected: 20)"
        local n_images
        n_images=$(find "$dest/Classification" -name "*.jpg" | wc -l)
        echo "  Images found: $n_images (expected: ~39,766)"
        return
    fi

    echo "Download DeepFish from:"
    echo "  https://alzayats.github.io/DeepFish/"
    echo "  or: git clone https://github.com/alzayats/DeepFish.git"
    echo ""
    echo "Expected structure after extraction:"
    echo "  $dest/Classification/{habitat_id}/empty/*.jpg"
    echo "  $dest/Classification/{habitat_id}/valid/*.jpg"
    echo "  (20 habitat directories with numeric IDs)"
}

# =============================================================================
# AQUA20
# =============================================================================
download_aqua20() {
    echo "=========================================="
    echo "AQUA20 dataset"
    echo "=========================================="
    local dest="$OUTPUT_DIR/AQUA20_dataset"
    if [ -d "$dest/train_images" ]; then
        echo "AQUA20 already present at $dest"
        echo "Verifying..."
        local n_train n_test
        n_train=$(find "$dest/train_images" -name "*.jpg" -o -name "*.png" | wc -l)
        n_test=$(find "$dest/test_images" -name "*.jpg" -o -name "*.png" | wc -l)
        echo "  Train images: $n_train (expected: ~6,559)"
        echo "  Test images: $n_test (expected: ~1,612)"
        echo "  Classes: $(ls "$dest/train_images" | wc -l) (expected: 20)"
        return
    fi

    echo "Download AQUA20 from Kaggle:"
    echo "  pip install kaggle"
    echo "  kaggle datasets download -d slavkoprytula/aquarium-data-cots -p $dest"
    echo ""
    echo "Or from the original source paper."
    echo ""
    echo "Expected structure:"
    echo "  $dest/train_images/{class_name}/*.jpg"
    echo "  $dest/test_images/{class_name}/*.jpg"
    echo "  $dest/labels.txt"
}

# =============================================================================
# COTS GBR (Crown-of-Thorns Starfish — full competition data)
# =============================================================================
download_cots_gbr() {
    echo "=========================================="
    echo "COTS GBR dataset (full competition)"
    echo "=========================================="
    local dest="$OUTPUT_DIR/Cots_GBR_dataset"
    if [ -f "$dest/train.csv" ] && [ -d "$dest/train_images" ]; then
        echo "COTS GBR already present at $dest"
        echo "Verifying..."
        local n_videos n_frames n_rows
        n_videos=$(ls -d "$dest/train_images/video_"* 2>/dev/null | wc -l)
        n_frames=$(find "$dest/train_images" -name "*.jpg" | wc -l)
        n_rows=$(tail -n +2 "$dest/train.csv" | wc -l)
        echo "  Videos: $n_videos (expected: 3)"
        echo "  Frames: $n_frames (expected: ~23,501)"
        echo "  CSV rows: $n_rows (expected: 23,501)"
        return
    fi

    echo "COTS GBR data from the Kaggle TensorFlow COTS competition."
    echo "  kaggle competitions download -c tensorflow-great-barrier-reef"
    echo ""
    echo "Expected structure:"
    echo "  $dest/train.csv"
    echo "  $dest/train_images/video_0/*.jpg"
    echo "  $dest/train_images/video_1/*.jpg"
    echo "  $dest/train_images/video_2/*.jpg"
    echo ""
    echo "  3 videos, ~23,501 frames total, 1-class detection (COTS)"
}

# =============================================================================
# Moorea Labeled Corals
# =============================================================================
download_moorea() {
    echo "=========================================="
    echo "Moorea Labeled Corals dataset"
    echo "=========================================="
    local dest="$OUTPUT_DIR/Moorea_Labeled_Corals"
    if [ -d "$dest/knb-lter-mcr.5006.3" ]; then
        echo "Moorea Labeled Corals already present at $dest"
        echo "Verifying..."
        local n_images
        n_images=$(find "$dest" -name "*.jpg" | wc -l)
        local n_anns
        n_anns=$(find "$dest" -name "*.jpg.txt" | wc -l)
        echo "  Images found: $n_images (expected: ~2,050)"
        echo "  Annotation files: $n_anns (expected: ~2,050)"
        return
    fi

    echo "Download Moorea Labeled Corals from EDI repository:"
    echo "  https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-mcr&identifier=5006"
    echo ""
    echo "Download package ID: knb-lter-mcr.5006.3"
    echo ""
    echo "Expected structure:"
    echo "  $dest/knb-lter-mcr.5006.3/MCR_LTER_*_2008_*/2008/*.jpg"
    echo "  $dest/knb-lter-mcr.5006.3/MCR_LTER_*_2008_*/2008/*.jpg.txt"
    echo "  (3 year directories: 2008, 2009, 2010)"
}

# =============================================================================
# Coralscapes
# =============================================================================
download_coralscapes() {
    echo "=========================================="
    echo "Coralscapes dataset"
    echo "=========================================="
    local dest="$OUTPUT_DIR/Coralscapes_dataset"
    if [ -d "$dest/train/images" ]; then
        echo "Coralscapes already present at $dest"
        echo "Verifying..."
        local n_train n_val n_test
        n_train=$(find "$dest/train/images" -name "*.jpg" | wc -l)
        n_val=$(find "$dest/validation/images" -name "*.jpg" | wc -l)
        n_test=$(find "$dest/test/images" -name "*.jpg" | wc -l)
        echo "  Train: $n_train (expected: 1,517)"
        echo "  Validation: $n_val (expected: 166)"
        echo "  Test: $n_test (expected: 392)"
        return
    fi

    echo "Download Coralscapes from HuggingFace:"
    echo "  pip install datasets"
    echo "  python -c \""
    echo "    from datasets import load_dataset"
    echo "    import os"
    echo "    ds = load_dataset('EPFL-ECEO/coralscapes')"
    echo "    base_dir = '$dest'"
    echo "    for split in ds.keys():"
    echo "        img_dir = os.path.join(base_dir, split, 'images')"
    echo "        lbl_dir = os.path.join(base_dir, split, 'labels')"
    echo "        os.makedirs(img_dir, exist_ok=True)"
    echo "        os.makedirs(lbl_dir, exist_ok=True)"
    echo "        for i, sample in enumerate(ds[split]):"
    echo "            name = f'{i:06d}'"
    echo "            sample['image'].save(os.path.join(img_dir, f'{name}.jpg'))"
    echo "            sample['label'].save(os.path.join(lbl_dir, f'{name}.png'))"
    echo "  \""
    echo ""
    echo "Expected structure:"
    echo "  $dest/{train,validation,test}/images/*.jpg"
    echo "  $dest/{train,validation,test}/labels/*.png"
}

# =============================================================================
# Brackish
# =============================================================================
download_brackish() {
    echo "=========================================="
    echo "Brackish Underwater dataset"
    echo "=========================================="
    local dest="$OUTPUT_DIR/Brackish_dataset"
    if [ -d "$dest/annotations" ]; then
        echo "Brackish dataset already present at $dest"
        echo "Verifying..."
        local n_videos
        n_videos=$(find "$dest/dataset/videos" -name "*.avi" 2>/dev/null | wc -l)
        echo "  Videos: $n_videos"
        echo "  COCO annotations: $(ls "$dest/annotations/annotations_COCO/" 2>/dev/null | wc -l) files"
        echo "  Train frames listed: $(wc -l < "$dest/train.txt" 2>/dev/null || echo 0)"
        echo ""
        echo "  NOTE: Frames must be extracted from videos before use."
        echo "  Run: python data/extract_brackish_frames.py"
        return
    fi

    echo "Download Brackish from:"
    echo "  https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset"
    echo "  or: https://github.com/AlexanderMelde/Brackish-Underwater-Dataset"
    echo ""
    echo "Expected structure:"
    echo "  $dest/dataset/videos/{category}/*.avi"
    echo "  $dest/annotations/annotations_COCO/{split}_groundtruth.json"
    echo "  $dest/train.txt, valid.txt, test.txt"
    echo ""
    echo "After download, extract video frames:"
    echo "  python data/extract_brackish_frames.py --root $dest"
}

# =============================================================================
# Verify all datasets
# =============================================================================
verify_all() {
    echo "=========================================="
    echo "Verifying all datasets at $OUTPUT_DIR"
    echo "=========================================="
    download_deepfish
    echo ""
    download_aqua20
    echo ""
    download_cots_gbr
    echo ""
    download_moorea
    echo ""
    download_coralscapes
    echo ""
    download_brackish
}

# =============================================================================
# Main
# =============================================================================
case "$DATASET" in
    deepfish)     download_deepfish ;;
    aqua20)       download_aqua20 ;;
    cots|cots_gbr) download_cots_gbr ;;
    moorea)       download_moorea ;;
    coralscapes)  download_coralscapes ;;
    brackish)     download_brackish ;;
    verify)       verify_all ;;
    all)
        download_deepfish
        echo ""
        download_aqua20
        echo ""
        download_cots_gbr
        echo ""
        download_moorea
        echo ""
        download_coralscapes
        echo ""
        download_brackish
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Usage: bash data/download_datasets.sh [deepfish|aqua20|cots_gbr|moorea|coralscapes|brackish|all|verify] [output_dir]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done. Use 'verify' to check all datasets."
echo "=========================================="
