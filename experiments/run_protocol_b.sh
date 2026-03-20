#!/usr/bin/env bash
set -euo pipefail

SEED=42
CONFIG="configs/base_config.yaml"
OUTPUT_B="./outputs/cross_dataset"

echo "=== Protocol B: Cross-Dataset Transfer ==="
echo "4 pairs × (2 foundation × 3 adapt + 2 CNN × 2 adapt) = 40 runs"
echo ""

PAIRS=("deepfish:aqua20" "deepfish:coralscapes" "deepfish:moorea" "deepfish:brackish")

for pair in "${PAIRS[@]}"; do
    IFS=':' read -r src tgt <<< "$pair"

    # Foundation models: LP, LoRA r4, VPT-Deep
    for model in dinov2_base clip_base; do
        for adapt in linear_probe lora_r4 vpt_deep; do
            echo "[RUNNING] $model + $adapt: $src -> $tgt"
            python -m experiments.run_cross_dataset \
                --model "$model" --adaptation "$adapt" \
                --source "$src" --target "$tgt" \
                --strategy open_set --seed $SEED \
                --config "$CONFIG" --output-dir "$OUTPUT_B"
        done
    done

    # CNN models: LP, Full Fine-tune
    for model in resnet50_imagenet efficientnet_b4; do
        for adapt in linear_probe full_finetune; do
            echo "[RUNNING] $model + $adapt: $src -> $tgt"
            python -m experiments.run_cross_dataset \
                --model "$model" --adaptation "$adapt" \
                --source "$src" --target "$tgt" \
                --strategy open_set --seed $SEED \
                --config "$CONFIG" --output-dir "$OUTPUT_B"
        done
    done
done

echo ""
echo "=== Protocol B complete ==="
