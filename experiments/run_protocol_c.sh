#!/usr/bin/env bash
set -euo pipefail

SEED=42
CONFIG="configs/base_config.yaml"
OUTPUT_C="./outputs/fewshot_curve"

echo "=== Protocol C: Few-Shot Adaptation Curves ==="
echo "3 pairs × (2 foundation × 2 adapt + 1 CNN × 1 adapt) = 15 runs"
echo ""

FEWSHOT_PAIRS=("deepfish:aqua20" "deepfish:moorea" "deepfish:brackish")

for pair in "${FEWSHOT_PAIRS[@]}"; do
    IFS=':' read -r src tgt <<< "$pair"

    # Foundation models: LP, LoRA r4
    for model in dinov2_base clip_base; do
        for adapt in linear_probe lora_r4; do
            echo "[RUNNING] $model + $adapt: $src -> $tgt"
            python -m experiments.run_fewshot_curve \
                --model "$model" --adaptation "$adapt" \
                --source "$src" --target "$tgt" \
                --strategy combined --seed $SEED \
                --config "$CONFIG" --output-dir "$OUTPUT_C"
        done
    done

    # CNN: ResNet50 LP only
    echo "[RUNNING] resnet50_imagenet + linear_probe: $src -> $tgt"
    python -m experiments.run_fewshot_curve \
        --model resnet50_imagenet --adaptation linear_probe \
        --source "$src" --target "$tgt" \
        --strategy combined --seed $SEED \
        --config "$CONFIG" --output-dir "$OUTPUT_C"
done

echo ""
echo "=== Protocol C complete ==="
