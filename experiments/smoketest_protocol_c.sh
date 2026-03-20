#!/usr/bin/env bash
# Smoke test: runs Protocol C pipeline with minimal compute to catch crashes.
# - 2 epochs max (vs 30-50)
# - 1 trial (vs 5)
# - k = [0, 1, 5] only (vs [0,1,2,5,10,20,50,100])
# - 1 seed (vs 3)
# Output goes to a separate dir so it doesn't pollute real results.
set -euo pipefail

SEED=42
CONFIG="configs/base_config_smoketest.yaml"
OUTPUT_C="./outputs/fewshot_curve_smoketest"

echo "=== SMOKE TEST: Protocol C ==="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_C"
echo ""

FEWSHOT_PAIRS=("deepfish:aqua20" "deepfish:moorea" "deepfish:brackish")

for pair in "${FEWSHOT_PAIRS[@]}"; do
    IFS=':' read -r src tgt <<< "$pair"

    for model in dinov2_base clip_base; do
        for adapt in linear_probe lora_r4; do
            echo "[SMOKETEST] $model + $adapt: $src -> $tgt"
            python -m experiments.run_fewshot_curve \
                --model "$model" --adaptation "$adapt" \
                --source "$src" --target "$tgt" \
                --strategy combined --seed $SEED \
                --config "$CONFIG" --output-dir "$OUTPUT_C"
        done
    done

    echo "[SMOKETEST] resnet50_imagenet + linear_probe: $src -> $tgt"
    python -m experiments.run_fewshot_curve \
        --model resnet50_imagenet --adaptation linear_probe \
        --source "$src" --target "$tgt" \
        --strategy combined --seed $SEED \
        --config "$CONFIG" --output-dir "$OUTPUT_C"
done

echo ""
echo "=== SMOKE TEST PASSED ==="
echo "Safe to run: bash experiments/run_protocol_c.sh"
