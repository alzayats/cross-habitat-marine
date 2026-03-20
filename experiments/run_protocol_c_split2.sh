#!/usr/bin/env bash
set -euo pipefail
SEED=42; CONFIG="configs/base_config.yaml"; OUT="./outputs/fewshot_curve"

# Split 2: all moorea runs
for model in dinov2_base clip_base; do
  for adapt in linear_probe lora_r4; do
    echo "[SPLIT2] $model + $adapt: deepfish -> moorea"
    python -m experiments.run_fewshot_curve \
      --model "$model" --adaptation "$adapt" \
      --source deepfish --target moorea \
      --strategy combined --seed $SEED \
      --config "$CONFIG" --output-dir "$OUT"
  done
done
echo "[SPLIT2] resnet50_imagenet + linear_probe: deepfish -> moorea"
python -m experiments.run_fewshot_curve \
  --model resnet50_imagenet --adaptation linear_probe \
  --source deepfish --target moorea \
  --strategy combined --seed $SEED \
  --config "$CONFIG" --output-dir "$OUT"
echo "=== Split 2 done ==="
