#!/usr/bin/env bash
set -euo pipefail
SEED=42; CONFIG="configs/base_config.yaml"; OUT="./outputs/fewshot_curve"

# Split 3: all brackish runs
for model in dinov2_base clip_base; do
  for adapt in linear_probe lora_r4; do
    echo "[SPLIT3] $model + $adapt: deepfish -> brackish"
    python -m experiments.run_fewshot_curve \
      --model "$model" --adaptation "$adapt" \
      --source deepfish --target brackish \
      --strategy combined --seed $SEED \
      --config "$CONFIG" --output-dir "$OUT"
  done
done
echo "[SPLIT3] resnet50_imagenet + linear_probe: deepfish -> brackish"
python -m experiments.run_fewshot_curve \
  --model resnet50_imagenet --adaptation linear_probe \
  --source deepfish --target brackish \
  --strategy combined --seed $SEED \
  --config "$CONFIG" --output-dir "$OUT"
echo "=== Split 3 done ==="
