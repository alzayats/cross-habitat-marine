#!/usr/bin/env bash
set -euo pipefail
SEED=42; CONFIG="configs/base_config.yaml"; OUT="./outputs/fewshot_curve"

# Split 1: aqua20 LoRA + all aqua20 LP
for model in dinov2_base clip_base; do
  for adapt in linear_probe lora_r4; do
    echo "[SPLIT1] $model + $adapt: deepfish -> aqua20"
    python -m experiments.run_fewshot_curve \
      --model "$model" --adaptation "$adapt" \
      --source deepfish --target aqua20 \
      --strategy combined --seed $SEED \
      --config "$CONFIG" --output-dir "$OUT"
  done
done
echo "[SPLIT1] resnet50_imagenet + linear_probe: deepfish -> aqua20"
python -m experiments.run_fewshot_curve \
  --model resnet50_imagenet --adaptation linear_probe \
  --source deepfish --target aqua20 \
  --strategy combined --seed $SEED \
  --config "$CONFIG" --output-dir "$OUT"
echo "=== Split 1 done ==="
