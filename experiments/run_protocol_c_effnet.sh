#!/usr/bin/env bash
set -euo pipefail
SEED=42; CONFIG="configs/base_config.yaml"; OUT="./outputs/fewshot_curve"

# EfficientNet-B4 LP for all 3 targets (~5min each)
for target in aqua20 moorea brackish; do
  echo "[EFFNET] efficientnet_b4 + linear_probe: deepfish -> $target"
  python -m experiments.run_fewshot_curve \
    --model efficientnet_b4 --adaptation linear_probe \
    --source deepfish --target "$target" \
    --strategy combined --seed $SEED \
    --config "$CONFIG" --output-dir "$OUT"
done
echo "=== EfficientNet LP done ==="
