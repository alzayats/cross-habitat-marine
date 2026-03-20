#!/usr/bin/env bash
# =============================================================================
# Master script: Run all cross-habitat marine species recognition experiments
# Usage: bash experiments/run_all.sh [--protocol A|B|C|detection|reduced|all] [--dry-run]
# =============================================================================

set -euo pipefail

PROTOCOL="${1:-all}"
DRY_RUN="${2:-}"
CONFIG="configs/base_config.yaml"

run_cmd() {
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "[DRY RUN] $*"
    else
        echo "[RUNNING] $*"
        eval "$@"
    fi
}

echo "=============================================="
echo "Cross-Habitat Marine Species Recognition"
echo "Experiment Suite (6 datasets, 20 habitats)"
echo "=============================================="
echo "Protocol: $PROTOCOL"
echo "Config:   $CONFIG"
echo ""

# --- Protocol A: Within-habitat transfer (20 DeepFish habitats) ---
run_protocol_a() {
    echo "=== Protocol A: Within-Dataset Cross-Habitat Transfer (20 habitats) ==="
    echo ""

    MODELS=(dinov2_base dinov2_large dinov3_large clip_base mae_base resnet50_imagenet resnet50_random efficientnet_b4)
    ADAPTATIONS=(linear_probe lora_r4 lora_r16 vpt_deep full_finetune)

    for model in "${MODELS[@]}"; do
        for adapt in "${ADAPTATIONS[@]}"; do
            run_cmd python -m experiments.run_within_habitat \
                --model "$model" --adaptation "$adapt" \
                --all --config "$CONFIG" \
                --output-dir ./outputs/within_habitat
        done
    done
}

# --- Protocol B: Cross-dataset transfer ---
run_protocol_b() {
    echo "=== Protocol B: Cross-Dataset Geographic Transfer ==="
    echo "Scenarios: DeepFish→AQUA20, DeepFish→COTS_GBR, DeepFish→Moorea,"
    echo "           AQUA20→Coralscapes, DeepFish→Brackish"
    echo ""

    run_cmd python -m experiments.run_cross_dataset \
        --all --config "$CONFIG" \
        --output-dir ./outputs/cross_dataset
}

# --- Protocol C: Few-shot adaptation curves ---
run_protocol_c() {
    echo "=== Protocol C: Few-Shot Adaptation Curves ==="
    echo ""

    run_cmd python -m experiments.run_fewshot_curve \
        --all --config "$CONFIG" \
        --output-dir ./outputs/fewshot_curve
}

# --- Detection experiments (COTS GBR + Brackish) ---
run_detection() {
    echo "=== Detection Experiments (COTS GBR + Brackish) ==="
    echo ""

    MODELS=(dinov2_base dinov2_large clip_base resnet50_imagenet)
    ADAPTATIONS=(linear_probe lora_r4 full_finetune)

    for model in "${MODELS[@]}"; do
        for adapt in "${ADAPTATIONS[@]}"; do
            run_cmd python -m experiments.run_detection \
                --model "$model" --adaptation "$adapt" \
                --config "$CONFIG" \
                --output-dir ./outputs/detection
        done
    done
}

# --- Reduced experiment suite (~595 runs, ~10 days on single 4090) ---
run_reduced() {
    echo "=== Reduced Experiment Suite (4 models, ~595 runs) ==="
    echo "DINOv2-B LP already done (3 seeds). All new runs: seed=42 only."
    echo ""

    SEED=42
    OUTPUT_A="./outputs/within_habitat"
    OUTPUT_B="./outputs/cross_dataset"
    OUTPUT_C="./outputs/fewshot_curve"

    # =====================================================================
    # Protocol A: Within-Habitat Transfer (20 habitats)
    # =====================================================================
    echo "--- Protocol A (Reduced): Within-Habitat Transfer ---"
    echo "Skipping: DINOv2-B + LP (already done with 3 seeds)"
    echo ""

    # ---- Group 1: Foundation LoRA + VPT (4 jobs, ~16 GB) ----
    echo "[PARALLEL GROUP 1] Foundation LoRA + VPT (4 jobs: DINOv2-B + CLIP-B x LoRA/VPT)"
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "[DRY RUN] python -m experiments.run_within_habitat --model dinov2_base --adaptation lora_r4 --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] python -m experiments.run_within_habitat --model dinov2_base --adaptation vpt_deep --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] python -m experiments.run_within_habitat --model clip_base --adaptation lora_r4 --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] python -m experiments.run_within_habitat --model clip_base --adaptation vpt_deep --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] wait"
    else
        python -m experiments.run_within_habitat \
            --model dinov2_base --adaptation lora_r4 \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID1=$!
        python -m experiments.run_within_habitat \
            --model dinov2_base --adaptation vpt_deep \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID2=$!
        python -m experiments.run_within_habitat \
            --model clip_base --adaptation lora_r4 \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID3=$!
        python -m experiments.run_within_habitat \
            --model clip_base --adaptation vpt_deep \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID4=$!
        echo "[WAITING] Group 1 PIDs: $PID1 $PID2 $PID3 $PID4"
        wait $PID1 $PID2 $PID3 $PID4
        echo "[DONE] Parallel group 1 complete"
    fi

    # ---- Group 2: All linear probes (3 jobs, ~10 GB) ----
    echo "[PARALLEL GROUP 2] All linear probes (CLIP-B + ResNet50-IN + EfficientNet-B4)"
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "[DRY RUN] python -m experiments.run_within_habitat --model clip_base --adaptation linear_probe --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] python -m experiments.run_within_habitat --model resnet50_imagenet --adaptation linear_probe --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] python -m experiments.run_within_habitat --model efficientnet_b4 --adaptation linear_probe --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] wait"
    else
        python -m experiments.run_within_habitat \
            --model clip_base --adaptation linear_probe \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID1=$!
        python -m experiments.run_within_habitat \
            --model resnet50_imagenet --adaptation linear_probe \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID2=$!
        python -m experiments.run_within_habitat \
            --model efficientnet_b4 --adaptation linear_probe \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID3=$!
        echo "[WAITING] Group 2 PIDs: $PID1 $PID2 $PID3"
        wait $PID1 $PID2 $PID3
        echo "[DONE] Parallel group 2 complete"
    fi

    # ---- Group 3: CNN full finetune (2 jobs, ~10 GB) ----
    echo "[PARALLEL GROUP 3] CNN full finetune (ResNet50-IN + EfficientNet-B4)"
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "[DRY RUN] python -m experiments.run_within_habitat --model resnet50_imagenet --adaptation full_finetune --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] python -m experiments.run_within_habitat --model efficientnet_b4 --adaptation full_finetune --seed $SEED --all --leave-one-out-only --config $CONFIG --output-dir $OUTPUT_A &"
        echo "[DRY RUN] wait"
    else
        python -m experiments.run_within_habitat \
            --model resnet50_imagenet --adaptation full_finetune \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID1=$!
        python -m experiments.run_within_habitat \
            --model efficientnet_b4 --adaptation full_finetune \
            --seed $SEED --all --leave-one-out-only --config "$CONFIG" \
            --output-dir "$OUTPUT_A" &
        PID2=$!
        echo "[WAITING] Group 3 PIDs: $PID1 $PID2"
        wait $PID1 $PID2
        echo "[DONE] Parallel group 3 complete"
    fi

    echo ""
    echo "--- Protocol A (Reduced) complete ---"
    echo ""

    # =====================================================================
    # Protocol B: Cross-Dataset Transfer (4 pairs, open_set only)
    # =====================================================================
    echo "--- Protocol B (Reduced): Cross-Dataset Transfer ---"
    echo "4 pairs: DeepFish -> {AQUA20, Coralscapes, Moorea, Brackish}"
    echo "Strategy: open_set only"
    echo ""

    PAIRS=("deepfish:aqua20" "deepfish:coralscapes" "deepfish:moorea" "deepfish:brackish")

    # Foundation models: LP, LoRA r4, VPT-Deep
    for pair in "${PAIRS[@]}"; do
        IFS=':' read -r src tgt <<< "$pair"
        for model in dinov2_base clip_base; do
            for adapt in linear_probe lora_r4 vpt_deep; do
                run_cmd python -m experiments.run_cross_dataset \
                    --model "$model" --adaptation "$adapt" \
                    --source "$src" --target "$tgt" \
                    --strategy open_set --seed $SEED \
                    --config "$CONFIG" --output-dir "$OUTPUT_B"
            done
        done
    done

    # CNN models: LP, Full Fine-tune
    for pair in "${PAIRS[@]}"; do
        IFS=':' read -r src tgt <<< "$pair"
        for model in resnet50_imagenet efficientnet_b4; do
            for adapt in linear_probe full_finetune; do
                run_cmd python -m experiments.run_cross_dataset \
                    --model "$model" --adaptation "$adapt" \
                    --source "$src" --target "$tgt" \
                    --strategy open_set --seed $SEED \
                    --config "$CONFIG" --output-dir "$OUTPUT_B"
            done
        done
    done

    echo ""
    echo "--- Protocol B (Reduced) complete ---"
    echo ""

    # =====================================================================
    # Protocol C: Few-Shot Curves (3 pairs, 7 k-values, 3 trials)
    # =====================================================================
    echo "--- Protocol C (Reduced): Few-Shot Adaptation Curves ---"
    echo "3 pairs: DeepFish -> {AQUA20, Moorea, Brackish}"
    echo "k = [0, 1, 2, 5, 10, 20, 50], 3 trials, combined only"
    echo ""

    FEWSHOT_PAIRS=("deepfish:aqua20" "deepfish:moorea" "deepfish:brackish")

    # Foundation models: LP, LoRA r4
    for pair in "${FEWSHOT_PAIRS[@]}"; do
        IFS=':' read -r src tgt <<< "$pair"
        for model in dinov2_base clip_base; do
            for adapt in linear_probe lora_r4; do
                run_cmd python -m experiments.run_fewshot_curve \
                    --model "$model" --adaptation "$adapt" \
                    --source "$src" --target "$tgt" \
                    --strategy combined --seed $SEED \
                    --config "$CONFIG" --output-dir "$OUTPUT_C"
            done
        done
    done

    # CNN models: LP only
    for pair in "${FEWSHOT_PAIRS[@]}"; do
        IFS=':' read -r src tgt <<< "$pair"
        run_cmd python -m experiments.run_fewshot_curve \
            --model resnet50_imagenet --adaptation linear_probe \
            --source "$src" --target "$tgt" \
            --strategy combined --seed $SEED \
            --config "$CONFIG" --output-dir "$OUTPUT_C"
    done

    echo ""
    echo "--- Protocol C (Reduced) complete ---"
    echo ""

    # =====================================================================
    # Post-experiment analysis and figure generation
    # =====================================================================
    echo "--- Post-Experiment: Analysis & Figure Generation ---"
    echo ""

    # Feature extraction for t-SNE/CKA analysis
    # (extract from best checkpoints of key model-adaptation combos on DeepFish)
    for model in dinov2_base clip_base resnet50_imagenet efficientnet_b4; do
        for adapt in linear_probe lora_r4 vpt_deep full_finetune; do
            CKPT="./outputs/within_habitat/${model}_${adapt}_*/checkpoints/best_model.pt"
            CKPT_FILE=$(ls $CKPT 2>/dev/null | head -1 || true)
            if [ -n "$CKPT_FILE" ]; then
                run_cmd python -m analysis.extract_features \
                    --model "$model" --adaptation "$adapt" \
                    --dataset deepfish --checkpoint "$CKPT_FILE"
            fi
        done
    done

    # Compute efficiency metrics
    run_cmd python -m analysis.compute_efficiency --all --output ./outputs/efficiency.json

    # Generate all figures
    run_cmd python -m figures.generate_all_figures

    echo ""
    echo "--- Analysis & Figure Generation complete ---"
    echo ""
}

# --- Run selected protocols ---
case "$PROTOCOL" in
    A|a)
        run_protocol_a
        ;;
    B|b)
        run_protocol_b
        ;;
    C|c)
        run_protocol_c
        ;;
    detection)
        run_detection
        ;;
    reduced)
        run_reduced
        ;;
    all)
        run_protocol_a
        run_protocol_b
        run_protocol_c
        run_detection
        ;;
    *)
        echo "Unknown protocol: $PROTOCOL"
        echo "Usage: bash experiments/run_all.sh [A|B|C|detection|reduced|all] [--dry-run]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results saved to ./outputs/"
echo ""
echo "Next steps:"
echo "  1. Generate figures: python -m figures.generate_all_figures"
echo "  2. Extract features: python -m analysis.extract_features"
echo "=============================================="
