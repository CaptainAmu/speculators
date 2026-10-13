#!/bin/bash
# Offline FlashMap Training Script
#
# Runs the full offline FlashMap pipeline: response regeneration, data preparation,
# hidden-state generation, then the two training stages from the cached hidden
# states -- the flow (stage 1) and the flow map distilled from it (stage 2). The
# hidden states are generated once and read by both stages.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/flashmap_qwen3_8b_two_stage_offline.sh
#
# For a detailed walkthrough, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/

### Example E2E run for FlashMap Qwen3-8B on 5k regenerated UltraChat samples ###

# Note: With just 5k samples, the model performance will not be very good, however there
# are enough samples to verify that the pipeline is working correctly and that the model
# is learning something. The released FlashMap drafters were trained on 500k target-
# regenerated samples: the flow for 20 epochs, the map for 12.

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
# Output JSONL from response regeneration (Step 0). FlashMap's flow models the
# data distribution, so the responses should come from the target model.
DATASET="./ultrachat_Qwen3-8B.jsonl"
OUTPUT_DIR="./output/flashmap_qwen3_8b"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
CONCURRENCY=32
TARGET_LAYER_IDS="2 10 18 26 34"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

# Stage 1: the flow (the released recipe: AdamW, weight decay 0.1, 512 anchors).
FLOW_EPOCHS=20
FLOW_LR=6e-4
FLOW_ANCHORS=512
# Stage 2: the flow map. It holds several vocabulary-sized states per slot during
# a step, so it trains on fewer anchors, at a lower learning rate.
MAP_EPOCHS=12
MAP_LR=5e-5
MAP_ANCHORS=128

# GPU assignments (offline reuses the same GPUs sequentially)
GPUS="0,1,2,3"
NUM_GPUS=4
# =======================================

# Step 0: Regenerate target responses; the helper starts and stops its server
bash scripts/response_regeneration/run_all.sh \
    --model "$MODEL" --dataset ultrachat \
    --gpus "$GPUS" --dp-size "$NUM_GPUS" --port "$VLLM_PORT" \
    --limit "$MAX_SAMPLES" --max-tokens "$SEQ_LENGTH" \
    --outfile "$DATASET" --resume

# Step 1: Launch vLLM server in the background
echo "=== Step 1: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$NUM_GPUS" --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 2: Prepare data
echo "=== Step 2: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --render-endpoint "http://localhost:${VLLM_PORT}" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH"

# Step 3: Generate hidden states
echo "=== Step 3: Generating hidden states ==="
speculators generate-offline-data \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY"

# Step 4: Stop vLLM server to free GPU memory for training
echo "=== Step 4: Stopping vLLM server ==="
cleanup

# Step 5: Train the flow
echo "=== Step 5: Training the flow (stage 1) ==="
CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_GPUS" \
    -m speculators.train \
    --speculator-type flashmap \
    --verifier-name-or-path "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$OUTPUT_DIR/flow" \
    --optimizer adamw \
    --lr "$FLOW_LR" \
    --weight-decay 0.1 \
    --scheduler-type cosine \
    --scheduler-warmup-ratio 0.04 \
    --epochs "$FLOW_EPOCHS" \
    --max-anchors "$FLOW_ANCHORS" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise

# Step 6: Distil the flow map from the last flow checkpoint
echo "=== Step 6: Training the flow map (stage 2) ==="
CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_GPUS" \
    -m speculators.train \
    --speculator-type flashmap \
    --flashmap-teacher "$OUTPUT_DIR/flow/$((FLOW_EPOCHS - 1))" \
    --verifier-name-or-path "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$OUTPUT_DIR/map" \
    --optimizer adamw \
    --lr "$MAP_LR" \
    --weight-decay 0 \
    --scheduler-warmup-ratio 0.04 \
    --epochs "$MAP_EPOCHS" \
    --max-anchors "$MAP_ANCHORS" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise

echo "Done. The flow map checkpoints in $OUTPUT_DIR/map/ are the drafter to serve."
