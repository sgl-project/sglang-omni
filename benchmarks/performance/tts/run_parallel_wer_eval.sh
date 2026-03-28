#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Parallel WER evaluation for Qwen3-Omni via sglang-omni server.
#
# Splits the seed-tts-eval EN meta.lst into N shards, launches N servers
# on separate GPU pairs, runs N benchmark clients in parallel, then
# merges results into a single JSON + summary.
#
# Usage:
#   bash benchmarks/performance/tts/run_parallel_wer_eval.sh
#
# Prerequisites:
#   - seed-tts-eval dataset at $DATASET_DIR
#   - 6 free GPUs for 3 servers (2 GPUs each) + 1 GPU for ASR
#   - sglang-omni installed with Qwen3 Omni speech pipeline fix

set -euo pipefail

# --- Configuration ---
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
DATASET_DIR="${DATASET_DIR:-/data/chenyang/seedtts_testset}"
META_FILE="${DATASET_DIR}/en/meta.lst"
OUTPUT_BASE="${OUTPUT_BASE:-results/qwen3_omni_server_en_full}"
NUM_SHARDS=3
BASE_PORT=8000
ASR_GPU=7  # Shared GPU for Whisper ASR (only ~3GB)

# GPU pairs for each server
SERVER_GPUS=("0,1" "2,3" "4,5")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "============================================================"
echo "  Parallel Qwen3-Omni WER Evaluation (${NUM_SHARDS} shards)"
echo "============================================================"
echo "  Model:       $MODEL_PATH"
echo "  Meta file:   $META_FILE"
echo "  Output base: $OUTPUT_BASE"
echo "  Server GPUs: ${SERVER_GPUS[*]}"
echo "  ASR GPU:     $ASR_GPU"
echo "============================================================"

# --- Step 1: Split meta.lst into shards ---
TOTAL_LINES=$(wc -l < "$META_FILE")
SHARD_SIZE=$(( (TOTAL_LINES + NUM_SHARDS - 1) / NUM_SHARDS ))

echo "[1/5] Splitting $META_FILE ($TOTAL_LINES samples) into $NUM_SHARDS shards of ~$SHARD_SIZE each..."

SHARD_DIR="$OUTPUT_BASE/shards"
mkdir -p "$SHARD_DIR"

split -l "$SHARD_SIZE" -d -a 1 "$META_FILE" "$SHARD_DIR/shard_"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_file="$SHARD_DIR/shard_$i"
    if [ -f "$shard_file" ]; then
        lines=$(wc -l < "$shard_file")
        echo "  Shard $i: $lines samples"
    fi
done

# --- Step 2: Start N servers ---
echo "[2/5] Starting $NUM_SHARDS servers..."

SERVER_PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    port=$((BASE_PORT + i))
    gpus="${SERVER_GPUS[$i]}"
    log_file="$OUTPUT_BASE/server_${i}.log"
    mkdir -p "$OUTPUT_BASE"

    echo "  Server $i: GPUs=$gpus, port=$port, log=$log_file"
    CUDA_VISIBLE_DEVICES="$gpus" python "$REPO_ROOT/examples/run_qwen3_omni_speech_server.py" \
        --model-path "$MODEL_PATH" \
        --gpu-thinker 0 --gpu-talker 1 \
        --gpu-code-predictor 1 --gpu-code2wav 0 \
        --port "$port" \
        > "$log_file" 2>&1 &
    SERVER_PIDS+=($!)
done

# --- Step 3: Wait for all servers to be ready ---
echo "[3/5] Waiting for all servers to be ready (timeout: 600s)..."

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    port=$((BASE_PORT + i))
    url="http://localhost:$port"
    echo -n "  Waiting for server $i at $url..."
    start_time=$(date +%s)
    while true; do
        if curl -sf "$url/health" > /dev/null 2>&1; then
            echo " ready!"
            break
        fi
        elapsed=$(( $(date +%s) - start_time ))
        if [ "$elapsed" -gt 600 ]; then
            echo " TIMEOUT!"
            echo "ERROR: Server $i did not start within 600s. Check $OUTPUT_BASE/server_${i}.log"
            # Kill all servers
            for pid in "${SERVER_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
            exit 1
        fi
        sleep 2
    done
done

echo "  All $NUM_SHARDS servers ready."

# --- Step 4: Run benchmark clients in parallel ---
echo "[4/5] Running $NUM_SHARDS benchmark clients in parallel..."

CLIENT_PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    port=$((BASE_PORT + i))
    shard_file="$SHARD_DIR/shard_$i"
    output_dir="$OUTPUT_BASE/part_$i"
    log_file="$OUTPUT_BASE/client_${i}.log"

    echo "  Client $i: port=$port, shard=$shard_file, output=$output_dir"
    CUDA_VISIBLE_DEVICES="$ASR_GPU" python -m benchmarks.performance.tts.benchmark_tts_wer_qwen3_omni_server \
        --meta "$shard_file" \
        --output-dir "$output_dir" \
        --lang en \
        --asr-device cuda:0 \
        --port "$port" \
        > "$log_file" 2>&1 &
    CLIENT_PIDS+=($!)
done

echo "  Waiting for all clients to finish..."
FAILED=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    pid="${CLIENT_PIDS[$i]}"
    if wait "$pid"; then
        echo "  Client $i finished successfully."
    else
        echo "  Client $i FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
done

# --- Step 5: Merge results ---
echo "[5/5] Merging results..."

python "$SCRIPT_DIR/merge_wer_results.py" \
    --parts "$OUTPUT_BASE/part_0/wer_results.json" \
             "$OUTPUT_BASE/part_1/wer_results.json" \
             "$OUTPUT_BASE/part_2/wer_results.json" \
    --output "$OUTPUT_BASE/wer_results_merged.json"

# --- Cleanup: stop servers ---
echo "Stopping servers..."
for pid in "${SERVER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
done
sleep 3
# Kill any orphaned stage processes
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    gpus="${SERVER_GPUS[$i]}"
    for gpu_id in ${gpus//,/ }; do
        nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
done

echo ""
echo "============================================================"
echo "  Done! Results saved to: $OUTPUT_BASE/wer_results_merged.json"
if [ "$FAILED" -gt 0 ]; then
    echo "  WARNING: $FAILED client(s) failed. Check logs."
fi
echo "============================================================"
