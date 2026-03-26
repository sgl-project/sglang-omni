#!/bin/bash
# Run 5 rounds of first-50-sample WER evaluation with different seeds.
# Restarts the server between each round to avoid deterministic output.
set -e

PORT=50247
SERVER_GPU=2
ASR_GPU=7
PYTHON=/data/chenyang/.python/omni/bin/python
SEEDS=(42 123 456 789 1024)

# Use explicit python path to avoid zshrc GPU auto-selection interference
export PATH="/data/chenyang/.python/omni/bin:$PATH"

start_server() {
    echo "Starting server on GPU $SERVER_GPU, port $PORT..."
    CUDA_VISIBLE_DEVICES=$SERVER_GPU $PYTHON -m sglang_omni.cli.cli serve \
        --model-path fishaudio/s2-pro \
        --config examples/configs/s2pro_tts.yaml \
        --port $PORT > /tmp/s2pro_server.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    for i in $(seq 1 300); do
        resp=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null || echo "000")
        if [ "$resp" = "200" ]; then
            echo "Server healthy after $((i*2))s"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not become healthy"
    kill $SERVER_PID 2>/dev/null
    exit 1
}

stop_server() {
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    sleep 5
    kill -9 $SERVER_PID 2>/dev/null || true
    # Also kill any children
    pkill -9 -f "sglang_omni.*--port $PORT" 2>/dev/null || true
    sleep 3
    echo "Server stopped."
}

for idx in 0 1 2 3 4; do
    round=$((idx + 1))
    seed=${SEEDS[$idx]}
    echo ""
    echo "=========================================="
    echo "  Round $round / 5  (seed=$seed)"
    echo "=========================================="

    start_server

    CUDA_VISIBLE_DEVICES=$ASR_GPU $PYTHON -m benchmarks.performance.tts.benchmark_tts_wer \
        --meta /data/chenyang/seedtts_testset/en/meta.lst \
        --model fishaudio/s2-pro \
        --port $PORT \
        --output-dir results/s2pro_en_wer_round${round} \
        --lang en \
        --device cuda:0 \
        --max-samples 50 \
        --seed $seed

    echo "Round $round done."
    stop_server
done

echo ""
echo "All 5 rounds complete."
