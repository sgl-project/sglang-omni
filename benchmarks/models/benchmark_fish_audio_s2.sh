#!/usr/bin/env bash
# set -euo pipefail

# ---------------------------------------------------------------------------
# Benchmark Fish Audio S2 via the TTS benchmark.
#
# Usage:
#   # Against an already-running server at localhost:8000
#   ./benchmarks/models/benchmark_fish_audio_s2.sh
#
#   # With custom options
#   ./benchmarks/models/benchmark_fish_audio_s2.sh \
#       --base-url http://localhost:8000 \
#       --max-concurrency 4 --stream
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR/.."

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"

python "$BENCHMARK_DIR/benchmark_tts.py" \
  --launch-server \
  --model-path "fishaudio/s2-pro" \
  --config "$BENCHMARK_DIR/../examples/configs/s2pro_tts.yaml" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dataset "$BENCHMARK_DIR/cache/seedtts_tts_5_samples/data.jsonl" \
  --output-dir "$BENCHMARK_DIR/results/fish_audio_s2" \
  --save-audio \
  "$@"
