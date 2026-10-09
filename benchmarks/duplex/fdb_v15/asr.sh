#!/usr/bin/env bash
# Step 2: Parakeet ASR of all four roles, then official VAD timing intervals.
# Usage: asr.sh REPEAT [--retry-failed]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

REPEAT="${1:?usage: asr.sh REPEAT [--retry-failed]}"
shift
REPEAT_DIR="$RUN_ROOT/repeat-$REPEAT"
SCORES="$REPEAT_DIR/scores"
COMMON_ARGS=(
    --reference-source "$FDB_SOURCE"
    --tree "$ENGINE_LABEL=$REPEAT_DIR/reference-audio"
    --out "$SCORES"
)

if [ ! -f "$REPEAT_DIR/reference-audio/reference-manifest.json" ]; then
    echo "ERROR: run generate.sh $REPEAT first." >&2
    exit 1
fi
cd "$SGLANG_OMNI_ROOT"
failed_phases=0

echo "== ASR (Parakeet on GPU $SCORING_GPU)"
CUDA_VISIBLE_DEVICES="$SCORING_GPU" "$SCORING_VENV/bin/python" \
    -m benchmarks.eval.benchmark_duplex_reference asr "${COMMON_ARGS[@]}" \
    --nemo "$PARAKEET_NEMO" --nemo-sha256 "$PARAKEET_SHA256" --device cuda "$@" \
    || failed_phases=$((failed_phases + 1))

echo "== Timing (official VAD intervals)"
CUDA_VISIBLE_DEVICES="" "$SCORING_VENV/bin/python" \
    -m benchmarks.eval.benchmark_duplex_reference timing "${COMMON_ARGS[@]}" \
    --audio-loader soundfile "$@" \
    || failed_phases=$((failed_phases + 1))

if [ "$failed_phases" -gt 0 ]; then
    echo "WARNING: $failed_phases phase(s) reported failures; logs are in $SCORES/logs." \
        "Rerun with: bash $(dirname "${BASH_SOURCE[0]}")/asr.sh $REPEAT --retry-failed"
    exit 1
fi
echo "Step 2 done: $SCORES"
