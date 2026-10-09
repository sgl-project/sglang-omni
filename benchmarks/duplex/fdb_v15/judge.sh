#!/usr/bin/env bash
# Step 3: LLM behavior judge (plus the semantic A/F/U judge for qwen), then the per-repeat report.
# Usage: judge.sh REPEAT [--retry-failed]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

REPEAT="${1:?usage: judge.sh REPEAT [--retry-failed]}"
shift
REPEAT_DIR="$RUN_ROOT/repeat-$REPEAT"
SCORES="$REPEAT_DIR/scores"
QWEN_SCORES="$REPEAT_DIR/judge-qwen"
SEMANTIC_SCORES="$REPEAT_DIR/semantic-qwen"
JUDGE_CONFIG="$FDB_WORK/judge/judge-config.json"
GPT_JUDGE_MODEL="gpt-4o-2024-08-06"
COMMON_ARGS=(
    --reference-source "$FDB_SOURCE"
    --tree "$ENGINE_LABEL=$REPEAT_DIR/reference-audio"
)
JUDGE_READY_TIMEOUT_S=1800

if [ ! -f "$SCORES/engines/$ENGINE_LABEL/manifest-receipt.json" ]; then
    echo "ERROR: run asr.sh $REPEAT first." >&2
    exit 1
fi
cd "$SGLANG_OMNI_ROOT"
reference() {
    "$SCORING_VENV/bin/python" -m benchmarks.eval.benchmark_duplex_reference "$@"
}
failed_phases=0
report_args=()

if [ "$JUDGE" = "qwen" ]; then
    echo "== Waiting for judge server at $JUDGE_URL"
    deadline=$((SECONDS + JUDGE_READY_TIMEOUT_S))
    until curl --silent --fail "$JUDGE_URL/models" | grep -q "$JUDGE_SERVED_MODEL"; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "ERROR: judge not ready; check the launch_judge.sh terminal." >&2
            exit 1
        fi
        sleep 10
    done
    echo "== Qwen judge -> $QWEN_SCORES"
    reference custom-judge "${COMMON_ARGS[@]}" --source-scores "$SCORES" --out "$QWEN_SCORES" \
        --judge-config "$JUDGE_CONFIG" --base-url "$JUDGE_URL" "$@" \
        || failed_phases=$((failed_phases + 1))
    reference custom-summarize "${COMMON_ARGS[@]}" --source-scores "$SCORES" --out "$QWEN_SCORES" \
        --judge-config "$JUDGE_CONFIG" \
        || failed_phases=$((failed_phases + 1))
    echo "== Semantic A/F/U judge -> $SEMANTIC_SCORES"
    "$SCORING_VENV/bin/python" -m benchmarks.duplex.semantic_judge \
        --reference-audio "$REPEAT_DIR/reference-audio" --scores "$SCORES" --engine "$ENGINE_LABEL" \
        --dataset-root "$FDB_DATASET" --out "$SEMANTIC_SCORES" --base-url "$JUDGE_URL" \
        --model-id "$JUDGE_MODEL_ID" --model-revision "$(cat "$JUDGE_MODEL_PATH.revision")" \
        --served-model "$JUDGE_SERVED_MODEL" \
        || failed_phases=$((failed_phases + 1))
    if [ -f "$SEMANTIC_SCORES/summary.json" ]; then
        report_args+=(--semantic-summary "$SEMANTIC_SCORES/summary.json")
    fi
elif [ "$JUDGE" = "gpt" ]; then
    : "${OPENAI_API_KEY:?export OPENAI_API_KEY before running the GPT judge}"
    base_url_args=()
    if [ -n "${OPENAI_BASE_URL:-}" ]; then
        base_url_args=(--base-url "$OPENAI_BASE_URL")
    fi
    echo "== GPT judge ($GPT_JUDGE_MODEL) -> $SCORES"
    reference prepare-judge "${COMMON_ARGS[@]}" --out "$SCORES" \
        || failed_phases=$((failed_phases + 1))
    reference judge "${COMMON_ARGS[@]}" --out "$SCORES" --judge "$GPT_JUDGE_MODEL" \
        --api-key-env OPENAI_API_KEY "${base_url_args[@]}" "$@" \
        || failed_phases=$((failed_phases + 1))
else
    echo "ERROR: JUDGE must be qwen or gpt, got '$JUDGE'." >&2
    exit 1
fi

echo "== Summarize timing, ASR coverage and behavior"
reference summarize "${COMMON_ARGS[@]}" --out "$SCORES" || failed_phases=$((failed_phases + 1))
reference report --scores "$SCORES" --engine "$ENGINE_LABEL" "${report_args[@]}" \
    > "$REPEAT_DIR/report.txt" \
    || failed_phases=$((failed_phases + 1))
echo "Wrote $REPEAT_DIR/report.txt"

if [ "$failed_phases" -gt 0 ]; then
    echo "WARNING: $failed_phases phase(s) reported failures." \
        "Rerun with: bash $(dirname "${BASH_SOURCE[0]}")/judge.sh $REPEAT --retry-failed"
    exit 1
fi
echo "Step 3 done. Aggregate repeats with: bash $(dirname "${BASH_SOURCE[0]}")/aggregate.sh"
