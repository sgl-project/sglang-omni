#!/usr/bin/env bash
# Step 1: record overlap and clean sessions against the server, then export scoring audio.
# Usage: generate.sh REPEAT    (REPEAT is 1, 2, 3, ...)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

REPEAT="${1:?usage: generate.sh REPEAT}"
REPEAT_DIR="$RUN_ROOT/repeat-$REPEAT"
CAPABILITIES_URL="http://127.0.0.1:$SERVER_PORT/v1/realtime/capabilities"
SERVER_READY_TIMEOUT_S=1800
PROGRESS_INTERVAL_S=60

if [ -e "$REPEAT_DIR/recording" ]; then
    echo "ERROR: $REPEAT_DIR/recording exists. Use a new REPEAT, or delete $REPEAT_DIR to redo it." >&2
    exit 1
fi
mkdir -p "$REPEAT_DIR/logs" "$REPEAT_DIR/recording"
cd "$SGLANG_OMNI_ROOT"

echo "== Waiting for $CAPABILITIES_URL"
deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
until curl --silent --fail "$CAPABILITIES_URL" | grep -q '"native_full_duplex":true'; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "ERROR: server not ready after ${SERVER_READY_TIMEOUT_S}s; check the launch_server.sh terminal." >&2
        exit 1
    fi
    sleep 10
done

"$OMNI_VENV/bin/python" - "$FDB_DATASET" "$MAX_PER_SUBSET" > "$REPEAT_DIR/sample-ids.txt" <<'EOF'
import sys
from pathlib import Path

from benchmarks.duplex.v15_dataset import SUBSETS, select_sample_ids

dataset_root, max_per_subset = sys.argv[1:]
limit = int(max_per_subset) if max_per_subset else None
print("\n".join(select_sample_ids(Path(dataset_root), SUBSETS, None, limit)))
EOF
pair_count=$(wc -l < "$REPEAT_DIR/sample-ids.txt")
session_count=$((pair_count * 2))
dataset_revision=$(cat "$FDB_WORK/dataset/v1.5.revision")
server_revision=$(git rev-parse HEAD)
echo "== Recording $pair_count pairs ($session_count sessions) in $NUM_SHARDS shard(s) -> $REPEAT_DIR"

shard_pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    mapfile -t shard_ids < <(awk -v shards="$NUM_SHARDS" -v shard="$shard" \
        '(NR - 1) % shards == shard' "$REPEAT_DIR/sample-ids.txt")
    if [ "${#shard_ids[@]}" -eq 0 ]; then
        continue
    fi
    sample_args=()
    for sample_id in "${shard_ids[@]}"; do
        sample_args+=(--sample-id "$sample_id")
    done
    "$OMNI_VENV/bin/python" -m benchmarks.eval.benchmark_duplex_v15 record \
        --profile minicpmo-native-pr2377 \
        --dataset-root "$FDB_DATASET" --dataset-revision "$dataset_revision" \
        --url "$REALTIME_URL" \
        --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
        --server-revision "$server_revision" \
        --timeout "$SESSION_TIMEOUT_S" \
        --output "$REPEAT_DIR/recording/shard-$shard" \
        "${sample_args[@]}" > "$REPEAT_DIR/logs/record-shard-$shard.log" 2>&1 &
    shard_pids+=("$!")
done

while kill -0 "${shard_pids[@]}" 2>/dev/null; do
    sleep "$PROGRESS_INTERVAL_S"
    finished=$(find "$REPEAT_DIR/recording" -name report.json | wc -l)
    echo "   $(date +%H:%M:%S) finished sessions: $finished / $session_count"
done
failed_shards=0
for pid in "${shard_pids[@]}"; do
    wait "$pid" || failed_shards=$((failed_shards + 1))
done
for log in "$REPEAT_DIR"/logs/record-shard-*.log; do
    echo "-- $(basename "$log"): variant status"
    sed -n '/^{/,/^}/p' "$log" | "$OMNI_VENV/bin/python" -c \
        "import json, sys; print(json.dumps(json.load(sys.stdin)['variant_status']))" \
        || tail -n 20 "$log"
done
if [ "$failed_shards" -gt 0 ]; then
    echo "WARNING: $failed_shards shard(s) had non-passing sessions; they stay in the denominator."
fi

echo "== Exporting fixed-window scoring audio"
run_args=()
for shard_dir in "$REPEAT_DIR"/recording/shard-*; do
    run_args+=(--run "$shard_dir")
done
only_args=()
while read -r sample_id; do
    only_args+=(--only "$sample_id")
done < "$REPEAT_DIR/sample-ids.txt"
"$OMNI_VENV/bin/python" -m benchmarks.eval.benchmark_duplex_reference export \
    --engine "$ENGINE_LABEL" --trace-format realtime-pcm16-v1 \
    "${run_args[@]}" --dataset-root "$FDB_DATASET" "${only_args[@]}" \
    --out "$REPEAT_DIR/reference-audio"
echo "Step 1 done: $REPEAT_DIR/reference-audio"
