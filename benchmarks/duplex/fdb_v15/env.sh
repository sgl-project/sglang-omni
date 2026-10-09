# Shared settings for the FDB v1.5 scripts; every script sources this file.
# Put machine-specific overrides in $FDB_LOCAL_ENV, or export them before running.

FDB_LOCAL_ENV="${FDB_LOCAL_ENV:-$HOME/.config/fdb_v15.env}"
if [ -f "$FDB_LOCAL_ENV" ]; then
    source "$FDB_LOCAL_ENV"
fi

export SGLANG_OMNI_ROOT="${SGLANG_OMNI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export FDB_WORK="${FDB_WORK:-$HOME/fdb}"

# Python environments: OMNI_VENV serves the models, SCORING_VENV runs NeMo ASR and timing.
export OMNI_VENV="${OMNI_VENV:-${VIRTUAL_ENV:-$SGLANG_OMNI_ROOT/.venv}}"
export SCORING_VENV="${SCORING_VENV:-$FDB_WORK/scoring-venv}"

# Pinned external inputs, all created by setup.sh.
export FDB_SOURCE="$FDB_WORK/Full-Duplex-Bench"
export FDB_SOURCE_REVISION="3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
export FDB_DATASET="$FDB_WORK/dataset/v1.5"
export PARAKEET_NEMO="$FDB_WORK/models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"
export PARAKEET_REVISION="ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
export PARAKEET_SHA256="d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"

# Model under test.
export MODEL_ID="openbmb/MiniCPM-o-4_5"
export MODEL_REVISION="${MODEL_REVISION:-503e754207c94da6bb26850b4469f367c9ea3582}"
export MODEL_PATH="${MODEL_PATH:-$FDB_WORK/models/MiniCPM-o-4_5}"
export SERVER_CONFIG="${SERVER_CONFIG:-$SGLANG_OMNI_ROOT/examples/full_duplex/minicpmo.yaml}"
export SERVER_PORT="${SERVER_PORT:-8097}"
export SERVER_GPU="${SERVER_GPU:-0}"
export REALTIME_URL="ws://127.0.0.1:$SERVER_PORT/v1/realtime"
export ENGINE_LABEL="minicpmo"

# Behavior judge: "qwen" (local Qwen3.8-27B through SGLang) or "gpt" (gpt-4o-2024-08-06).
export JUDGE="${JUDGE:-qwen}"
export JUDGE_MODEL_ID="Qwen/Qwen3.8-27B"
export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$FDB_WORK/models/Qwen3.8-27B}"
export JUDGE_SERVED_MODEL="qwen3.8-27b"
export JUDGE_PORT="${JUDGE_PORT:-30000}"
export JUDGE_URL="http://127.0.0.1:$JUDGE_PORT/v1"
export CUSTOM_JUDGE_API_KEY="${CUSTOM_JUDGE_API_KEY:-EMPTY}"

# ASR and the judge server share one GPU; the generation server keeps its own.
export SCORING_GPU="${SCORING_GPU:-1}"

# Selection. MAX_PER_SUBSET=12 gives 48 pairs (12 per category); empty runs all 498 pairs.
export RUN_NAME="${RUN_NAME:-minicpmo-48}"
export MAX_PER_SUBSET="${MAX_PER_SUBSET-12}"
# Parallel recording sessions against one server; must not exceed max_sessions in SERVER_CONFIG.
export NUM_SHARDS="${NUM_SHARDS:-1}"
export SESSION_TIMEOUT_S="${SESSION_TIMEOUT_S:-90}"

export RUN_ROOT="$FDB_WORK/runs/$RUN_NAME"
