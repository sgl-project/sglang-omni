#!/usr/bin/env bash
# One-time setup: reference checkout, dataset, scoring venv and model checkpoints.
# Safe to rerun; finished steps are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

mkdir -p "$FDB_WORK/models" "$FDB_WORK/dataset/zips"

echo "== [1/6] Full-Duplex-Bench reference checkout at $FDB_SOURCE_REVISION"
if [ ! -d "$FDB_SOURCE/.git" ]; then
    git clone https://github.com/DanielLin94144/Full-Duplex-Bench.git "$FDB_SOURCE"
fi
git -C "$FDB_SOURCE" checkout --quiet "$FDB_SOURCE_REVISION"
(cd "$FDB_SOURCE" && sha256sum --check --quiet) <<'EOF'
aedaee0d50f2bc47947caf6f3899939461e290225a98c0ddc434c360189596cf  v1_v1.5/get_transcript/asr.py
4f551da4194ab4d9584db964f4b27223914eecf98cbc312388ecf45eaf6f8a17  v1_v1.5/evaluation/get_timing.py
0ff8179a437503581d65787da3a43924b45310c31a98d1bcbadce5c8605ca6f2  v1_v1.5/evaluation/eval_behavior.py
19e5477dac9a9a1e11de126783a0b820b3ecb70db5e91181824fa944e1947977  v1_v1.5/evaluation/instruction/behavior.txt
EOF

echo "== [2/6] Scoring venv at $SCORING_VENV"
if [ ! -x "$SCORING_VENV/bin/python" ]; then
    uv venv --python 3.12 "$SCORING_VENV"
fi
VIRTUAL_ENV="$SCORING_VENV" uv pip install --quiet \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.11.0" "torchaudio==2.11.0" "nemo_toolkit[asr]==3.0.0" "silero-vad==6.2.1" \
    openai pydantic scipy soundfile tqdm websockets gdown "huggingface_hub[cli]"

echo "== [3/6] FDB v1.5 dataset at $FDB_DATASET"
declare -A DRIVE_FILE_IDS=(
    [user_interruption]=1wqYcYS4-30W2YMc3TfeMeoiaW9PLf4yT
    [user_backchannel]=1EGwCd9CdGuh8jeqwPENzbBs5FSEgnYL3
    [talking_to_other]=1Jh6ER4AUmqGgEZBTV0pcbDaMMQIWA7Kt
    [background_speech]=1W63k1BlQ0QCgvYCb_8YJNqFfUhBwI97W
)
for subset in "${!DRIVE_FILE_IDS[@]}"; do
    zip_path="$FDB_WORK/dataset/zips/$subset.zip"
    if [ ! -s "$zip_path" ]; then
        "$SCORING_VENV/bin/gdown" --quiet "${DRIVE_FILE_IDS[$subset]}" -O "$zip_path"
    fi
    if [ ! -d "$FDB_DATASET/$subset" ]; then
        "$SCORING_VENV/bin/python" -m zipfile -e "$zip_path" "$FDB_DATASET"
    fi
done
rm -rf "$FDB_DATASET/__MACOSX"
(cd "$FDB_WORK/dataset/zips" && sha256sum ./*.zip) > "$FDB_WORK/dataset/zips.sha256"
echo "sha256:$(sha256sum < "$FDB_WORK/dataset/zips.sha256" | cut -d' ' -f1)" \
    > "$FDB_WORK/dataset/v1.5.revision"
for subset in "${!DRIVE_FILE_IDS[@]}"; do
    echo "   $subset: $(find "$FDB_DATASET/$subset" -mindepth 1 -maxdepth 1 -type d | wc -l) samples"
done

echo "== [4/6] Parakeet ASR checkpoint"
if ! echo "$PARAKEET_SHA256  $PARAKEET_NEMO" | sha256sum --check --quiet 2>/dev/null; then
    "$SCORING_VENV/bin/hf" download nvidia/parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v2.nemo \
        --revision "$PARAKEET_REVISION" --local-dir "$(dirname "$PARAKEET_NEMO")"
    echo "$PARAKEET_SHA256  $PARAKEET_NEMO" | sha256sum --check --quiet
fi

echo "== [5/6] Model under test at $MODEL_PATH"
if [ ! -f "$MODEL_PATH/config.json" ]; then
    "$SCORING_VENV/bin/hf" download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_PATH"
fi

echo "== [6/6] Judge model at $JUDGE_MODEL_PATH (JUDGE=$JUDGE)"
if [ "$JUDGE" = "qwen" ] && [ ! -f "$JUDGE_MODEL_PATH.revision" ]; then
    judge_revision=$("$SCORING_VENV/bin/python" -c \
        "from huggingface_hub import model_info; print(model_info('$JUDGE_MODEL_ID').sha)")
    "$SCORING_VENV/bin/hf" download "$JUDGE_MODEL_ID" --revision "$judge_revision" \
        --local-dir "$JUDGE_MODEL_PATH"
    echo "$judge_revision" > "$JUDGE_MODEL_PATH.revision"
fi

echo "Setup complete."
