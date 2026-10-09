#!/usr/bin/env bash
# Serve the Qwen behavior judge and write its pinned judge config.
# Runs in the foreground; stop with Ctrl-C.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

JUDGE_DIR="$FDB_WORK/judge"
mkdir -p "$JUDGE_DIR"
judge_revision=$(cat "$JUDGE_MODEL_PATH.revision")
sglang_version=$("$OMNI_VENV/bin/python" -c "import sglang; print(sglang.__version__)")
server_command=(
    "$OMNI_VENV/bin/sglang" serve
    --model-path "$JUDGE_MODEL_PATH"
    --served-model-name "$JUDGE_SERVED_MODEL"
    --host 127.0.0.1 --port "$JUDGE_PORT"
    --reasoning-parser qwen3
    --mem-fraction-static 0.8
)

"$OMNI_VENV/bin/python" - "$JUDGE_DIR" "$judge_revision" "$sglang_version" "${server_command[@]}" <<'EOF'
import hashlib
import json
import os
import sys
from pathlib import Path

judge_dir, revision, sglang_version, *command = sys.argv[1:]
judge_dir = Path(judge_dir)
receipt_path = judge_dir / "launch-receipt.json"
receipt_path.write_text(json.dumps({
    "server_command": command,
    "runtime": f"sglang {sglang_version}",
    "model_id": os.environ["JUDGE_MODEL_ID"],
    "model_revision": revision,
    "tokenizer_revision": revision,
    "precision": "bf16",
}, indent=2) + "\n")
config = {
    "model_id": os.environ["JUDGE_MODEL_ID"],
    "model_revision": revision,
    "tokenizer_id": os.environ["JUDGE_MODEL_ID"],
    "tokenizer_revision": revision,
    "served_model": os.environ["JUDGE_SERVED_MODEL"],
    "precision": "bf16",
    "enable_thinking": False,
    "decoding": {
        "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": 512,
    },
    "seeds": [1, 2, 3],
    "server_launch_receipt": receipt_path.name,
    "server_launch_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
}
(judge_dir / "judge-config.json").write_text(json.dumps(config, indent=2) + "\n")
print(f"wrote {judge_dir / 'judge-config.json'}")
EOF

CUDA_VISIBLE_DEVICES="$SCORING_GPU" exec "${server_command[@]}"
