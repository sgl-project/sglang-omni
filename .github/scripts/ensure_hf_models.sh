#!/usr/bin/env bash
# Ensure CI model *weights* are present in the Hugging Face Hub cache.
#
# Datasets are unchanged — tests continue to resolve HF datasets as today.
#
# Per model repo id:
#   1. If the native HF cache already has complete weights -> OK.
#   2. Otherwise download directly from huggingface.co into the native HF cache.
#   3. Validate the resulting snapshot; fail setup if weights are incomplete.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <venv-name> <hf-repo-id> [<hf-repo-id> ...]" >&2
  exit 1
fi

VENV_NAME="$1"
shift

export HOME="${HOME:-/github/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/github/home/.cache}"
export HF_HOME="${HF_HOME:-/github/home/.cache/huggingface}"
export HF_ENDPOINT="https://huggingface.co"
export HF_HUB_DISABLE_XET=0
unset HF_HUB_ENABLE_HF_TRANSFER

source "${VENV_NAME}/bin/activate"

python - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def weights_ready(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False

    indexes = [
        path
        for path in model_dir.rglob("*.index.json")
        if path.name.endswith((".safetensors.index.json", ".bin.index.json"))
    ]
    if indexes:
        for index_path in indexes:
            try:
                weight_map = json.loads(index_path.read_text()).get("weight_map", {})
            except (OSError, json.JSONDecodeError):
                return False
            shard_names = set(weight_map.values())
            if not shard_names or any(
                not (index_path.parent / shard_name).is_file()
                for shard_name in shard_names
            ):
                return False
        return True

    return any(
        path.is_file() and path.suffix in (".safetensors", ".bin")
        for path in model_dir.rglob("*")
    )


def hf_cache_snapshot(repo_id: str) -> Path | None:
    try:
        snapshot = Path(
            snapshot_download(repo_id, repo_type="model", local_files_only=True)
        )
    except Exception:
        return None
    if not weights_ready(snapshot):
        return None
    return snapshot


def download_via_hf_hub(repo_id: str) -> Path:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    endpoint = os.environ["HF_ENDPOINT"]
    print(f"Downloading model weights via Hugging Face Hub ({endpoint}): {repo_id}")
    snapshot = Path(
        snapshot_download(
            repo_id,
            repo_type="model",
            token=token,
            endpoint=endpoint,
        )
    )
    if not weights_ready(snapshot):
        raise SystemExit(
            f"Hugging Face download for {repo_id} finished but weights are "
            f"incomplete under {snapshot}"
        )
    print(f"Hugging Face download complete: {repo_id} -> {snapshot}")
    return snapshot


def ensure_model(repo_id: str) -> Path:
    local_path = Path(repo_id)
    if local_path.is_dir():
        if not weights_ready(local_path):
            raise SystemExit(
                f"Local model path {repo_id} is missing weight files"
            )
        print(f"OK local path: {repo_id}")
        return local_path

    cached = hf_cache_snapshot(repo_id)
    if cached is not None:
        print(f"OK HF cache: {repo_id} -> {cached}")
        return cached

    return download_via_hf_hub(repo_id)


for model_id in sys.argv[1:]:
    ensure_model(model_id)
PY

echo "All model weights verified"
