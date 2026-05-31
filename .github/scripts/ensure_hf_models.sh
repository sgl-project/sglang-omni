#!/usr/bin/env bash
# Ensure CI model *weights* are present in the HuggingFace hub cache.
#
# Datasets are unchanged — tests continue to resolve HF datasets as today.
#
# Per model repo id:
#   1. If HF hub cache already has weights -> OK.
#   2. Else if ModelScope cache has complete weights -> seed HF hub cache.
#   3. Else download from ModelScope -> seed HF hub cache.
#   4. Re-check HF hub cache (local_files_only); fail setup if still missing.
#
# ModelScope artifacts are symlinked into the HF cache layout so existing CI
# code paths (snapshot_download(repo_id), server --model-path repo id) work
# without modification.
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
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/github/home/.cache/modelscope}"

source "${VENV_NAME}/bin/activate"

python - "$@" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download


def hf_hub_dir() -> Path:
    return Path(os.environ["HF_HOME"]) / "hub"


def modelscope_model_dir(repo_id: str) -> Path:
    root = Path(os.environ["MODELSCOPE_CACHE"])
    return root / "hub" / "models" / repo_id


def weights_ready(model_dir: Path) -> bool:
    if not (model_dir / "config.json").exists():
        return False
    weight_files = [
        path
        for path in model_dir.rglob("*")
        if path.is_file()
        and path.suffix in (".safetensors", ".bin")
        and not path.name.endswith(".incomplete")
    ]
    return bool(weight_files)


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


def hf_repo_commit(repo_id: str) -> str:
    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
    return api.repo_info(repo_id, repo_type="model").sha


def seed_hf_cache_from_local(repo_id: str, local_dir: Path) -> Path:
    commit = hf_repo_commit(repo_id)
    repo_cache = hf_hub_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshot_dir = repo_cache / "snapshots" / commit
    refs_dir = repo_cache / "refs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(f"{commit}\n")

    for src in local_dir.rglob("*"):
        if not src.is_file() or src.name.startswith("."):
            continue
        dest = snapshot_dir / src.relative_to(local_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(src.resolve())

    return snapshot_dir


def modelscope_weights_dir(repo_id: str) -> Path | None:
    cache_dir = modelscope_model_dir(repo_id)
    if weights_ready(cache_dir):
        return cache_dir
    return None


def download_via_modelscope(repo_id: str) -> Path:
    cached = modelscope_weights_dir(repo_id)
    if cached is not None:
        print(f"ModelScope cache hit: {repo_id} -> {cached}")
        return cached

    print(f"HF cache miss; downloading model weights via ModelScope: {repo_id}")
    ms_dir = Path(ms_snapshot_download(repo_id))
    cached = modelscope_weights_dir(repo_id)
    if cached is None:
        raise SystemExit(
            f"ModelScope download for {repo_id} finished but weights look incomplete "
            f"under {ms_dir}"
        )
    print(f"ModelScope download complete: {repo_id} -> {cached}")
    return cached


def verify_hf_cache(repo_id: str) -> Path:
    cached = hf_cache_snapshot(repo_id)
    if cached is None:
        raise SystemExit(
            f"HF cache verification failed for model {repo_id}: "
            "snapshot missing or weight files incomplete"
        )
    print(f"Verified HF cache: {repo_id} -> {cached}")
    return cached


def hf_download_endpoints() -> list[str]:
    """Prefer configured mirror, then huggingface.co (private/large weights on repro hosts)."""
    endpoints: list[str] = []
    configured = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    if configured:
        endpoints.append(configured)
    direct = "https://huggingface.co"
    if direct not in endpoints:
        endpoints.append(direct)
    return endpoints


def download_via_hf_hub(repo_id: str) -> Path:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    last_error: Exception | None = None
    for endpoint in hf_download_endpoints():
        print(
            f"Downloading model weights via HuggingFace Hub ({endpoint}): {repo_id}"
        )
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id,
                    repo_type="model",
                    token=token,
                    endpoint=endpoint,
                )
            )
        except Exception as exc:
            last_error = exc
            print(
                f"HuggingFace download via {endpoint} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if weights_ready(snapshot):
            print(f"HuggingFace download complete: {repo_id} -> {snapshot}")
            return snapshot
        print(
            f"HuggingFace download via {endpoint} finished but weights look "
            f"incomplete under {snapshot}"
        )
    if last_error is not None:
        raise SystemExit(
            f"HuggingFace download for {repo_id} failed on all endpoints: {last_error}"
        )
    raise SystemExit(
        f"HuggingFace download for {repo_id} finished but weights look incomplete"
    )


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

    ms_dir = modelscope_weights_dir(repo_id)
    if ms_dir is None:
        try:
            ms_dir = download_via_modelscope(repo_id)
        except Exception as exc:
            print(
                f"ModelScope unavailable for {repo_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            ms_dir = None

    if ms_dir is not None:
        seeded = seed_hf_cache_from_local(repo_id, ms_dir)
        print(f"Seeded HF cache from ModelScope: {repo_id} -> {seeded}")
        return verify_hf_cache(repo_id)

    return download_via_hf_hub(repo_id)


for model_id in sys.argv[1:]:
    ensure_model(model_id)
PY

echo "All model weights verified"
