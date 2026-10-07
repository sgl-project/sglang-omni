#!/usr/bin/env bash
# Creates the pinned Python environment the Voxt Omni backend runs in.
#
#   setup_env.sh <environment-directory>
#
# Installs the pinned packages of requirements-mac.lock (MLX, the tokenizer and
# a small web server) into a Python 3.12 virtual environment, then this
# checkout's sglang-omni without its dependencies: the backend runs only its
# standalone sglang_omni_mlx package. Needs uv. Nothing outside the target
# directory changes.
set -euo pipefail

target="${1:?usage: setup_env.sh <environment-directory>}"
backend_dir="$(cd "$(dirname "$0")" && pwd)"
omni_dir="$(cd "$backend_dir/../.." && pwd)"

test "$(uname -m)" = arm64 || { echo "Apple Silicon is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }

mkdir -p "$target"
uv venv -p 3.12 "$target/venv"
export VIRTUAL_ENV="$target/venv"
uv pip install -r "$backend_dir/requirements-mac.lock"
uv pip install --no-deps -e "$omni_dir"

"$target/venv/bin/python" - <<'PY'
import mlx.core as mx
import sglang_omni_mlx.qwen3_asr.server
assert mx.metal.is_available()
print("environment ready")
PY
echo "VOXT_OMNI_PYTHON=$target/venv/bin/python"
