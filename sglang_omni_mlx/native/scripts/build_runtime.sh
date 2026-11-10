#!/usr/bin/env bash
# Builds the native Qwen3-ASR runtime into a self-contained directory.
# Note (XXXXRT666): MLX is built from a pinned main commit, since no release has
# ml-explore/mlx#4791 yet; the build is kept per commit under $work/mlx.
set -euo pipefail

output="${1:?usage: build_runtime.sh <output-directory> [<work-directory>]}"
work="${2:-$output/.build}"
native_dir="$(cd "$(dirname "$0")/.." && pwd)"

test "$(uname -m)" = arm64 || { echo "Apple Silicon is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }

deployment_target="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
mlx_commit=fc344c986c4169d99f76b5d95ae6b4a58a359a63
mlx_prefix="$work/mlx/$mlx_commit-macos$deployment_target"

mkdir -p "$work"
uv venv -q -p 3.12 --allow-existing "$work/venv"
VIRTUAL_ENV="$work/venv" uv pip install -q --python-platform aarch64-apple-darwin \
  -r "$native_dir/scripts/build-tools.lock"
export PATH="$work/venv/bin:$PATH"

if [[ ! -f "$mlx_prefix/share/cmake/MLX/MLXConfig.cmake" ]]; then
  git init -q "$work/mlx-source"
  git -C "$work/mlx-source" fetch -q --depth 1 https://github.com/ml-explore/mlx.git "$mlx_commit"
  git -C "$work/mlx-source" checkout -q --detach FETCH_HEAD
  cmake -S "$work/mlx-source" -B "$work/mlx-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DCMAKE_INSTALL_PREFIX="$mlx_prefix" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DCMAKE_OSX_ARCHITECTURES=arm64
  cmake --build "$work/mlx-build"
  cmake --install "$work/mlx-build"
fi

cmake -S "$native_dir" -B "$work/cmake" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_DIR="$mlx_prefix/share/cmake/MLX" -DCMAKE_INSTALL_PREFIX="$output" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
  -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build "$work/cmake"
cmake --install "$work/cmake"
echo "VOXT_OMNI_RUNTIME=$output/bin/qwen3_asr_server"
