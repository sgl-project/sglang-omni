#!/usr/bin/env bash
# Builds "Voxt Omni Dev" and runs it with an isolated home directory.
#
#   run_omni_dev.sh build
#   run_omni_dev.sh run [--swift-backend]
#
# Environment:
#   VOXT_OMNI_PYTHON      Python with sglang-omni and the MLX extras installed (required to run).
#   VOXT_DEV_HOME         Home directory the app sees (default: ~/.voxt-omni-dev).
#   VOXT_SHARED_MODELS    Existing <root>/mlx-audio directory to share instead of downloading again.
#   DEVELOPER_DIR         Xcode to use (default: /Applications/Xcode.app/Contents/Developer).
set -euo pipefail

voxt_dir="$(cd "$(dirname "$0")/.." && pwd)"
backend_dir="$voxt_dir/backend"
derived_data="$voxt_dir/build/omni-dev"
app="$derived_data/Build/Products/Debug/Voxt Omni Dev.app"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

case "${1:-}" in
  build)
    xcodebuild build \
      -project "$voxt_dir/Voxt.xcodeproj" \
      -scheme Voxt \
      -configuration Debug \
      -destination 'platform=macOS' \
      -xcconfig "$voxt_dir/Config/OmniDev.xcconfig" \
      -derivedDataPath "$derived_data" \
      -skipPackagePluginValidation
    ;;
  run)
    test -d "$app" || { echo "Build first: $0 build" >&2; exit 1; }
    dev_home="${VOXT_DEV_HOME:-$HOME/.voxt-omni-dev}"
    model_root="$dev_home/Library/Application Support/Voxt/model-storage"
    mkdir -p "$model_root"
    if [[ -n "${VOXT_SHARED_MODELS:-}" && ! -e "$model_root/mlx-audio" ]]; then
      ln -s "$VOXT_SHARED_MODELS" "$model_root/mlx-audio"
    fi
    backend_env=()
    if [[ "${2:-}" != "--swift-backend" ]]; then
      : "${VOXT_OMNI_PYTHON:?set VOXT_OMNI_PYTHON to the backend Python}"
      backend_env=(
        VOXT_ASR_BACKEND=omni
        VOXT_OMNI_PYTHON="$VOXT_OMNI_PYTHON"
        VOXT_OMNI_BACKEND_DIR="$backend_dir"
      )
    fi
    exec env CFFIXED_USER_HOME="$dev_home" ${backend_env[@]+"${backend_env[@]}"} "$app/Contents/MacOS/Voxt Omni Dev"
    ;;
  *)
    sed -n '2,11p' "$0" >&2
    exit 2
    ;;
esac
