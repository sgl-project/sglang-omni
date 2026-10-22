#!/usr/bin/env bash
# Builds "Voxt Omni Dev" and runs it with an isolated home directory.
#
#   run_omni_dev.sh build
#   run_omni_dev.sh run [--swift-backend]
#
# build also builds the native Qwen3-ASR runtime (sglang_omni_mlx/native) into
# Voxt/build/omni-runtime; run starts the app on it.
#
# Environment:
#   VOXT_OMNI_RUNTIME     Runtime binary to use (default: Voxt/build/omni-runtime/bin/qwen3_asr_server).
#   VOXT_DEV_HOME         Home directory the app sees (default: ~/.voxt-omni-dev).
#   VOXT_SHARED_MODELS    Existing <root>/mlx-audio directory to share instead of downloading again.
#   DEVELOPER_DIR         Xcode to use (default: /Applications/Xcode.app/Contents/Developer).
set -euo pipefail

voxt_dir="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="$voxt_dir/build/omni-runtime"
derived_data="$voxt_dir/build/omni-dev"
app="$derived_data/Build/Products/Debug/Voxt Omni Dev.app"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

case "${1:-}" in
  build)
    "$voxt_dir/../sglang_omni_mlx/native/scripts/build_runtime.sh" "$runtime_dir"
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
      runtime="${VOXT_OMNI_RUNTIME:-$runtime_dir/bin/qwen3_asr_server}"
      test -x "$runtime" || { echo "Build the runtime first: $0 build" >&2; exit 1; }
      backend_env=(
        VOXT_ASR_BACKEND=omni
        VOXT_OMNI_RUNTIME="$runtime"
      )
    fi
    exec env CFFIXED_USER_HOME="$dev_home" ${backend_env[@]+"${backend_env[@]}"} "$app/Contents/MacOS/Voxt Omni Dev"
    ;;
  *)
    sed -n '2,14p' "$0" >&2
    exit 2
    ;;
esac
