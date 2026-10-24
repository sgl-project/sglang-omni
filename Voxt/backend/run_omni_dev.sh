#!/usr/bin/env bash
# Builds and runs "Voxt Dev" with an isolated home directory.
#
#   run_omni_dev.sh build
#   run_omni_dev.sh run [--swift-backend]
#   run_omni_dev.sh install
#
# build compiles and embeds the native runtime; install copies the app to /Applications.
#
# Environment:
#   VOXT_OMNI_RUNTIME     Runtime binary to use (defaults to the bundled runtime).
#   VOXT_DEV_HOME         Home directory the app sees (default: ~/.voxt-omni-dev).
#   VOXT_SHARED_MODELS    Existing <root>/mlx-audio directory to share instead of downloading again.
#   DEVELOPER_DIR         Xcode to use (default: /Applications/Xcode.app/Contents/Developer).
set -euo pipefail

voxt_dir="$(cd "$(dirname "$0")/.." && pwd)"
runtime_dir="$voxt_dir/build/omni-runtime"
derived_data="$voxt_dir/build/omni-dev"
app_name="Voxt Dev"
app="$derived_data/Build/Products/Debug/$app_name.app"
app_bundle="$app"
app_executable="$app/Contents/MacOS/$app_name"
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
    runtime_bundle="$app/Contents/Resources/OmniRuntime/bin"
    mkdir -p "$runtime_bundle"
    ditto "$runtime_dir/bin" "$runtime_bundle"
    ;;
  run | install)
    test -d "$app" || { echo "Build first: $0 build" >&2; exit 1; }
    if [[ "$1" == "install" ]]; then
      install_dir="${VOXT_DEV_INSTALL_DIR:-/Applications}"
      installed_app="$install_dir/$app_name.app"
      mkdir -p "$install_dir"
      osascript -e "tell application \"$app_name\" to quit" >/dev/null 2>&1 || true
      rm -rf "$installed_app"
      ditto "$app" "$installed_app"
      xattr -dr com.apple.quarantine "$installed_app" >/dev/null 2>&1 || true
      app_bundle="$installed_app"
      app_executable="$installed_app/Contents/MacOS/$app_name"
    else
      :
    fi
    dev_home="${VOXT_DEV_HOME:-$HOME/.voxt-omni-dev}"
    model_root="$dev_home/Library/Application Support/Voxt/model-storage"
    mkdir -p "$model_root"
    if [[ -n "${VOXT_SHARED_MODELS:-}" && ! -e "$model_root/mlx-audio" ]]; then
      ln -s "$VOXT_SHARED_MODELS" "$model_root/mlx-audio"
    fi
    backend_env=()
    if [[ "${2:-}" != "--swift-backend" ]]; then
      runtime="${VOXT_OMNI_RUNTIME:-$app_bundle/Contents/Resources/OmniRuntime/bin/qwen3_asr_server}"
      test -x "$runtime" || { echo "Build the runtime first: $0 build" >&2; exit 1; }
      backend_env=(
        VOXT_ASR_BACKEND=omni
        VOXT_OMNI_RUNTIME="$runtime"
      )
    fi
    exec env CFFIXED_USER_HOME="$dev_home" ${backend_env[@]+"${backend_env[@]}"} "$app_executable"
    ;;
  *)
    sed -n '2,14p' "$0" >&2
    exit 2
    ;;
esac
