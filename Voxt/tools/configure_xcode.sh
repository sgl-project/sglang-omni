#!/usr/bin/env bash
# Source this script before local builds. CI may pass --select-system on its dedicated runner.
set -euo pipefail
if [[ "$(uname -s)" != Darwin ]]; then
  echo "Xcode validation requires macOS." >&2
  return 1 2>/dev/null || exit 1
fi
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode_26.5.app/Contents/Developer}"
if [[ "${1:-}" == --select-system ]]; then
  sudo xcode-select --switch "$DEVELOPER_DIR"
fi
# Read the system selection without the environment override; MLX CPU JIT child
# processes must not discover a different Xcode while inheriting XCTest libraries.
SYSTEM_DEVELOPER_DIR="$(env -u DEVELOPER_DIR xcode-select --print-path)"
if [[ "$SYSTEM_DEVELOPER_DIR" != "$DEVELOPER_DIR" ]]; then
  echo "System Xcode differs: $SYSTEM_DEVELOPER_DIR" >&2
  echo "Select $DEVELOPER_DIR with xcode-select before running model tests." >&2
  return 1 2>/dev/null || exit 1
fi
xcrun --kill-cache
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
test -d "$SDKROOT"
export PATH="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin:$PATH"
xcodebuild -version
xcrun swift --version
xcrun --sdk macosx --find clang++
if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'DEVELOPER_DIR=%s\nSDKROOT=%s\n' "$DEVELOPER_DIR" "$SDKROOT" >> "$GITHUB_ENV"
  printf '%s\n' "$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin" >> "$GITHUB_PATH"
fi
