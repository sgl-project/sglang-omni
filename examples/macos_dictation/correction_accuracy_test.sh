#!/bin/bash
# Opt-in comparison against a source snapshot; never records or edits other apps.
set -euo pipefail
if [[ "${OMNI_CORRECTION_LIVE_TEST:-0}" != 1 ]]; then
    printf 'Set OMNI_CORRECTION_LIVE_TEST=1 to use the local Ollama server.\n' >&2
    exit 1
fi
project_dir="$(cd "$(dirname "$0")" && pwd)"
source_dir="${OMNI_CORRECTION_SOURCE_DIR:-$project_dir/Sources/DictationCore}"
mkdir -p "$project_dir/.build"
test_dir="$(mktemp -d "$project_dir/.build/correction-accuracy.XXXXXX")"
trap 'rm -rf "$test_dir"' EXIT
mkdir -p "$test_dir/module-cache"
xcrun swiftc -whole-module-optimization -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "$source_dir"/*.swift "$project_dir/Tests/CorrectionAccuracy_test.swift" -o "$test_dir/CorrectionAccuracy_test"
/usr/bin/codesign --force --sign - "$test_dir/CorrectionAccuracy_test"
"$test_dir/CorrectionAccuracy_test"
