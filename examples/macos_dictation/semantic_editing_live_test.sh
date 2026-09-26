#!/bin/bash
# Opt-in local model evaluation, with optional baseline source snapshot.
set -euo pipefail
[[ "${OMNI_CORRECTION_LIVE_TEST:-0}" == 1 ]] || { printf 'Set OMNI_CORRECTION_LIVE_TEST=1.\n' >&2; exit 1; }
project_dir="$(cd "$(dirname "$0")" && pwd)"
source_dir="${OMNI_CORRECTION_SOURCE_DIR:-$project_dir/Sources/DictationCore}"
test_dir="$project_dir/.build/semantic-editing-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -whole-module-optimization -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "$source_dir"/*.swift "$project_dir/Tests/SemanticEditingLive_test.swift" -o "$test_dir/SemanticEditingLive_test"
/usr/bin/codesign --force --sign - "$test_dir/SemanticEditingLive_test"
"$test_dir/SemanticEditingLive_test"
