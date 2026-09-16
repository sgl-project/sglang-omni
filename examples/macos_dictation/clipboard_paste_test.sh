#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/clipboard-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -swift-version 5 -parse-as-library \
  -module-cache-path "$test_dir/module-cache" \
  -sdk "$(xcrun --show-sdk-path)" \
  "$project_dir"/Sources/DictationCore/*.swift \
  "$project_dir/Sources/DictationApp/ClipboardPaste.swift" \
  "$project_dir/Tests/ClipboardPaste_test.swift" \
  -o "$test_dir/ClipboardPaste_test"
"$test_dir/ClipboardPaste_test"
