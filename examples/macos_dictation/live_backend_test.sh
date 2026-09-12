#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/client-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -swift-version 5 -parse-as-library \
  -module-cache-path "$test_dir/module-cache" \
  -sdk "$(xcrun --show-sdk-path)" \
  "$project_dir"/Sources/DictationCore/*.swift \
  "$project_dir"/Sources/DictationApp/AudioCapture.swift \
  "$project_dir/Tests/LiveBackend_test.swift" \
  -o "$test_dir/LiveBackend_test"
"$test_dir/LiveBackend_test" "${1:?请提供已有录音的路径}"
