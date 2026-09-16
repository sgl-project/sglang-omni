#!/bin/bash
# Synthetic audio only; does not access the microphone or local model services.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/recording-limit-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
  -module-cache-path "$test_dir/module-cache" \
  -sdk "$(xcrun --show-sdk-path)" \
  "$project_dir"/Sources/DictationCore/*.swift \
  "$project_dir/Sources/DictationApp/AudioCapture.swift" \
  "$project_dir/Tests/RecordingLimit_test.swift" \
  -o "$test_dir/RecordingLimit_test"
"$test_dir/RecordingLimit_test"
