#!/bin/bash
# Synthetic audio only: no microphone, network, or input events.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/offline-tests"
mkdir -p "$test_dir/module-cache"
cat "$project_dir/Sources/DictationApp/AudioCapture.swift" \
    "$project_dir/Tests/CaptureStop_test.swift" > "$test_dir/CaptureStop_test.swift"
xcrun swiftc -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "$project_dir"/Sources/DictationCore/*.swift "$test_dir/CaptureStop_test.swift" \
    -o "$test_dir/CaptureStop_test"
/usr/bin/codesign --force --sign - "$test_dir/CaptureStop_test"
"$test_dir/CaptureStop_test"
