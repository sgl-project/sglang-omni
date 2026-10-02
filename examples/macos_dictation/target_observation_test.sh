#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/target-observation-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
  -module-cache-path "$test_dir/module-cache" \
  -sdk "$(xcrun --show-sdk-path)" \
  "$project_dir"/Sources/DictationCore/*.swift \
  "$project_dir/Sources/DictationApp/ClipboardPaste.swift" \
  "$project_dir/Sources/DictationApp/TextInsertion.swift" \
  "$project_dir/Tests/TargetObservation_test.swift" \
  -o "$test_dir/TargetObservation_test"
"$test_dir/TargetObservation_test"
