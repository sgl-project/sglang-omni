#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/insertion-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -swift-version 5 -parse-as-library \
  -module-cache-path "$test_dir/module-cache" \
  -sdk "$(xcrun --show-sdk-path)" \
  "$project_dir"/Sources/DictationCore/*.swift \
  "$project_dir/Tests/Insertion_test.swift" \
  -o "$test_dir/Insertion_test"
"$test_dir/Insertion_test"
