#!/bin/bash
# Offline local edits and bounded model-result validation; no real model or editor.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/correction-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -whole-module-optimization -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "$project_dir"/Sources/DictationCore/*.swift "$project_dir/Tests/Support/HTTPStub.swift" \
    "$project_dir/Tests/CorrectionPlan_test.swift" -o "$test_dir/CorrectionPlan_test"
/usr/bin/codesign --force --sign - "$test_dir/CorrectionPlan_test"
"$test_dir/CorrectionPlan_test"
