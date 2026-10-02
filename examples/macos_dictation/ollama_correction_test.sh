#!/bin/bash
# URLProtocol stubs only; does not contact a model server.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/correction-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -whole-module-optimization -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "$project_dir"/Sources/DictationCore/*.swift "$project_dir/Tests/Support/HTTPStub.swift" \
    "$project_dir/Tests/OllamaCorrection_test.swift" -o "$test_dir/OllamaCorrection_test"
/usr/bin/codesign --force --sign - "$test_dir/OllamaCorrection_test"
"$test_dir/OllamaCorrection_test"
