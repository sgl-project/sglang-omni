#!/bin/bash
# Mock ASR/Ollama and editor; no hardware or network access.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/correction-tests"
mkdir -p "$test_dir/module-cache"
sources=("$project_dir"/Sources/DictationCore/*.swift)
for name in ClientState ClientPreferences ServicePreferences RecordingShortcut AudioCapture ClipboardPaste CurrentCursorTarget TextInsertion; do
    sources+=("$project_dir/Sources/DictationApp/$name.swift")
done
xcrun swiftc -whole-module-optimization -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "${sources[@]}" "$project_dir/Tests/Support/HTTPStub.swift" \
    "$project_dir/Tests/CorrectionIntegration_test.swift" -o "$test_dir/CorrectionIntegration_test"
/usr/bin/codesign --force --sign - "$test_dir/CorrectionIntegration_test"
"$test_dir/CorrectionIntegration_test"
