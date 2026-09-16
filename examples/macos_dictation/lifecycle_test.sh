#!/bin/bash
# Offline lifecycle checks using fake HTTP, a fake recorder and disposable processes.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_python="${SGLANG_OMNI_VENV:-$project_dir/../../.venv-apple}/bin/python"
if [[ ! -x "$test_python" ]]; then test_python="$(command -v python3)"; fi
"$test_python" "$project_dir/Tests/RuntimeLifecycle_test.py" -v
test_dir="$project_dir/.build/offline-tests"
mkdir -p "$test_dir/module-cache"
app_sources=(ClientState ClientPreferences ServicePreferences RecordingShortcut
             AudioCapture ClipboardPaste CurrentCursorTarget TextInsertion)
sources=("$project_dir"/Sources/DictationCore/*.swift)
for name in "${app_sources[@]}"; do
    sources+=("$project_dir/Sources/DictationApp/$name.swift")
done
xcrun swiftc -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "${sources[@]}" "$project_dir/Tests/Support/HTTPStub.swift" \
    "$project_dir/Tests/WarmupLifecycle_test.swift" -o "$test_dir/WarmupLifecycle_test"
"$test_dir/WarmupLifecycle_test"
