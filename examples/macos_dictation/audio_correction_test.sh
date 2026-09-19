#!/bin/bash
# Opt-in: sends supplied files to local models and prints transcripts to the terminal.
set -euo pipefail
if [[ "${OMNI_CORRECTION_AUDIO_TEST:-0}" != 1 ]]; then
    printf 'Set OMNI_CORRECTION_AUDIO_TEST=1 to run the local audio correction test.\n' >&2
    exit 1
fi
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/correction-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -whole-module-optimization -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "$project_dir"/Sources/DictationCore/*.swift "$project_dir/Sources/DictationApp/AudioCapture.swift" \
    "$project_dir/Tests/AudioCorrection_test.swift" -o "$test_dir/AudioCorrection_test"
/usr/bin/codesign --force --sign - "$test_dir/AudioCorrection_test"
"$test_dir/AudioCorrection_test" "$@"
