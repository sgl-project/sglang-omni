#!/bin/bash
# Delayed in-memory editor and isolated pasteboard; no real input events.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
test_dir="$project_dir/.build/correction-tests"
mkdir -p "$test_dir/module-cache"
xcrun swiftc -whole-module-optimization -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$(xcrun --show-sdk-path)" \
    "$project_dir"/Sources/DictationCore/*.swift \
    "$project_dir/Sources/DictationApp/ClipboardPaste.swift" \
    "$project_dir/Sources/DictationApp/TextInsertion.swift" \
    "$project_dir/Sources/DictationApp/EditableTextAnchor.swift" \
    "$project_dir/Tests/RevisionSelectionDelay_test.swift" -o "$test_dir/RevisionSelectionDelay_test"
/usr/bin/codesign --force --sign - "$test_dir/RevisionSelectionDelay_test"
"$test_dir/RevisionSelectionDelay_test"
