#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
app_dir="${1:-$project_dir/.build/client/OmniDictation.app}"
binary="$app_dir/Contents/MacOS/OmniDictation"
minimum=$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$app_dir/Contents/Info.plist")
compiled_minimum=$(xcrun vtool -show-build "$binary" | awk '$1 == "minos" { print $2 }')
if [[ "$minimum" != "14.0" || "$compiled_minimum" != "$minimum" ]]; then
  printf 'FAIL: declared macOS %s, binary requires macOS %s\n' "$minimum" "$compiled_minimum" >&2
  exit 1
fi
[[ "$(xcrun lipo -archs "$binary")" == "arm64" ]]
/usr/bin/codesign --verify --strict "$app_dir"
printf 'PASS: Apple Silicon binary and bundle both target macOS 14.0; signature valid\n'
