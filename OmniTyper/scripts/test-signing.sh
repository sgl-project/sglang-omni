#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Uses the same local Keychain identity as build.sh; does not launch an app or change TCC.
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
APP="$TEST_DIR/OmniTyper.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp /usr/bin/true "$APP/Contents/MacOS/OmniTyper"
cp "$APP_ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"
echo first > "$APP/Contents/Resources/version"
bash "$APP_ROOT/scripts/sign.sh" "$APP"
codesign -d -r- "$APP" > "$TEST_DIR/first.requirement" 2>/dev/null
codesign -d --verbose=4 "$APP" 2>&1 | awk '/^CDHash=/' > "$TEST_DIR/first.signature"
echo second > "$APP/Contents/Resources/version"
bash "$APP_ROOT/scripts/sign.sh" "$APP"
codesign -d -r- "$APP" > "$TEST_DIR/second.requirement" 2>/dev/null
codesign -d --verbose=4 "$APP" 2>&1 | awk '/^CDHash=/' > "$TEST_DIR/second.signature"
diff "$TEST_DIR/first.requirement" "$TEST_DIR/second.requirement"
if diff "$TEST_DIR/first.signature" "$TEST_DIR/second.signature" >/dev/null; then
  echo 'The resource edit did not change the code signature.' >&2
  exit 1
fi
echo 'Changed bundle, same signing requirement; both signatures verified.'
