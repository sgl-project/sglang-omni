#!/usr/bin/env bash
# Builds voxtnew-helper (helper/, Swift, release, arm64) on a Mac and
# stages it where mygo ships it: resources/darwin-arm64/helper/ becomes
# VoxtNew.app/Contents/Resources/helper/ (go tool mygo build, mygo dev).
#
#   scripts/build-helper.sh              # stage it for the app
#   scripts/build-helper.sh --out DIR    # build it into DIR instead: a helper
#                                        # outside the app, for VOXTNEW_HELPER
#   scripts/mac.sh 'scripts/build-helper.sh && go tool mygo build'
#
# SWIFT_BUILD_JOBS limits swift build's jobs (scripts/mac-e2e.sh: 2).
#
# It is signed with helper/Support/helper.entitlements and the hardened
# runtime, ad hoc unless VOXTNEW_SIGN_IDENTITY names an identity (see
# "Signing during development" in README.md: an ad hoc rebuild of the app
# makes macOS forget its permissions). mygo build keeps a valid signature.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "voxtnew-helper builds on macOS: scripts/mac.sh scripts/build-helper.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/resources/darwin-arm64/helper"
if [[ "${1:-}" == "--out" ]]; then
  [[ -n "${2:-}" ]] || { echo "--out needs a directory" >&2; exit 1; }
  OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
fi
IDENTITY="${VOXTNEW_SIGN_IDENTITY:--}"

# The helper reports the app's version (helper.hello): keep them in step.
app_version="$(sed -n 's/^  "version": *"\([^"]*\)".*/\1/p' "$ROOT/mygo.json")"
helper_version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$ROOT/helper/Support/Info.plist")"
if [[ "$app_version" != "$helper_version" ]]; then
  echo "helper/Support/Info.plist says $helper_version, mygo.json $app_version: make them agree" >&2
  exit 1
fi

cd "$ROOT/helper"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
# Linked again each time: Support/Info.plist goes in at link time
# (Package.swift), and swift build does not see it change.
rm -f .build/arm64-apple-macosx/release/voxtnew-helper
if ! swift build -c release --arch arm64 ${SWIFT_BUILD_JOBS:+-j "$SWIFT_BUILD_JOBS"} >"$log" 2>&1; then
  grep -E "error:" "$log" | sort -u | head -40 >&2 || cat "$log" >&2
  echo "swift build failed" >&2
  exit 1
fi
grep -E "warning:" "$log" | sort -u || true
embedded="$(grep -a -A1 '<key>CFBundleShortVersionString</key>' .build/arm64-apple-macosx/release/voxtnew-helper | sed -n 's/.*<string>\(.*\)<\/string>.*/\1/p' | head -1)"
if [[ "$embedded" != "$helper_version" ]]; then
  echo "the helper was linked with version ${embedded:-none}, not $helper_version" >&2
  exit 1
fi
mkdir -p "$OUT"
cp -f .build/arm64-apple-macosx/release/voxtnew-helper "$OUT/voxtnew-helper"
codesign --force --sign "$IDENTITY" --entitlements Support/helper.entitlements --options runtime "$OUT/voxtnew-helper"
echo "built $OUT/voxtnew-helper ($helper_version, signed ${IDENTITY/#-/ad hoc})"
