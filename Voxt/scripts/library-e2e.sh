#!/bin/bash
# The library transfer end to end in the app binary (smoke_library.go):
# Mac A exports, Mac B imports and restarts, then B's library is A's.
# Each Mac is a --smoke folder of its own under a temporary HOME, so
# nothing of this Mac's Voxt is read or written.
#
#   scripts/library-e2e.sh [Voxt binary]
#
# The binary defaults to build/darwin-arm64/Voxt.app/Contents/MacOS/Voxt.
set -euo pipefail
BIN=$(cd "$(dirname "${1:-build/darwin-arm64/Voxt.app/Contents/MacOS/Voxt}")" && pwd)/$(basename "${1:-Voxt}")
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
FILE="$OUT/Voxt.voxtlibrary"
run() { # <mac> <mode>
  mkdir -p "$OUT/$1/data" "$OUT/home"
  if ! HOME="$OUT/home" ACTNOW_FAMILY_DIR="$OUT/$1/ActNow" VOXTNEW_SMOKE_LIBRARY="$2" VOXTNEW_SMOKE_LIBRARY_FILE="$FILE" \
    "$BIN" --smoke "$OUT/$1" >"$OUT/$1-$2.log" 2>&1; then
    echo "library-e2e: FAIL Mac $1 $2"
    tail -30 "$OUT/$1-$2.log"
    exit 1
  fi
  grep '^smoke: library' "$OUT/$1-$2.log"
}
run A export
run B import
run B check
echo "library-e2e: ok"
