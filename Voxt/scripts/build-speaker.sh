#!/bin/bash
# Build the independent, audio-only diarization process. Never installs or
# launches Voxt. Models are downloaded through the app's shared model library.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/resources/darwin-arm64/speaker"
if [[ "${1:-}" == "--out" ]]; then
  OUT="${2:?--out needs a directory}"
fi
mkdir -p "$OUT"
swift build --package-path "$ROOT/speaker" --disable-keychain -c release -j "${SWIFT_BUILD_JOBS:-1}" --product voxtnew-speaker
BIN="$(swift build --package-path "$ROOT/speaker" --disable-keychain -c release --show-bin-path)/voxtnew-speaker"
cp "$BIN" "$OUT/voxtnew-speaker"
codesign --force --sign "${VOXTNEW_SIGN_IDENTITY:--}" --options runtime "$OUT/voxtnew-speaker"
