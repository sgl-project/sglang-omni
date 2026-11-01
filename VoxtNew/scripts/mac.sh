#!/usr/bin/env bash
# Mirrors this project to a Mac and runs a command there, for developing on
# Linux. The Mac needs Xcode (for the Swift helper) and Go (GOTOOLCHAIN=auto
# fetches the Go the module asks for).
#
#   scripts/mac.sh go tool mygo build
#   scripts/mac.sh go test ./internal/permission
#   scripts/mac.sh --pull build/darwin-arm64   # copy a path back
#
# MAC_HOST (default mac-air) and MAC_DIR (default code/VoxtNew) pick the
# target; MAC_GOPROXY its Go module proxy (proxy.golang.org times out there).
# The Omni runtime's sources (../sglang_omni_mlx/native, 3 MB) go along to
# MAC_DIR/.omni-native, for scripts/build-omni.sh.
#
# ACTNOW_INGEST_TOKEN, when set here, reaches the command's environment
# through ssh's standard input (never a command line, which the Mac's ps
# would show), for scripts/release.sh --publish (docs/release.md §2.1):
#
#   ACTNOW_INGEST_TOKEN="$(sed -n 's/^RELEASE_INGEST_TOKEN_VOXTNEW=//p' ~/.config/actnow/ingest-tokens.env)" \
#     scripts/mac.sh 'scripts/mac-gui.sh scripts/release.sh --notarize voxtnew-notary --publish'
#
# The command then reads an empty standard input.
set -euo pipefail

HOST="${MAC_HOST:-mac-air}"
GOPROXY_MAC="${MAC_GOPROXY:-https://goproxy.cn,direct}"
DIR="${MAC_DIR:-code/VoxtNew}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${1:-}" == "--pull" ]]; then
  shift
  for p in "$@"; do
    mkdir -p "$ROOT/$(dirname "$p")"
    rsync -a --delete "$HOST:$DIR/$p/" "$ROOT/$p/"
  done
  exit 0
fi

ssh "$HOST" "mkdir -p '$DIR'"
rsync -a --delete \
  --exclude .git --exclude /build --exclude /.mygo \
  --exclude /helper/.build --exclude /helper/.swiftpm --exclude /llm/.build --exclude /llm/.swiftpm --exclude /resources/darwin-arm64 \
  --exclude /VoxtNew --exclude /.omni-native \
  "$ROOT/" "$HOST:$DIR/"
if [[ -d "$ROOT/../sglang_omni_mlx/native" ]]; then
  rsync -a --delete "$ROOT/../sglang_omni_mlx/native/" "$HOST:$DIR/.omni-native/"
fi

[[ $# -eq 0 ]] && exit 0
setup="cd '$DIR' && export PATH=/opt/homebrew/bin:\$PATH GOTOOLCHAIN=auto GOPROXY='$GOPROXY_MAC'"
if [[ -n "${ACTNOW_INGEST_TOKEN:-}" ]]; then
  # shellcheck disable=SC2029
  printf '%s\n' "$ACTNOW_INGEST_TOKEN" |
    ssh "$HOST" "IFS= read -r ACTNOW_INGEST_TOKEN && export ACTNOW_INGEST_TOKEN && $setup && $*"
else
  # shellcheck disable=SC2029
  ssh "$HOST" "$setup && $*"
fi
