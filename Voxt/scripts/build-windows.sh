#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(go env GOOS)" != "windows" ]]; then
  echo "The Windows NSIS installer must be built on Windows." >&2
  exit 1
fi

mkdir -p resources/windows-amd64
CGO_ENABLED=0 go build -trimpath -ldflags='-s -w -H=windowsgui' -o resources/windows-amd64/voxtnew-helper.exe ./helper-windows
CGO_ENABLED=0 go tool mygo build -platform windows/amd64 "$@"
