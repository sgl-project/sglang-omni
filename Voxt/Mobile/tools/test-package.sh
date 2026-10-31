#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
export TMPDIR="$root/build/connector-tmp/"
export CLANG_MODULE_CACHE_PATH="$root/build/connector-module-cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$root/build/connector-module-cache"
mkdir -p "$TMPDIR" "$CLANG_MODULE_CACHE_PATH" "$root/build/connector-spm-cache"
exec nice -n 10 swift test --package-path "$root/connector/VoxtConnectionKit" --scratch-path "$root/build/connector-package" --cache-path "$root/build/connector-spm-cache" --disable-sandbox --jobs 4 "$@"
