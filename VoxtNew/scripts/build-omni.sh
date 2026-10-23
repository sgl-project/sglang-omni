#!/usr/bin/env bash
# Builds the Omni native runtime (sglang_omni_mlx/native, C++ on MLX) on a
# Mac and stages what VoxtNew runs of it where mygo ships it:
# resources/darwin-arm64/omni/bin/ becomes
# VoxtNew.app/Contents/Resources/omni/bin/ (go tool mygo build, mygo dev):
# qwen3_asr_server (Qwen3-ASR and, with --model-kind silero_vad, Silero
# VAD), MLX's libraries and mlx.metallib, which MLX loads from beside its
# library. mygo build signs the binaries and libraries with the app (they
# keep the hardened runtime of a real identity); the metallib is a
# resource the app's signature seals.
#
#   scripts/build-omni.sh               # build and stage it for the app
#   scripts/build-omni.sh --out DIR     # stage it in DIR instead: a runtime
#                                       # outside the app, for VOXTNEW_OMNI_RUNTIME
#   scripts/build-omni.sh --clean       # delete the build tree afterwards (~1 GB)
#   scripts/mac.sh 'scripts/build-omni.sh && go tool mygo build'
#
# It reuses sglang_omni_mlx/native/scripts/build_runtime.sh (uv, the pinned
# MLX wheel, CMake, Ninja). The sources are the repository's
# ($ROOT/../sglang_omni_mlx/native), or the copy scripts/mac.sh syncs
# beside the project (.omni-native), or $OMNI_NATIVE. Without uv on PATH a
# private one is installed into the build tree (python3 -m venv).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != arm64 ]]; then
  echo "the Omni runtime builds on an Apple silicon Mac: scripts/mac.sh scripts/build-omni.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/resources/darwin-arm64/omni"
CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) [[ -n "${2:-}" ]] || { echo "--out needs a directory" >&2; exit 1; }
      OUT="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
done

NATIVE="${OMNI_NATIVE:-}"
if [[ -z "$NATIVE" ]]; then
  for d in "$ROOT/../sglang_omni_mlx/native" "$ROOT/.omni-native"; do
    if [[ -f "$d/scripts/build_runtime.sh" ]]; then NATIVE="$d"; break; fi
  done
fi
[[ -n "$NATIVE" && -f "$NATIVE/scripts/build_runtime.sh" ]] || {
  echo "no sglang_omni_mlx/native sources: set OMNI_NATIVE" >&2; exit 1; }
NATIVE="$(cd "$NATIVE" && pwd)"

WORK="$ROOT/build/omni"
mkdir -p "$WORK"
if ! command -v uv >/dev/null; then
  if [[ ! -x "$WORK/tools/bin/uv" ]]; then
    python3 -m venv "$WORK/tools"
    "$WORK/tools/bin/pip" install -q uv
  fi
  export PATH="$WORK/tools/bin:$PATH"
fi

log="$WORK/build.log"
echo "building the Omni runtime from $NATIVE (log: $log)"
if ! bash "$NATIVE/scripts/build_runtime.sh" "$WORK/runtime" "$WORK/work" >"$log" 2>&1; then
  tail -40 "$log" >&2
  echo "build_runtime.sh failed" >&2
  exit 1
fi

# Only what VoxtNew runs: the one server, and MLX beside it.
rm -rf "$OUT/bin"
mkdir -p "$OUT/bin"
cp -f "$WORK/runtime/bin/qwen3_asr_server" "$OUT/bin/"
cp -f "$WORK/runtime/bin/"*.dylib "$WORK/runtime/bin/mlx.metallib" "$OUT/bin/"
chmod 0755 "$OUT/bin/qwen3_asr_server"
# Unsigned code does not run on Apple silicon: sign it ad hoc for a runtime
# used outside the app (mygo build signs the app's copy again).
for f in "$OUT/bin/qwen3_asr_server" "$OUT/bin/"*.dylib; do
  codesign --force --sign - "$f" >/dev/null 2>&1
done
# It loads (its libraries resolve) when it gets as far as refusing a kind.
probe="$("$OUT/bin/qwen3_asr_server" --model-kind none 2>&1 || true)"
if [[ "$probe" != *"--model-kind must be"* ]]; then
  echo "the staged qwen3_asr_server does not start: $probe" >&2
  exit 1
fi
du -sh "$OUT/bin" | sed "s|^|staged |"
echo "VOXTNEW_OMNI_RUNTIME=$OUT"

if [[ $CLEAN == 1 ]]; then
  rm -rf "$WORK"
  echo "deleted $WORK"
fi
