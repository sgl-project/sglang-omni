#!/usr/bin/env bash
# Builds voxtnew-llm (llm/, Swift: mlx-swift-lm and llama.swift, release,
# arm64) on a Mac and stages it where mygo ships it:
# resources/darwin-arm64/llm/ becomes VoxtNew.app/Contents/Resources/llm/
# (go tool mygo build, mygo dev), with what it needs beside it:
#
#   voxtnew-llm                     the executable (llm protocol 1 on stdio)
#   mlx.metallib                    MLX's Metal kernels (MLX looks beside the binary)
#   llama.framework                 llama.cpp (the binary's rpath is @executable_path)
#   swift-transformers_Hub.bundle   the tokenizers' fallback configurations
#
#   scripts/build-llm.sh              # stage it for the app
#   scripts/build-llm.sh --out DIR    # build it into DIR instead (VOXTNEW_LLM=DIR/voxtnew-llm)
#   scripts/build-llm.sh --test       # the Kit's tests only (fixtures, server, text)
#   scripts/mac.sh 'scripts/build-helper.sh && scripts/build-llm.sh && go tool mygo build'
#
# SwiftPM on the command line cannot compile MLX's Metal shaders, so the
# build is xcodebuild's (into llm/.build/xcode, its packages in
# llm/.build/xcode-packages: about 2 GB together; the first build takes a
# few minutes). The tests build with SwiftPM, the Kit alone (llm/.build).
#
# Everything is signed, the binary with llm/Support/llm.entitlements, ad
# hoc unless VOXTNEW_SIGN_IDENTITY names an identity (README.md, "Signing
# during development"), and then with the hardened runtime; mygo build
# keeps an ad hoc signature, and signs the app's nested code again with a
# real identity (keeping the entitlements).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "voxtnew-llm builds on macOS: scripts/mac.sh scripts/build-llm.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/resources/darwin-arm64/llm"
case "${1:-}" in
  --out)
    [[ -n "${2:-}" ]] || { echo "--out needs a directory" >&2; exit 1; }
    OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
    ;;
  --test)
    cd "$ROOT/llm"
    swift build --product VoxtNewLLMPackageTests
    exec swift test --skip-build
    ;;
esac
IDENTITY="${VOXTNEW_SIGN_IDENTITY:--}"

# The process reports the app's version (llm.hello): keep them in step.
app_version="$(sed -n 's/^  "version": *"\([^"]*\)".*/\1/p' "$ROOT/mygo.json")"
llm_version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$ROOT/llm/Support/Info.plist")"
if [[ "$app_version" != "$llm_version" ]]; then
  echo "llm/Support/Info.plist says $llm_version, mygo.json $app_version: make them agree" >&2
  exit 1
fi

cd "$ROOT/llm"
PRODUCTS=.build/xcode/Build/Products/Release
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
# Linked again each time: Support/Info.plist goes in at link time.
rm -f "$PRODUCTS/voxtnew-llm"
if ! xcodebuild build -scheme VoxtNewLLM -configuration Release -destination "platform=macOS,arch=arm64" \
    -derivedDataPath .build/xcode -clonedSourcePackagesDirPath .build/xcode-packages \
    -skipPackagePluginValidation -skipMacroValidation \
    CLANG_ENABLE_CODE_COVERAGE=NO ENABLE_CODE_COVERAGE=NO >"$log" 2>&1; then
  grep -E "error:" "$log" | sort -u | head -40 >&2 || tail -40 "$log" >&2
  echo "xcodebuild failed" >&2
  exit 1
fi
grep -E "^/.*/llm/Sources/.*warning:" "$log" | sort -u || true
embedded="$(grep -a -A1 '<key>CFBundleShortVersionString</key>' "$PRODUCTS/voxtnew-llm" | sed -n 's/.*<string>\(.*\)<\/string>.*/\1/p' | head -1)"
if [[ "$embedded" != "$llm_version" ]]; then
  echo "voxtnew-llm was linked with version ${embedded:-none}, not $llm_version" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cp -f "$PRODUCTS/voxtnew-llm" "$OUT/voxtnew-llm"
cp -f "$PRODUCTS/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" "$OUT/mlx.metallib"
ditto "$PRODUCTS/llama.framework" "$OUT/llama.framework"
ditto "$PRODUCTS/swift-transformers_Hub.bundle" "$OUT/swift-transformers_Hub.bundle"
# The hardened runtime with a real identity only: its library validation
# refuses an ad hoc llama.framework (no Team ID to match).
runtime=()
[[ "$IDENTITY" != "-" ]] && runtime=(--options runtime --timestamp)
codesign --force --sign "$IDENTITY" ${runtime[@]+"${runtime[@]}"} "$OUT/llama.framework"
codesign --force --sign "$IDENTITY" --entitlements Support/llm.entitlements ${runtime[@]+"${runtime[@]}"} "$OUT/voxtnew-llm"

# Xcode's package schemes gather coverage: a release build must not count
# every branch (it slowed generation, and wrote default.profraw files).
if nm "$OUT/voxtnew-llm" | grep -q __llvm_profile; then
  echo "voxtnew-llm was built with coverage instrumentation" >&2
  exit 1
fi

# It must start and answer the handshake from where it is staged.
hello="$(printf '%s\n' '{"id":"1","method":"llm.hello","params":{"protocol":1,"app":"build-llm"}}' | "$OUT/voxtnew-llm" 2>/dev/null | grep '"id":"1"' || true)"
if [[ "$hello" != *'"protocol":1'* ]]; then
  echo "the staged voxtnew-llm did not answer llm.hello: $hello" >&2
  exit 1
fi
echo "built $OUT ($llm_version, signed ${IDENTITY/#-/ad hoc}, $(du -sh "$OUT" | cut -f1))"
