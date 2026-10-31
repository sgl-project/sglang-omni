#!/bin/bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
native="$root/tailcat/native"
out="$root/build/voxt-tailcat-native/carchive"
artifact="$root/ios/VoxtTailcat/Artifacts/CVoxtTailcat.xcframework"
source "$root/tools/tailcat-go-env.sh"
clang="$(xcrun --find clang)"
sdk_mac="$(xcrun --sdk macosx --show-sdk-path)"
sdk_ios="$(xcrun --sdk iphoneos --show-sdk-path)"
sdk_sim="$(xcrun --sdk iphonesimulator --show-sdk-path)"

unset CC CGO_CFLAGS CGO_LDFLAGS CGO_CXXFLAGS CGO_CPPFLAGS SDKROOT
tailcat_prepare_go "$native"
export CGO_ENABLED=1
export GOARCH=arm64
export CC="$clang"

cd "$native"
module_line="$("$GO_BIN" list -mod=readonly -m github.com/tailscale/tailcat)"
if [[ "$module_line" != "github.com/tailscale/tailcat v0.6.0" ]]; then
  echo "tailcat module is not the pinned v0.6.0" >&2
  exit 1
fi
if [[ "${VOXT_TAILCAT_PREPARE_ONLY:-}" == 1 ]]; then
  echo "toolchain ready"
  exit 0
fi

build_slice() {
  local goos="$1"
  local dest="$2"
  local flags="$3"
  mkdir -p "$dest"
  export GOOS="$goos"
  export CGO_CFLAGS="$flags"
  export CGO_LDFLAGS="$flags"
  "$GO_BIN" build -trimpath -mod=readonly -tags ts_omit_ssh -buildmode=c-archive -o "$dest/libCVoxtTailcat.a" .
}

build_slice darwin "$out/darwin" "-mmacosx-version-min=14.0 -isysroot $sdk_mac"
build_slice ios "$out/ios" "-target arm64-apple-ios17.0 -miphoneos-version-min=17.0 -isysroot $sdk_ios"
build_slice ios "$out/iossim" "-target arm64-apple-ios17.0-simulator -mios-simulator-version-min=17.0 -isysroot $sdk_sim"
unset GOOS CGO_CFLAGS CGO_LDFLAGS

cmp -s "$out/darwin/libCVoxtTailcat.h" "$out/ios/libCVoxtTailcat.h"
cmp -s "$out/darwin/libCVoxtTailcat.h" "$out/iossim/libCVoxtTailcat.h"
mkdir -p "$out/headers"
cp "$out/darwin/libCVoxtTailcat.h" "$out/headers/VoxtTailcat.h"
cp "$native/include/module.modulemap" "$out/headers/module.modulemap"

python3 - "$out" <<'PY'
import re
import subprocess
import sys

out = sys.argv[1]
slices = [
    ("darwin", "MACOS", (14, 0)),
    ("ios", "IOS", (17, 0)),
    ("iossim", "IOSSIMULATOR", (17, 0)),
]
names = {
    "1": "MACOS",
    "2": "IOS",
    "7": "IOSSIMULATOR",
    "MACOS": "MACOS",
    "IOS": "IOS",
    "IOSSIMULATOR": "IOSSIMULATOR",
}

def version(text):
    parts = tuple(int(part) for part in text.split("."))
    if not parts:
        raise ValueError(text)
    return parts

def higher(left, right):
    width = max(len(left), len(right))
    left += (0,) * (width - len(left))
    right += (0,) * (width - len(right))
    return left > right

for name, expect, limit in slices:
    archive = f"{out}/{name}/libCVoxtTailcat.a"
    text = subprocess.check_output(["otool", "-l", archive], text=True, errors="replace")
    current = None
    platform = None
    found = {}
    for line in text.splitlines():
        member = re.search(r"\(([^)]+\.o)\):\s*$", line)
        if member:
            current = member.group(1)
            found.setdefault(current, [])
            platform = None
            continue
        stripped = line.strip()
        if stripped.startswith("platform "):
            platform = names.get(stripped.split()[-1])
            continue
        if stripped.startswith("minos ") and current:
            if platform is None:
                raise SystemExit(f"{name} {current} has minos without a platform")
            found[current].append((platform, version(stripped.split()[-1])))
            platform = None
    if not found:
        raise SystemExit(f"{name} archive has no members")
    for member, versions in sorted(found.items()):
        if not versions:
            raise SystemExit(f"{name} {member} has no minimum OS")
        for platform, minos in versions:
            if platform != expect or higher(minos, limit):
                rendered = ".".join(str(part) for part in minos)
                raise SystemExit(f"{name} {member} is {platform} {rendered}")
    go_versions = found.get("go.o")
    if not go_versions:
        raise SystemExit(f"{name} archive has no go.o")
    rendered = ".".join(str(part) for part in go_versions[0][1])
    print(f"{name} go.o {go_versions[0][0]} {rendered} members {len(found)}")
PY

cat > "$out/linkcheck.c" <<'EOF'
#include "VoxtTailcat.h"

int main(void) {
    if (VoxtTailcatStop(0) != 0) {
        return 1;
    }
    VoxtTailcatFree(0);
    return 0;
}
EOF

"$clang" -mmacosx-version-min=14.0 -isysroot "$sdk_mac" -I "$out/headers" \
  -framework CoreFoundation -framework Security \
  "$out/linkcheck.c" "$out/darwin/libCVoxtTailcat.a" -o "$out/linkcheck_macos"
"$out/linkcheck_macos" > "$out/link-macos.out" 2> "$out/link-macos.err"
if [[ -s "$out/link-macos.out" || -s "$out/link-macos.err" ]]; then
  echo "macOS link check produced output" >&2
  exit 1
fi

"$clang" -target arm64-apple-ios17.0 -miphoneos-version-min=17.0 -isysroot "$sdk_ios" \
  -I "$out/headers" -framework CoreFoundation -framework Security \
  "$out/linkcheck.c" "$out/ios/libCVoxtTailcat.a" -o "$out/linkcheck_ios"
"$clang" -target arm64-apple-ios17.0-simulator -mios-simulator-version-min=17.0 -isysroot "$sdk_sim" \
  -I "$out/headers" -framework CoreFoundation -framework Security \
  "$out/linkcheck.c" "$out/iossim/libCVoxtTailcat.a" -o "$out/linkcheck_iossim"

rm -rf "$artifact"
xcodebuild -create-xcframework \
  -library "$out/ios/libCVoxtTailcat.a" -headers "$out/headers" \
  -library "$out/iossim/libCVoxtTailcat.a" -headers "$out/headers" \
  -library "$out/darwin/libCVoxtTailcat.a" -headers "$out/headers" \
  -output "$artifact" > "$out/xcframework.log"
echo "xcframework ready"
