#!/usr/bin/env bash
# Builds voxtcore's C ABI (ios/voxtcore/cabi) as a static library for one
# SDK, into ios/build/<sdk>/libvoxtcore.a (gitignored).
#
#   scripts/build-core.sh macos             this Mac, for `swift test` in VoxtKit (Command Line Tools are enough)
#   scripts/build-core.sh iphoneos          an iPhone (needs Xcode)
#   scripts/build-core.sh iphonesimulator   the simulator on Apple silicon (needs Xcode)
#   scripts/build-core.sh ios               both iOS SDKs
#   scripts/build-core.sh xcode             the SDK Xcode builds for ($PLATFORM_NAME): the app's pre-build step
#
# VoxtKit links the macos library in `swift test`; the app's project
# (App/project.yml) links the iOS ones by SDK. Run from anywhere.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)" # Voxt/ios
module="$(cd "$here/.." && pwd)"         # Voxt
out="$here/build"
min_ios=17.0
min_macos=14.0 # VoxtKit's platforms (Package.swift)
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"
# Xcode's build phases run with a short PATH.
export PATH="$PATH:/usr/local/go/bin:/opt/homebrew/bin:$HOME/go/bin"

archive() { # archive <sdk> <GOOS> [CC]
	local dir="$out/$1"
	mkdir -p "$dir"
	# Xcode exports SDKROOT and deployment targets for its own SDK: the
	# compiler gets its SDK from CC (or the Mac's) instead.
	(cd "$module" && env -u SDKROOT -u IPHONEOS_DEPLOYMENT_TARGET CGO_ENABLED=1 GOOS="$2" GOARCH=arm64 ${3:+"CC=$3"} \
		go build -trimpath -buildmode=c-archive -o "$dir/libvoxtcore.a" ./ios/voxtcore/cabi)
	rm -f "$dir/libvoxtcore.h" # the ABI's header is the hand-written one in VoxtKit
	echo "built $dir/libvoxtcore.a"
}

ios() { # ios <sdk>
	xcrun --sdk "$1" --show-sdk-path >/dev/null 2>&1 ||
		{ echo "the $1 SDK needs Xcode (xcode-select -s /Applications/Xcode.app)" >&2; exit 1; }
	local target="arm64-apple-ios$min_ios"
	[ "$1" = iphonesimulator ] && target="$target-simulator"
	archive "$1" ios "$(xcrun --sdk "$1" --find clang) -isysroot $(xcrun --sdk "$1" --show-sdk-path) -target $target"
}

case "${1:-macos}" in
macos)
	# In the flags, not MACOSX_DEPLOYMENT_TARGET: Go's build cache keys on them.
	export CGO_CFLAGS="${CGO_CFLAGS:--O2 -g} -mmacosx-version-min=$min_macos" CGO_LDFLAGS="${CGO_LDFLAGS:-} -mmacosx-version-min=$min_macos"
	archive macos darwin
	;;
iphoneos | iphonesimulator) ios "$1" ;;
ios) ios iphoneos && ios iphonesimulator ;;
xcode) ios "${PLATFORM_NAME:?Xcode sets PLATFORM_NAME}" ;;
*)
	echo "usage: $0 [macos|iphoneos|iphonesimulator|ios|xcode]" >&2
	exit 2
	;;
esac
