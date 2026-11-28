#!/usr/bin/env bash
# Builds the Go core for this Mac and runs VoxtKit's Swift tests against it.
# Works with Xcode or with the Command Line Tools only. Without Xcode the
# script adds the tools' Swift Testing (outside the default search path)
# and builds against their macOS 26 SDK: the newer SDK's SwiftUI needs
# macro plugins that only Xcode ships.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)" # Voxt/ios
"$here/scripts/build-core.sh" macos
cd "$here/VoxtKit"
# SwiftPM does not track libvoxtcore.a: drop the linked tests so that they
# link the library just built.
rm -rf .build/debug/VoxtKitPackageTests.xctest
flags=()
clt=/Library/Developer/CommandLineTools
if ! xcodebuild -version >/dev/null 2>&1; then
	frameworks="$clt/Library/Developer/Frameworks"
	[ -d "$frameworks/Testing.framework" ] &&
		flags=(-Xswiftc -F"$frameworks" -Xlinker -F"$frameworks" -Xlinker -rpath -Xlinker "$frameworks")
	[ -d "$clt/SDKs/MacOSX26.sdk" ] && export SDKROOT="$clt/SDKs/MacOSX26.sdk"
fi
swift test --build-system native "${flags[@]}" "$@"
