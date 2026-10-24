#!/bin/bash
# Builds the app and keyboard for the simulator without ios/InputMethods.yml, the way a build that
# leaves out the Japanese, pinyin and Zhuyin packages does, and checks that it lists none of them.
# Needs Xcode and xcodegen; no signing. Usage: ios/tools/test-input-methods-optional.sh [output dir]
set -uo pipefail
cd "$(dirname "$0")/.."
out=${1:-$PWD/build/without-input-methods}
mkdir -p "$out"
# The spec and its project stay in ios/ so their relative paths resolve as project.yml's do; the
# project gets its own name so the committed VoxtMobile.xcodeproj is untouched.
spec=.project-without-input-methods.yml
name=VoxtMobileWithoutInputMethods
trap 'rm -rf "$spec" "$name.xcodeproj"' EXIT
grep -v -e '^include:' -e '^  - InputMethods.yml' project.yml | sed "s/^name: VoxtMobile$/name: $name/" > "$spec"
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: expected [$3], got [$2]"; fail=1; fi; }
xcodegen generate --quiet --spec "$spec" || { echo "FAIL xcodegen"; exit 1; }
check "no input method package in the project" "$(grep -cE 'productName = (JapaneseInput|PinyinInput|ZhuyinInput);' "$name.xcodeproj/project.pbxproj")" 0
xcodebuild build -project "$name.xcodeproj" -scheme VoxtMobile -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath "$out/DerivedData" -packageCachePath "${VOXT_PACKAGE_CACHE:-$out/swiftpm-cache}" \
  CODE_SIGNING_ALLOWED=NO ARCHS=arm64 > "$out/build.log" 2>&1  # the Tailcat library is built for arm64 only
check "builds without the input method packages" "$?" 0
grep -E "error:" "$out/build.log" | sort -u | head -5
app="$out/DerivedData/Build/Products/Debug-iphonesimulator/VoxtMobile.app"
for plist in "$app/Info.plist" "$app/PlugIns/VoxtKeyboard.appex/Info.plist"; do
  [ -f "$plist" ] || { check "$plist exists" no yes; continue; }
  check "${plist#"$app"/} lists no optional layout" "$(/usr/libexec/PlistBuddy -c 'Print :VoxtInputMethods' "$plist" 2>&1)" ""
done
exit $fail
