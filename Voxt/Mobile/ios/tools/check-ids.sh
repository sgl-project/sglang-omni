#!/bin/bash
# Checks that one prefix drives every ID: app and keyboard bundle IDs and the shared App Group.
# Reads build settings only (no build, no signing). Usage: ios/tools/check-ids.sh [prefix]
set -uo pipefail
cd "$(dirname "$0")/.."
prefix=${1:-com.example.idcheck}
fail=0
setting() { xcodebuild -project VoxtMobile.xcodeproj -target "$1" -configuration Release -showBuildSettings \
  VOXT_MOBILE_ID_PREFIX="$prefix" 2>/dev/null | sed -n "s/^ *$2 = //p" | head -1; }
check() { if [ "$2" = "$3" ]; then echo "ok   $1 = $2"; else echo "FAIL $1: expected [$3], got [$2]"; fail=1; fi; }
app=$(setting VoxtMobile PRODUCT_BUNDLE_IDENTIFIER)
check "app bundle ID" "$app" "$prefix"
check "keyboard bundle ID" "$(setting VoxtKeyboard PRODUCT_BUNDLE_IDENTIFIER)" "$prefix.keyboard"
# OnboardingView finds the Voxt keyboard as the app's bundle ID + ".keyboard".
check "keyboard = app + .keyboard" "$(setting VoxtKeyboard PRODUCT_BUNDLE_IDENTIFIER)" "$app.keyboard"
check "unit tests bundle ID" "$(setting VoxtMobileTests PRODUCT_BUNDLE_IDENTIFIER)" "$prefix.tests"
check "UI tests bundle ID" "$(setting VoxtMobileUITests PRODUCT_BUNDLE_IDENTIFIER)" "$prefix.uitests"
check "URL scheme name is the app ID" "$(grep -A1 CFBundleURLName App/Info.plist | grep -c '\$(PRODUCT_BUNDLE_IDENTIFIER)')" 1
# Neutral defaults; each signer sets their own prefix in Signing.local.xcconfig.
check "default prefix" "$(sed -n 's/^VOXT_MOBILE_ID_PREFIX = //p' Config/Signing.shared.xcconfig)" com.voxt.VoxtMobile
check "default team is empty" "$(grep -c '^DEVELOPMENT_TEAM =$' Config/Signing.shared.xcconfig)" 1
check "Keychain names follow the prefix" "$(sed -n 's/^VOXT_KEYCHAIN_SERVICE_PREFIX = //p' Config/Signing.shared.xcconfig)" '$(VOXT_MOBILE_ID_PREFIX)'
check "App/Info.plist publishes VoxtKeychainServicePrefix" "$(grep -A1 VoxtKeychainServicePrefix App/Info.plist | grep -c '\$(VOXT_KEYCHAIN_SERVICE_PREFIX)')" 1
check "default source link is upstream" "$(sed -n 's/^VOXT_SOURCE_URL = //p' Config/Signing.shared.xcconfig)" github.com/sgl-project/sglang-omni
check "App/Info.plist publishes VoxtSourceURL" "$(grep -A1 VoxtSourceURL App/Info.plist | grep -c '\$(VOXT_SOURCE_URL)')" 1
check "app App Group" "$(setting VoxtMobile VOXT_APP_GROUP)" "group.$prefix.shared"
check "keyboard App Group" "$(setting VoxtKeyboard VOXT_APP_GROUP)" "group.$prefix.shared"
for f in App/VoxtMobile.entitlements Keyboard/VoxtKeyboard.entitlements; do
  check "$f uses the setting" "$(grep -c '\$(VOXT_APP_GROUP)' "$f")" 1
done
for f in App/Info.plist Keyboard/Info.plist; do
  check "$f publishes VoxtAppGroup" "$(grep -A1 VoxtAppGroup "$f" | grep -c '\$(VOXT_APP_GROUP)')" 1
done
# Version: one place (project.yml MARKETING_VERSION / CURRENT_PROJECT_VERSION), read by both plists.
for f in App/Info.plist Keyboard/Info.plist; do
  check "$f version from MARKETING_VERSION" "$(grep -A1 CFBundleShortVersionString "$f" | grep -c '\$(MARKETING_VERSION)')" 1
  check "$f build from CURRENT_PROJECT_VERSION" "$(grep -A1 '<key>CFBundleVersion</key>' "$f" | grep -c '\$(CURRENT_PROJECT_VERSION)')" 1
done
check "iPhone version" "$(setting VoxtMobile MARKETING_VERSION)" "${VOXT_EXPECTED_VERSION:-0.1.2}"
check "Connector version" "$(cd ../connector/VoxtConnector && xcodebuild -project VoxtConnector.xcodeproj -target VoxtConnector -configuration Release -showBuildSettings 2>/dev/null | sed -n 's/^ *MARKETING_VERSION = //p' | head -1)" "${VOXT_EXPECTED_VERSION:-0.1.2}"
exit $fail
