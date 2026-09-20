#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGURATION="${CONFIGURATION:-release}"
PYTHON_BIN="${OMNITYPER_PYTHON:-$APP_ROOT/.venv/bin/python}"
swift build --package-path "$APP_ROOT" -c "$CONFIGURATION"
BIN_DIR="$(swift build --package-path "$APP_ROOT" -c "$CONFIGURATION" --show-bin-path)"
APP_BUNDLE="$APP_ROOT/dist/OmniTyper.app"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources/backend"
cp "$BIN_DIR/OmniTyper" "$APP_BUNDLE/Contents/MacOS/"
cp "$APP_ROOT/Resources/Info.plist" "$APP_BUNDLE/Contents/Info.plist"
cp "$APP_ROOT/backend/worker.py" "$APP_ROOT/backend/server.py" "$APP_ROOT/backend/text_api.py" "$APP_BUNDLE/Contents/Resources/backend/"
rm -rf "$APP_BUNDLE/Contents/Resources/backend/__pycache__"
cp "$APP_ROOT/../LICENSE" "$APP_BUNDLE/Contents/Resources/LICENSE"
for LPROJ in "$APP_ROOT"/Sources/OmniTyper/Resources/*.lproj; do
  # Note (Jiaxin Deng): Replace rather than merge, so a rebuild cannot nest directories or
  # keep a locale that was removed.
  rm -rf "$APP_BUNDLE/Contents/Resources/$(basename "$LPROJ")"
  cp -R "$LPROJ" "$APP_BUNDLE/Contents/Resources/"
done
/usr/libexec/PlistBuddy -c "Add :OmniTyperPython string $PYTHON_BIN" "$APP_BUNDLE/Contents/Info.plist"
ICONSET="$APP_ROOT/.build/AppIcon.iconset"
mkdir -p "$ICONSET"
swift "$APP_ROOT/scripts/icon.swift" "$ICONSET"
iconutil -c icns "$ICONSET" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
if [[ "${CODE_SIGN_IDENTITY:--}" == "-" ]]; then
  echo "note: signing ad-hoc. macOS ties the Accessibility grant to this build, so" >&2
  echo "      updating the app requires granting it again. Set CODE_SIGN_IDENTITY to" >&2
  echo "      a stable signing identity to keep the grant across updates." >&2
fi
codesign --force --sign "${CODE_SIGN_IDENTITY:--}" --options runtime \
  --entitlements "$APP_ROOT/Resources/Entitlements.plist" "$APP_BUNDLE"
codesign --verify --strict "$APP_BUNDLE"
echo "$APP_BUNDLE"
