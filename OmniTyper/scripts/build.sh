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
bash "$APP_ROOT/scripts/sign.sh" "$APP_BUNDLE"
echo "$APP_BUNDLE"
