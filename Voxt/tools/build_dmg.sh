#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <Voxt.app path> <output.dmg path>" >&2
  exit 64
fi

APP_PATH="$1"
OUTPUT_PATH="$2"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKGROUND_PATH="$ROOT_DIR/packaging/dmg/background.tiff"
SETTINGS_PATH="$ROOT_DIR/packaging/dmg/settings.py"
CONSTRAINTS_PATH="$ROOT_DIR/packaging/dmg/constraints.txt"
VOLUME_ICON_PATH="$APP_PATH/Contents/Resources/VoxtIcon.icns"

DMGBUILD_VERSION="1.6.7"

if [ ! -d "$APP_PATH" ]; then
  echo "Voxt.app not found at $APP_PATH" >&2
  exit 1
fi

if [ ! -f "$BACKGROUND_PATH" ]; then
  echo "DMG background not found at $BACKGROUND_PATH" >&2
  exit 1
fi

if [ ! -f "$VOLUME_ICON_PATH" ]; then
  echo "Built app icon not found at $VOLUME_ICON_PATH" >&2
  exit 1
fi

if [ ! -f "$SETTINGS_PATH" ]; then
  echo "DMG settings not found at $SETTINGS_PATH" >&2
  exit 1
fi

if [ ! -f "$CONSTRAINTS_PATH" ]; then
  echo "DMG build constraints not found at $CONSTRAINTS_PATH" >&2
  exit 1
fi

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx is required to run dmgbuild" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
rm -f "$OUTPUT_PATH"

UV_NO_PROGRESS=1 uvx \
  --from "dmgbuild==${DMGBUILD_VERSION}" \
  --constraints "$CONSTRAINTS_PATH" \
  dmgbuild \
  --settings "$SETTINGS_PATH" \
  -D "app=$APP_PATH" \
  -D "background=$BACKGROUND_PATH" \
  -D "volume_icon=$VOLUME_ICON_PATH" \
  --detach-retries 5 \
  "Voxt" \
  "$OUTPUT_PATH"

hdiutil verify "$OUTPUT_PATH"
