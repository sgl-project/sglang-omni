#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT/Voxt.xcodeproj"
SCHEME="Voxt"
CONFIGURATION="Release"
NATIVE_ARCH="$(uname -m)"
DESTINATION="platform=macOS,arch=$NATIVE_ARCH"
DERIVED_DATA="$ROOT/build/LocalPackageDerivedData"
OUTPUT_ROOT="$ROOT/build/local-package"
INSTALL_DIR="/Applications"
INSTALL_APP=false
OPEN_AFTER_INSTALL=false
CLEAN_BUILD=true
LOCAL_OPTIMIZATION=true
UNIVERSAL_BUILD=false
SIGN_APP=true
APP_SIGN_IDENTITY="${DEVELOPER_ID_APP_IDENTITY:-}"
INSTALLER_SIGN_IDENTITY="${DEVELOPER_ID_INSTALLER_IDENTITY:-}"
VERSION=""
BUILD_NUMBER=""
APP_NAME="Voxt"

usage() {
  cat <<'EOF'
Usage: tools/package_local_app.sh [options]

Builds a local Release app and packages it for manual verification. This does
not create a GitHub release and does not publish Sparkle appcast metadata.

Options:
  --version VERSION        Override MARKETING_VERSION for this local build.
  --build-number NUMBER   Override CURRENT_PROJECT_VERSION. Defaults to a timestamp.
  --output-dir PATH       Output directory. Defaults to build/local-package.
  --derived-data PATH     DerivedData directory. Defaults to build/LocalPackageDerivedData.
  --install               Install the built app into /Applications after packaging.
  --install-dir PATH      Install target directory. Defaults to /Applications.
  --open                  Open the app after --install.
  --no-clean              Do not run xcodebuild clean before build.
  --optimized             Use Release optimization. Default disables Swift optimization
                          for local packaging to avoid local optimizer crashes.
  --universal             Build both arm64 and x86_64 via generic macOS destination.
  --app-sign-identity ID  Developer ID Application identity for app signing.
  --pkg-sign-identity ID  Developer ID Installer identity for pkg signing.
  --no-sign               Do not sign the app after building.
  -h, --help              Show this help.

Optional environment:
  DEVELOPER_ID_APP_IDENTITY        Signs the app when set.
  DEVELOPER_ID_INSTALLER_IDENTITY  Signs the .pkg when set.

Examples:
  tools/package_local_app.sh
  tools/package_local_app.sh --install --open
  tools/package_local_app.sh --version 1.12.0 --build-number 101200099 --install
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo
  echo "==> $*"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"
      [[ -n "$VERSION" ]] || die "--version requires a value"
      shift 2
      ;;
    --build-number)
      BUILD_NUMBER="${2:-}"
      [[ -n "$BUILD_NUMBER" ]] || die "--build-number requires a value"
      shift 2
      ;;
    --output-dir)
      OUTPUT_ROOT="${2:-}"
      [[ -n "$OUTPUT_ROOT" ]] || die "--output-dir requires a value"
      shift 2
      ;;
    --derived-data)
      DERIVED_DATA="${2:-}"
      [[ -n "$DERIVED_DATA" ]] || die "--derived-data requires a value"
      shift 2
      ;;
    --install)
      INSTALL_APP=true
      shift
      ;;
    --install-dir)
      INSTALL_DIR="${2:-}"
      [[ -n "$INSTALL_DIR" ]] || die "--install-dir requires a value"
      shift 2
      ;;
    --open)
      OPEN_AFTER_INSTALL=true
      INSTALL_APP=true
      shift
      ;;
    --no-clean)
      CLEAN_BUILD=false
      shift
      ;;
    --optimized)
      LOCAL_OPTIMIZATION=false
      shift
      ;;
    --universal)
      UNIVERSAL_BUILD=true
      DESTINATION="generic/platform=macOS"
      shift
      ;;
    --app-sign-identity)
      APP_SIGN_IDENTITY="${2:-}"
      [[ -n "$APP_SIGN_IDENTITY" ]] || die "--app-sign-identity requires a value"
      shift 2
      ;;
    --pkg-sign-identity)
      INSTALLER_SIGN_IDENTITY="${2:-}"
      [[ -n "$INSTALLER_SIGN_IDENTITY" ]] || die "--pkg-sign-identity requires a value"
      shift 2
      ;;
    --no-sign)
      SIGN_APP=false
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -d "$PROJECT" ]] || die "Xcode project not found: $PROJECT"
command -v xcodebuild >/dev/null || die "xcodebuild is required"
command -v hdiutil >/dev/null || die "hdiutil is required"
command -v productbuild >/dev/null || die "productbuild is required"

if [[ "$OPEN_AFTER_INSTALL" == "true" && "$INSTALL_APP" != "true" ]]; then
  die "--open requires --install"
fi

if [[ -z "$VERSION" ]]; then
  VERSION="$(
    xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration "$CONFIGURATION" -showBuildSettings \
      | awk -F' = ' '/ MARKETING_VERSION = / {print $2; exit}'
  )"
fi
[[ -n "$VERSION" ]] || die "unable to resolve MARKETING_VERSION"

if [[ -z "$BUILD_NUMBER" ]]; then
  BUILD_NUMBER="$(date +%Y%m%d%H%M)"
fi
[[ "$BUILD_NUMBER" =~ ^[0-9]+$ ]] || die "--build-number must be numeric"

if [[ "$SIGN_APP" == "true" && -z "$APP_SIGN_IDENTITY" ]]; then
  APP_SIGN_IDENTITY="$(
    security find-identity -v -p codesigning 2>/dev/null \
      | awk '/Developer ID Application/ {print $2; exit}'
  )"
fi

if [[ -z "$INSTALLER_SIGN_IDENTITY" ]]; then
  INSTALLER_SIGN_IDENTITY="$(
    security find-identity -v -p basic 2>/dev/null \
      | awk -F '"' '/Developer ID Installer/ {print $2; exit}'
  )"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="$OUTPUT_ROOT/$VERSION-$BUILD_NUMBER-$STAMP"
APP_PATH="$DERIVED_DATA/Build/Products/$CONFIGURATION/$APP_NAME.app"
ZIP_PATH="$OUTPUT_DIR/$APP_NAME-$VERSION-macOS.zip"
DMG_PATH="$OUTPUT_DIR/$APP_NAME-$VERSION.dmg"
PKG_PATH="$OUTPUT_DIR/$APP_NAME-$VERSION.pkg"
DMG_STAGING_DIR="$OUTPUT_DIR/dmg-staging"

mkdir -p "$OUTPUT_DIR"

log "Building $APP_NAME $VERSION ($BUILD_NUMBER)"
BUILD_ACTIONS=()
if [[ "$CLEAN_BUILD" == "true" ]]; then
  BUILD_ACTIONS+=(clean)
fi
BUILD_ACTIONS+=(build)

BUILD_OVERRIDES=(
  MARKETING_VERSION="$VERSION"
  CURRENT_PROJECT_VERSION="$BUILD_NUMBER"
  ENABLE_DEBUG_DYLIB=NO
  CODE_SIGNING_ALLOWED=NO
  CODE_SIGNING_REQUIRED=NO
  CODE_SIGN_IDENTITY=
)
if [[ "$UNIVERSAL_BUILD" != "true" ]]; then
  BUILD_OVERRIDES+=(ONLY_ACTIVE_ARCH=YES)
fi
if [[ "$LOCAL_OPTIMIZATION" == "true" ]]; then
  BUILD_OVERRIDES+=(
    SWIFT_OPTIMIZATION_LEVEL=-Onone
    GCC_OPTIMIZATION_LEVEL=0
  )
fi

xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -destination "$DESTINATION" \
  -derivedDataPath "$DERIVED_DATA" \
  "${BUILD_ACTIONS[@]}" \
  "${BUILD_OVERRIDES[@]}"

[[ -d "$APP_PATH" ]] || die "built app not found: $APP_PATH"

log "Validating app bundle"
PLIST_PATH="$APP_PATH/Contents/Info.plist"
SHORT_VERSION="$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$PLIST_PATH" 2>/dev/null || true)"
BUNDLE_VERSION="$(/usr/libexec/PlistBuddy -c "Print :CFBundleVersion" "$PLIST_PATH" 2>/dev/null || true)"
[[ "$SHORT_VERSION" == "$VERSION" ]] || die "CFBundleShortVersionString is $SHORT_VERSION, expected $VERSION"
[[ "$BUNDLE_VERSION" == "$BUILD_NUMBER" ]] || die "CFBundleVersion is $BUNDLE_VERSION, expected $BUILD_NUMBER"

if [[ "$SIGN_APP" == "true" && -n "$APP_SIGN_IDENTITY" ]]; then
  log "Signing app with $APP_SIGN_IDENTITY"

  APP_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "$PLIST_PATH" 2>/dev/null || true)"
  [[ -n "$APP_BUNDLE_ID" ]] || die "unable to resolve CFBundleIdentifier"

  SIGNING_ENTITLEMENTS="$(mktemp)"
  sed "s#\\\$(PRODUCT_BUNDLE_IDENTIFIER)#${APP_BUNDLE_ID}#g" "$ROOT/Voxt/Voxt.entitlements" > "$SIGNING_ENTITLEMENTS"
  if grep -q '\\$(PRODUCT_BUNDLE_IDENTIFIER)' "$SIGNING_ENTITLEMENTS"; then
    rm -f "$SIGNING_ENTITLEMENTS"
    die "failed to expand PRODUCT_BUNDLE_IDENTIFIER in entitlements"
  fi

  sign_if_exists() {
    local target="$1"
    if [[ -e "$target" ]]; then
      echo "Signing: $target"
      codesign --force --options runtime --timestamp --sign "$APP_SIGN_IDENTITY" "$target"
    fi
  }

  SPARKLE_FRAMEWORK="$APP_PATH/Contents/Frameworks/Sparkle.framework"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/XPCServices/Installer.xpc"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/XPCServices/InstallerLauncher.xpc"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/XPCServices/Downloader.xpc"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/Autoupdate"
  sign_if_exists "$SPARKLE_FRAMEWORK/Versions/B/Updater.app"
  sign_if_exists "$SPARKLE_FRAMEWORK"

  while IFS= read -r dylib; do
    sign_if_exists "$dylib"
  done < <(find "$APP_PATH/Contents/Frameworks" -maxdepth 1 -type f -name '*.dylib' 2>/dev/null | sort)

  codesign \
    --force \
    --options runtime \
    --timestamp \
    --entitlements "$SIGNING_ENTITLEMENTS" \
    --sign "$APP_SIGN_IDENTITY" \
    "$APP_PATH"

  rm -f "$SIGNING_ENTITLEMENTS"
elif [[ "$SIGN_APP" == "true" ]]; then
  echo "warning: no Developer ID Application identity found; app will remain unsigned" >&2
fi

if codesign --verify --strict --verbose=2 "$APP_PATH"; then
  echo "codesign verification passed"
else
  echo "warning: codesign verification failed; artifact was still packaged for local inspection" >&2
fi

log "Creating artifacts in $OUTPUT_DIR"
ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"

rm -rf "$DMG_STAGING_DIR"
mkdir -p "$DMG_STAGING_DIR"
ditto "$APP_PATH" "$DMG_STAGING_DIR/$APP_NAME.app"
ln -s /Applications "$DMG_STAGING_DIR/Applications"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$DMG_STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"
rm -rf "$DMG_STAGING_DIR"

PKG_ARGS=(--component "$APP_PATH" /Applications)
if [[ -n "$INSTALLER_SIGN_IDENTITY" ]]; then
  PKG_ARGS+=(--sign "$INSTALLER_SIGN_IDENTITY")
fi
PKG_ARGS+=("$PKG_PATH")
productbuild "${PKG_ARGS[@]}"

shasum -a 256 "$ZIP_PATH" > "$ZIP_PATH.sha256"
shasum -a 256 "$PKG_PATH" > "$PKG_PATH.sha256"

if [[ "$INSTALL_APP" == "true" ]]; then
  log "Installing $APP_NAME.app into $INSTALL_DIR"
  mkdir -p "$INSTALL_DIR"
  osascript -e "tell application \"$APP_NAME\" to quit" >/dev/null 2>&1 || true
  rm -rf "$INSTALL_DIR/$APP_NAME.app"
  ditto "$APP_PATH" "$INSTALL_DIR/$APP_NAME.app"
  xattr -dr com.apple.quarantine "$INSTALL_DIR/$APP_NAME.app" >/dev/null 2>&1 || true

  if [[ "$OPEN_AFTER_INSTALL" == "true" ]]; then
    open "$INSTALL_DIR/$APP_NAME.app"
  fi
fi

cat <<EOF

Done.
App: $APP_PATH
Zip: $ZIP_PATH
Dmg: $DMG_PATH
Pkg: $PKG_PATH
EOF
