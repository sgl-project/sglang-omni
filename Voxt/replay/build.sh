#!/bin/bash
# note (luojiaxuan): Builds build/replay-driver and build/ReplaySink.app (executable: build/ReplaySink.app/Contents/MacOS/replay-sink).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

flags=(-O -swift-version 5)

# note (luojiaxuan): Some Command Line Tools installs keep a stale usr/include/swift/module.modulemap next to
# bridging.modulemap; both define SwiftBridging and every import fails. A VFS overlay hides
# the stale file from this build without touching the toolchain.
include_dir="$(cd "$(dirname "$(xcrun --find swiftc)")/../include/swift" 2>/dev/null && pwd || true)"
if [[ -n "$include_dir" && -f "$include_dir/module.modulemap" && -f "$include_dir/bridging.modulemap" ]]; then
  : > build/empty.modulemap
  cat > build/vfsoverlay.yaml <<EOF
{"version": 0, "case-sensitive": "false", "roots": [{"type": "directory", "name": "$include_dir",
  "contents": [{"type": "file", "name": "module.modulemap", "external-contents": "$PWD/build/empty.modulemap"}]}]}
EOF
  flags+=(-vfsoverlay build/vfsoverlay.yaml)
fi

swiftc "${flags[@]}" Sources/Common/*.swift Sources/Driver/*.swift -o build/replay-driver

app=build/ReplaySink.app/Contents
mkdir -p "$app/MacOS"
swiftc "${flags[@]}" Sources/Common/*.swift Sources/Sink/*.swift -o "$app/MacOS/replay-sink"
cat > "$app/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>local.voxt.replay-sink</string>
  <key>CFBundleName</key><string>Replay Sink</string>
  <key>CFBundleExecutable</key><string>replay-sink</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF
echo "built build/replay-driver and build/ReplaySink.app"
