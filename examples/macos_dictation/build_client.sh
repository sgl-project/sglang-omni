#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
client_dir="$project_dir/.build/client"
app_dir="$client_dir/OmniDictation.app"
mkdir -p "$app_dir/Contents/MacOS" "$client_dir/module-cache"

build_args=(--package-path "$project_dir" --scratch-path "$project_dir/.build/package"
  --cache-path "$project_dir/.build/package-cache" --configuration release --arch arm64)
CLANG_MODULE_CACHE_PATH="$client_dir/module-cache" xcrun swift build "${build_args[@]}" --product OmniDictation
binary_dir="$(xcrun swift build "${build_args[@]}" --show-bin-path)"
cp "$binary_dir/OmniDictation" "$app_dir/Contents/MacOS/OmniDictation"

cat > "$app_dir/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>OmniDictation</string>
<key>CFBundleIdentifier</key><string>local.omni.dictation</string>
<key>CFBundleName</key><string>Omni 听写</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.5.3</string>
<key>CFBundleVersion</key><string>19</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>LSUIElement</key><true/>
<key>NSMicrophoneUsageDescription</key><string>录下你主动开始的语音，并发送给本机 Omni ASR 转写。录音不保存到磁盘。</string>
<key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
PLIST

/usr/bin/codesign --force --sign - "$app_dir"
printf '真实客户端构建完成：%s\n' "$app_dir"
printf '请先退出已运行的旧版 Omni 听写，再运行：open "%s"\n' "$app_dir"
