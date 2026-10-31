#!/usr/bin/env bash
# The logs' selection on a Mac, captured (the ActNow family's log-selection
# spec): builds VoxtNew with -tags logscheck as a small bundle of its own
# (another bundle ID, a temporary library, no helper: not the installed
# VoxtNew), opens it on Logs with sample lines, works the real pointer and
# keys over it (scripts/logs-shots/drive.swift: hover, click, Shift-click,
# ⌘-click, the context menu, ⌘C, dragging, ⌘A, Esc) and captures the
# window after each, in English and in Chinese. It moves the pointer and
# takes the front for about half a minute: it starts only when nobody has
# touched the Mac for VOXTNEW_SHOTS_IDLE seconds (300), and the driver
# stops as soon as the pointer moves under it. The pointer and the app in
# front come back. VOXTNEW_SHOTS_DRY=1 only opens the window in the
# background and captures it as it opens (no pointer, any time).
# VOXTNEW_SHOTS_ORDER=1 captures the order instead (the newest first; new
# lines while at the top, and while scrolled down, with the pill), into
# build/logs-shots/order-<lang>.
#
#   MAC_DIR=code/VoxtNew-logsel scripts/mac.sh scripts/mac-logs-shots.sh
#   MAC_DIR=code/VoxtNew-logsel scripts/mac.sh --pull build/logs-shots
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run it on a Mac: MAC_DIR=code/VoxtNew-logsel scripts/mac.sh scripts/mac-logs-shots.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export GOMAXPROCS=2 GOFLAGS=-p=2

idle() { echo $(( $(ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF; exit}') / 1000000000 )); }
dry="${VOXTNEW_SHOTS_DRY:-}"
if [[ -z "$dry" ]] && (( $(idle) < ${VOXTNEW_SHOTS_IDLE:-300} )); then
  echo "someone used the Mac $(idle) s ago: not moving the pointer (VOXTNEW_SHOTS_IDLE=${VOXTNEW_SHOTS_IDLE:-300})" >&2
  exit 4
fi

out="$ROOT/build/logs-shots"
app="$out/VoxtNewLogsCheck.app"
rm -rf "$out"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
echo "== build"
nice -n 10 go build -tags logscheck -o "$app/Contents/MacOS/VoxtNewLogsCheck" .
swiftc -O scripts/logs-shots/drive.swift -o "$out/drive"
# The store's SQLite, which mygo build puts in an app: from a built one.
for lib in "$HOME"/code/VoxtNew*/build/darwin-arm64/VoxtNew.app/Contents/Resources/libmygo-sqlite3.dylib \
  /Applications/VoxtNew.app/Contents/Resources/libmygo-sqlite3.dylib; do
  [[ -f "$lib" ]] && { cp "$lib" "$app/Contents/Resources/"; break; }
done
cat >"$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>dev.hehehai.voxtnew.logscheck</string>
  <key>CFBundleName</key><string>VoxtNewLogsCheck</string>
  <key>CFBundleExecutable</key><string>VoxtNewLogsCheck</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

front() { lsappinfo info -only bundleid "$(lsappinfo front)" 2>/dev/null | sed 's/.*="\(.*\)"/\1/'; }
before="$(front)"
for lang in en zh-Hans; do
  shots="$out/${VOXTNEW_SHOTS_ORDER:+order-}$lang"
  mkdir -p "$shots"
  echo "== $lang (in front: $before)"
  open ${dry:+-g} -n -W --stdout "$out/check-$lang.log" --stderr "$out/check-$lang.log" "$app" --args --logs-check "$shots" --lang "$lang" &
  opener=$!
  for _ in $(seq 100); do [[ -f "$shots/geometry.json" ]] && break; sleep 0.2; done
  [[ -f "$shots/geometry.json" ]] || { echo "no geometry.json" >&2; cat "$out/check-$lang.log" >&2; exit 1; }
  sleep 0.5
  status=0
  if [[ -n "$dry" ]]; then
    wid="$(sed -n 's/.*"window":\([0-9]*\).*/\1/p' "$shots/geometry.json")"
    screencapture -x -o -l"$wid" "$shots/0-opened.png"
    cat "$shots/geometry.json"; echo
  else
    "$out/drive" "$shots" ${VOXTNEW_SHOTS_ORDER:+order} || status=$?
  fi
  touch "$shots/finished"
  wait "$opener" || true
  (( status == 0 )) || { echo "the driver stopped ($status)" >&2; break; }
done
[[ -n "$before" ]] && open -b "$before" || true
rm -rf "$app" "$out/drive"
ls -la "$out"/*/
