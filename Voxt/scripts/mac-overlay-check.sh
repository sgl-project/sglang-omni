#!/usr/bin/env bash
# The overlay's visual check on a Mac (docs/manual-test.md §10.5): builds
# the app as a small accessory bundle (not the installed VoxtNew: another
# bundle ID, no data, no helper), opens it in the background (open -g: it
# never comes to the front), and captures each state of the overlay it
# walks through over a light and a dark backdrop at the overlay's own
# place (about a second each, ~15 s in all). Captures are screencapture's,
# from this script: the window alone (-l) and the backdrop with the
# overlay on it (-R); the clipboard, the keyboard and the app in front are
# left alone. Niced, two jobs; it stops when less than VOXTNEW_MIN_FREE
# percent of memory (40) is free; nothing stays running.
#
#   MAC_DIR=code/VoxtNew-ov scripts/mac.sh scripts/mac-overlay-check.sh
#   scripts/mac.sh --pull build/overlay-check     # (same MAC_DIR) bring the results back
#
# Output in build/overlay-check/shots: step-N-window.png, step-N-screen.png,
# frames.txt (each step's frame; FAIL lines when the anchor moved), and
# sheet.png (the screens side by side, the anchor drawn in red).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run it on a Mac: MAC_DIR=code/VoxtNew-ov scripts/mac.sh scripts/mac-overlay-check.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export GOMAXPROCS=2 GOFLAGS=-p=2

free="$(memory_pressure -Q | sed -n 's/.*percentage: \([0-9]*\)%.*/\1/p')"
echo "== free memory ${free}%"
if (( ${free:-0} < ${VOXTNEW_MIN_FREE:-40} )); then
  echo "less than ${VOXTNEW_MIN_FREE:-40}% of memory is free: not starting" >&2
  exit 3
fi

out="$ROOT/build/overlay-check"
app="$out/VoxtNewOverlayCheck.app"
shots="$out/shots"
rm -rf "$out"
mkdir -p "$app/Contents/MacOS" "$shots"

echo "== build"
nice -n 10 go build -o "$app/Contents/MacOS/VoxtNewOverlayCheck" .
cat >"$app/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>dev.hehehai.voxtnew.overlaycheck</string>
  <key>CFBundleName</key><string>VoxtNewOverlayCheck</string>
  <key>CFBundleExecutable</key><string>VoxtNewOverlayCheck</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

front() { lsappinfo info -only bundleid "$(lsappinfo front)" 2>/dev/null | sed 's/.*="\(.*\)"/\1/'; }
before="$(front)"
echo "== run (in front: $before)"
open -g -n -W --stdout "$out/check.log" --stderr "$out/check.log" "$app" --args --overlay-check "$shots" &
opener=$!

n=0
reniced=0
deadline=$((SECONDS + 120))
while [[ ! -f "$shots/finished" ]]; do
  if ((reniced == 0)) && pid="$(pgrep -f "$app/Contents/MacOS/VoxtNewOverlayCheck")"; then
    renice -n 10 -p $pid >/dev/null || true
    reniced=1
  fi
  if [[ -f "$shots/step-$n.ready" ]]; then
    read -r wid x y w h <"$shots/step-$n.ready"
    screencapture -x -o -l"$wid" "$shots/step-$n-window.png"
    screencapture -x -R"$x,$y,$w,$h" "$shots/step-$n-screen.png"
    touch "$shots/step-$n.done"
    n=$((n + 1))
    continue
  fi
  # Steps without a capture leave no file: look further ahead.
  for ((k = n + 1; k <= n + 3; k++)); do
    if [[ -f "$shots/step-$k.ready" ]]; then n=$k; continue 2; fi
  done
  if ((SECONDS > deadline)) || ! kill -0 "$opener" 2>/dev/null; then
    echo "the check did not finish" >&2
    tail -40 "$out/check.log" >&2 || true
    pkill -f "$app/Contents/MacOS/VoxtNewOverlayCheck" || true
    exit 1
  fi
  sleep 0.05
done
wait "$opener" || true
after="$(front)"

rm -f "$shots"/*.ready "$shots"/*.done "$shots/finished"
cat "$shots/frames.txt"
echo "== in front before: $before, after: $after; $(ls "$shots"/*-screen.png | wc -l | tr -d ' ') states captured; $shots/sheet.png"
status=0
if pgrep -f "$app/Contents/MacOS/VoxtNewOverlayCheck" >/dev/null; then
  echo "FAIL the check is still running" >&2
  status=1
fi
if [[ "$before" != "$after" ]]; then
  echo "FAIL the app in front changed" >&2
  status=1
fi
if grep -q '^FAIL' "$shots/frames.txt"; then
  status=1
fi
exit $status
