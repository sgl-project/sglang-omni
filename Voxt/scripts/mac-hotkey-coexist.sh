#!/usr/bin/env bash
# VoxtNew beside Auralink (docs/release.md §8, the ActNow family's first-
# release check): on a Mac where Auralink runs with its fn shortcuts, the
# helper of a built VoxtNew.app gets VoxtNew's default shortcuts (right ⌥,
# right ⌥ ⇧, ⌃ right ⌥, ⌃⌥V) and installs its event tap; synthetic key
# events (scripts/hotkey-coexist/poster.swift, posted at the HID level with
# a real keyboard's flags) then go through both apps' taps, and:
#
#   - right ⌥ alone, with ⇧, with ⌃ fire VoxtNew's Transcribe, Translate,
#     Rewrite, and nothing in Auralink (its log gets no hotkey line);
#   - left ⌥ fires nothing;
#   - fn (marked as Auralink's own posted event, so that the running
#     Auralink lets it through instead of starting a recording) fires
#     nothing in VoxtNew, and VoxtNew does not swallow it;
#   - a right ⌥ tap marked "VOXT" (VoxtNew's own posted events) is let
#     through unjudged; ⌃⌥V marked "AURA" (another app's posted keys) is
#     judged: VoxtNew's paste-last fires and the V is swallowed;
#   - no modifier event is swallowed: a tap appended after every other one
#     sees them all.
#
# Only modifier events reach the app in front (they do nothing there); the
# ⌃⌥V is swallowed. fn is never posted unmarked: that would start a
# recording in the user's Auralink. Nothing is recorded, pasted or typed:
# the helper only reports what fired. It needs Accessibility for the
# process that runs it (over ssh on mac-air, sshd-keygen-wrapper has it)
# and asks VOXTNEW_COEXIST_OK=1, since it posts key events on a Mac that may
# be in use.
#
#   MAC_DIR=code/VoxtNew-f2 scripts/mac.sh 'VOXTNEW_COEXIST_OK=1 scripts/mac-hotkey-coexist.sh [APP]'
#
# APP defaults to build/darwin-arm64/VoxtNew.app (go tool mygo build or
# scripts/release.sh); a helper binary works too
# (resources/darwin-arm64/helper/voxtnew-helper of scripts/build-helper.sh).
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
  echo "run it on the Mac: scripts/mac.sh 'VOXTNEW_COEXIST_OK=1 scripts/mac-hotkey-coexist.sh'" >&2
  exit 1
fi
if [[ "${VOXTNEW_COEXIST_OK:-}" != 1 ]]; then
  echo "it posts modifier key events (and a swallowed ⌃⌥V) on this Mac: set VOXTNEW_COEXIST_OK=1" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$ROOT/build/darwin-arm64/VoxtNew.app}"
HELPER="$APP/Contents/Resources/helper/voxtnew-helper"
[[ -f "$APP" && -x "$APP" ]] && HELPER="$APP" # the helper itself (scripts/build-helper.sh)
[[ -x "$HELPER" ]] || { echo "no helper in $APP" >&2; exit 1; }
AURALOG="$HOME/Library/Logs/Auralink/auralink.log"
WORK="$ROOT/build/hotkey-coexist"
rm -rf "$WORK"
mkdir -p "$WORK"
command -v jq >/dev/null || { echo "needs jq" >&2; exit 1; }

swiftc -O "$ROOT/scripts/hotkey-coexist/poster.swift" -o "$WORK/poster" 2>"$WORK/swiftc.log" ||
  { cat "$WORK/swiftc.log" >&2; exit 1; }

tail_pid="" helper_pid=""
cleanup() {
  exec 3>&- 2>/dev/null || true
  [[ -n "$helper_pid" ]] && kill "$helper_pid" 2>/dev/null || true
  [[ -n "$tail_pid" ]] && kill "$tail_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

auralink=""
if pgrep -fq 'Auralink.app/Contents/Resources/helper/auralink-helper'; then
  auralink="$(pgrep -fl 'Auralink.app/Contents/MacOS/Auralink' | head -1 | cut -d' ' -f2-)"
  echo "Auralink runs: $auralink"
else
  echo "Auralink does not run: only VoxtNew's side is checked"
fi
aura_before=0
[[ -f "$AURALOG" ]] && aura_before="$(grep -c ' hotkey ' "$AURALOG" || true)"

# The helper, with VoxtNew's defaults (internal/hotkey Build of settings.Defaults).
mkfifo "$WORK/in"
"$HELPER" <"$WORK/in" >"$WORK/out.jsonl" 2>"$WORK/helper.log" &
helper_pid=$!
exec 3>"$WORK/in"
send() { printf '%s\n' "$1" >&3; }
wait_for() { # wait_for ID: the response line
  for _ in $(seq 1 50); do
    if grep -q "\"id\":\"$1\"" "$WORK/out.jsonl"; then
      grep "\"id\":\"$1\"" "$WORK/out.jsonl" | head -1
      return 0
    fi
    sleep 0.1
  done
  echo "no answer to $1" >&2
  return 1
}
send '{"id":"c1","method":"helper.hello","params":{"protocol":1,"app":"coexist","logLevel":"debug"}}'
wait_for c1 >/dev/null
send '{"id":"c2","method":"hotkey.configure","params":{"bindings":[{"id":"transcribe/0","modifiers":{"option":"right"},"behavior":"tap"},{"id":"translate/0","modifiers":{"option":"right","shift":"any"},"behavior":"tap"},{"id":"rewrite/0","modifiers":{"control":"any","option":"right"},"behavior":"tap"},{"id":"paste-last","modifiers":{"control":"any","option":"any"},"key":9,"behavior":"tap"}]}}'
status="$(wait_for c2)"
if [[ "$(jq -r .result.installed <<<"$status")" != true ]]; then
  echo "VoxtNew's event tap is not installed (Accessibility?): $status" >&2
  exit 1
fi
# The tail tap after it (and after Auralink's).
"$WORK/poster" tail >"$WORK/tail.txt" 2>"$WORK/tail.err" &
tail_pid=$!
for _ in $(seq 1 30); do grep -q ready "$WORK/tail.txt" 2>/dev/null && break; sleep 0.1; done
grep -q ready "$WORK/tail.txt" || { echo "the tail tap did not start: $(cat "$WORK/tail.err")" >&2; exit 1; }

AURA=0x41555241 VOXT=0x564F5854
FAILS=0
# check NAME "EXPECTED BINDINGS" "TAIL GREP (events that must pass)" SPEC…
check() {
  local name="$1" want="$2" pass="$3"
  shift 3
  local n0 t0 got passed
  n0="$(wc -l <"$WORK/out.jsonl")" t0="$(wc -l <"$WORK/tail.txt")"
  "$WORK/poster" post "$@"
  sleep 0.7
  got="$(tail -n "+$((n0 + 1))" "$WORK/out.jsonl" | jq -r 'select(.method == "hotkey.event") | .params.binding' | tr '\n' ' ' | sed 's/ $//')"
  passed="$(tail -n "+$((t0 + 1))" "$WORK/tail.txt" | awk '{print $1 ":" $2}' | tr '\n' ' ' | sed 's/ $//')"
  if [[ "$got" == "$want" && "$passed" == "$pass" ]]; then
    echo "  ok   $name: fired [${got}], passed [${passed}]"
  else
    echo "  FAIL $name: fired [${got}] (want [${want}]), passed [${passed}] (want [${pass}])"
    FAILS=$((FAILS + 1))
  fi
}
echo "== VoxtNew's shortcuts beside Auralink's"
check "right ⌥ tap" "transcribe/0" "f:61 f:61" f:61:0x80140 s:90 f:61:0x100
check "right ⌥ ⇧" "translate/0" "f:61 f:56 f:56 f:61" f:61:0x80140 s:30 f:56:0xa0142 s:60 f:56:0x80140 s:20 f:61:0x100
check "⌃ right ⌥" "rewrite/0" "f:59 f:61 f:61 f:59" f:59:0x40101 s:30 f:61:0xc0141 s:60 f:61:0x40101 s:20 f:59:0x100
check "left ⌥ tap" "" "f:58 f:58" f:58:0x80120 s:90 f:58:0x100
# fn passes VoxtNew's tap and Auralink's (marked): the system then acts on
# it as "Press 🌐 key to" says; only when that is "Do Nothing".
if [[ "$(defaults read com.apple.HIToolbox AppleFnUsageType 2>/dev/null || echo 1)" == 0 ]]; then
  check "fn tap (Auralink's mark)" "" "f:63 f:63" f:63:0x800100:$AURA s:90 f:63:0x100:$AURA
else
  echo "  skip fn: the 🌐 key does something on this Mac (System Settings → Keyboard)"
fi
check "right ⌥ tap (VoxtNew's mark)" "" "f:61 f:61" f:61:0x80140:$VOXT s:90 f:61:0x100:$VOXT
check "⌃⌥V (Auralink's mark)" "paste-last" "" d:9:0xc0121:$AURA s:40 u:9:0xc0121:$AURA

send '{"id":"c3","method":"hotkey.status"}'
echo "   helper: $(wait_for c3 | jq -c '.result | {installed, events, missedEvents, maxCallbackMicros}')"
if [[ -n "$auralink" && -f "$AURALOG" ]]; then
  aura_after="$(grep -c ' hotkey ' "$AURALOG" || true)"
  if ((aura_after == aura_before)); then
    echo "  ok   Auralink fired nothing (no new hotkey line in its log)"
  else
    echo "  FAIL Auralink fired: $(tail -n "$((aura_after - aura_before))" <(grep ' hotkey ' "$AURALOG") | tr '\n' ' ')"
    FAILS=$((FAILS + 1))
  fi
fi
send '{"id":"c4","method":"helper.shutdown"}'
wait_for c4 >/dev/null || true
if ((FAILS)); then
  echo "$FAILS check(s) failed; the helper's log: $WORK/helper.log"
  exit 1
fi
echo "all checks passed"
