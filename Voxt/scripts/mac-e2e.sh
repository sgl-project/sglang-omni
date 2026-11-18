#!/usr/bin/env bash
# The headless Mac tests (docs/manual-test.md, "自动真机测试（无界面）"), on
# a Mac that may be busy with other work: no window, no change of the app
# in front, no key posted, no event tap, no clipboard, no microphone (the
# helper records a file). Niced, two jobs, one step at a time; it stops
# when less than VOXTNEW_MIN_FREE percent of memory (40) is free.
#
#   MAC_DIR=code/VoxtNew-mt scripts/mac.sh scripts/mac-e2e.sh
#
# Nothing reaches the network: a session runs over the fake local models
# (TestMacE2ESessionLocal). --offline is accepted and changes nothing (it
# skipped Auralink's real-service tests).
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run it on a Mac: MAC_DIR=code/VoxtNew-mt scripts/mac.sh scripts/mac-e2e.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export GOMAXPROCS=2 GOFLAGS=-p=2 SWIFT_BUILD_JOBS=2

step() {
  local free
  free="$(memory_pressure -Q | sed -n 's/.*percentage: \([0-9]*\)%.*/\1/p')"
  echo "== $1 (free memory ${free}%)"
  if (( ${free:-0} < ${VOXTNEW_MIN_FREE:-40} )); then
    echo "less than ${VOXTNEW_MIN_FREE:-40}% of memory is free: stopping" >&2
    exit 3
  fi
}

step "the helper (release build)"
nice -n 10 scripts/build-helper.sh --out build/mac-e2e/helper
export VOXTNEW_HELPER="$ROOT/build/mac-e2e/helper/voxtnew-helper" VOXTNEW_MAC_E2E=1

step "swift test"
(cd helper && nice -n 10 swift test -j 2 2>&1 | grep -E "error:|failed|Executed .* tests" | tail -5)

step "the helper, read-only and the file input"
nice -n 10 go test -count=1 -run 'TestMacHelperReadOnly|TestMacHelperIcons|TestMacHelperCaptureFile' -v . | grep -vE '^\s*$'

step "text fit (Core Text)"
VOXTNEW_TEXTFIT_PROBE=1 VOXTNEW_TEXTFIT_AUDIT=strict nice -n 10 go test -count=1 -run 'TestTextFitsItsOwnWidth|TestTranslationsFit' -v . |
  grep -E '^(---|ok|FAIL|PASS)|break at'

step "a session over the fake local models"
nice -n 10 go test -count=1 -run 'TestMacE2ESessionLocal' -v . | grep -vE '^\s*$'

if pgrep -f "$VOXTNEW_HELPER" >/dev/null; then
  echo "a helper is still running: $(pgrep -f "$VOXTNEW_HELPER" | tr '\n' ' ')" >&2
  exit 1
fi
echo "== done; no helper left running"
