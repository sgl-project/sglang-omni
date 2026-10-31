#!/bin/bash
# End to end without a phone: the iPhone simulator dictates through Voxt's Local API on this Mac
# (Try it, and the app's own text field), then runs the design tour against
# tools/design-tour-fixture.py. Results go to docs/evidence/loopback-e2e/simulator/, without paths or tokens.
# The build comes first, so a Local API that serves for a limited time only has to cover the tests.
#
#   VOXT_E2E_URL=http://127.0.0.1:PORT VOXT_E2E_TOKEN=TOKEN tools/loopback-e2e.sh [options]
#
# Port and token: Voxt Settings › Local API. VOXT_E2E_ENV_FILE may name a file with those two
# KEY=VALUE lines instead; with VOXT_E2E_READY_FILE it is read once that file exists (waiting up
# to VOXT_E2E_WAIT seconds, default 1800). VOXT_E2E_EXPECT is text the Mac's result must contain ("country" for
# testdata/jfk.wav). Needs Xcode with an iOS simulator runtime and the Tailcat library
# (tools/build-tailcat-native.sh).
#   --simulator UDID   which simulator (default: the first available iPhone)
#   --erase            erase that simulator first, as a fresh machine has it
#   --dry-run          print the steps instead of running them
# VOXT_E2E_WORK holds the build and result bundles (default DerivedData/loopback-e2e).
set -uo pipefail
mobile="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
simulator="" erase=0 dry=0
while (($#)); do
  case $1 in
    --simulator) simulator=$2; shift 2 ;;
    --erase) erase=1; shift ;;
    --dry-run) dry=1; shift ;;
    --check-evidence) check_only=$2; shift 2 ;;
    --test-list) list_only=$2; shift 2 ;;
    *) echo "loopback-e2e: unknown argument $1" >&2; exit 2 ;;
  esac
done

# Fails when an evidence file holds a local path, the user name or the token.
check_evidence() {
  # Built from parts so this file itself names no local path.
  local folder=$1 token=${2:-} hits s=/ pattern
  pattern="${s}Users${s}|${s}Volumes${s}|${s}private${s}"
  # The user name only as part of a path; a short name also occurs inside ordinary words.
  [[ -n ${USER:-} ]] && pattern+="|~$USER|/$USER/"
  hits=$(grep -rlE "$pattern" "$folder" 2>/dev/null)
  [[ -n $token ]] && hits+=$(grep -rlF -- "$token" "$folder" 2>/dev/null)
  [[ -z $hits ]] || { echo "loopback-e2e: evidence names a local path or the token: $hits" >&2; return 1; }
}
if [[ -n ${check_only:-} ]]; then check_evidence "$check_only" "${VOXT_E2E_TOKEN:-}"; exit; fi

# One line per test in an xcodebuild log, then totals over every launch: after a runner restart,
# xcodebuild's own "Executed N tests" line covers only the last launch.
test_list() {
  grep -E "^Test Case .* (passed|failed|skipped)" "$1" |
    sed -E "s/^Test Case '-\[VoxtMobileUITests\.([A-Za-z]+) ([A-Za-z]+)\]' (passed|failed|skipped) \(([0-9.]+) seconds\)\./\1.\2 \3 \4 s/" |
    awk '{ print; n++; count[$2]++; total += $3 }
      END { printf "%d tests: %d passed, %d failed", n, count["passed"], count["failed"]
            if (count["skipped"]) printf ", %d skipped", count["skipped"]
            printf ", %.1f s\n", total }'
}
if [[ -n ${list_only:-} ]]; then test_list "$list_only"; exit; fi

# Sets VOXT_E2E_URL and VOXT_E2E_TOKEN, from VOXT_E2E_ENV_FILE when one is named.
read_local_api() {
  if [[ -n ${VOXT_E2E_ENV_FILE:-} ]]; then
    [[ -f $VOXT_E2E_ENV_FILE ]] || { echo "loopback-e2e: VOXT_E2E_ENV_FILE is not a file" >&2; return 2; }
    while IFS='=' read -r key value; do
      case $key in VOXT_E2E_URL | VOXT_E2E_TOKEN) export "$key=$value" ;; esac
    done < "$VOXT_E2E_ENV_FILE"
  fi
  [[ -n ${VOXT_E2E_URL:-} && -n ${VOXT_E2E_TOKEN:-} ]] || {
    echo "loopback-e2e: set VOXT_E2E_URL=http://127.0.0.1:PORT and VOXT_E2E_TOKEN from Voxt Settings › Local API" >&2
    return 2
  }
}
# Without a ready file the Local API must be known before anything runs.
if [[ -z ${VOXT_E2E_READY_FILE:-} ]]; then read_local_api || exit 2; fi
expect=${VOXT_E2E_EXPECT:-country}
wav="$mobile/testdata/jfk.wav"
work=${VOXT_E2E_WORK:-$mobile/DerivedData/loopback-e2e}
evidence="$mobile/docs/evidence/loopback-e2e/simulator"
project="$mobile/ios/VoxtMobile.xcodeproj"

run() {
  local line="$*"
  [[ -n ${VOXT_E2E_TOKEN:-} ]] && line=${line//"$VOXT_E2E_TOKEN"/<token>}
  if ((dry)); then printf '+ %s\n' "$line"; else "$@"; fi
}
# run_logged LOG command…: like run, with the command's output in LOG.
run_logged() {
  local log=$1; shift
  if ((dry)); then run "$@"; else "$@" > "$log" 2>&1; fi
}

# Waits for the ready file, reads the Local API and checks it answers.
local_api_up() {
  if [[ -n ${VOXT_E2E_READY_FILE:-} ]]; then
    echo "loopback-e2e: waiting for $(basename "$VOXT_E2E_READY_FILE")"
    local waited=0
    until [[ -e $VOXT_E2E_READY_FILE ]]; do
      ((waited >= ${VOXT_E2E_WAIT:-1800})) && { echo "loopback-e2e: the Local API never became ready" >&2; return 2; }
      sleep 1; ((waited++))
    done
    read_local_api || return 2
  fi
  ((dry)) && return 0
  curl -fsS -m 5 -H "Authorization: Bearer $VOXT_E2E_TOKEN" "$VOXT_E2E_URL/v1/health" > /dev/null || {
    echo "loopback-e2e: Voxt's Local API does not answer at $VOXT_E2E_URL with that token" >&2; return 2; }
}

if ((!dry)); then
  [[ -d "$mobile/ios/VoxtTailcat/Artifacts/CVoxtTailcat.xcframework" ]] || {
    echo "loopback-e2e: build the Tailcat library first: tools/build-tailcat-native.sh" >&2; exit 2; }
fi
if [[ -z $simulator ]]; then
  simulator=$(xcrun simctl list devices available -j 2>/dev/null | python3 -c '
import json, sys
devices = [d for runtime in json.load(sys.stdin)["devices"].values() for d in runtime]
print(next((d["udid"] for d in devices if d["name"].startswith("iPhone")), ""))')
  [[ -n $simulator ]] || { echo "loopback-e2e: no iPhone simulator; add one in Xcode › Settings › Components" >&2; exit 2; }
fi

((dry)) || mkdir -p "$work" "$evidence"
if ((erase)); then
  run xcrun simctl shutdown "$simulator"
  run xcrun simctl erase "$simulator"
fi
run xcrun simctl boot "$simulator"
run xcrun simctl bootstatus "$simulator" -b

# Simulator builds sign to run locally; no team or Keychain is needed.
common=(-project "$project" -scheme VoxtMobile -destination "id=$simulator" -derivedDataPath "$work/DerivedData"
  -packageCachePath "${VOXT_PACKAGE_CACHE:-$work/SourcePackages}" -parallel-testing-enabled NO
  -collect-test-diagnostics never DEVELOPMENT_TEAM= CODE_SIGN_IDENTITY=-)
run_logged "$work/build.log" xcodebuild build-for-testing "${common[@]}" || {
  echo "loopback-e2e: build failed; see $work/build.log" >&2; exit 1; }

# test GROUP URL TOKEN EXPECT -only-testing:…  — one xcodebuild run, its results kept in evidence.
test_group() {
  local group=$1 url=$2 token=$3 expected=$4; shift 4
  rm -rf "$work/$group.xcresult"
  run_logged "$work/$group.log" env TEST_RUNNER_VOXT_E2E_URL="$url" TEST_RUNNER_VOXT_E2E_TOKEN="$token" \
    TEST_RUNNER_VOXT_E2E_WAV="$wav" TEST_RUNNER_VOXT_E2E_EXPECT="$expected" TEST_RUNNER_VOXT_DESIGN_FIXTURE_URL="$url" \
    xcodebuild test-without-building "${common[@]}" -resultBundlePath "$work/$group.xcresult" "$@"
  local status=$?
  ((dry)) && return 0
  {
    echo "exit $status"
    test_list "$work/$group.log"
    grep -E "^UI E2E (in-app|keyboard) text: " "$work/$group.log"
  } > "$evidence/$group.txt"
  return $status
}

# What the evidence shows and the fresh-simulator cases the tests handle themselves.
readme() {
  cat <<'TEXT'
# Loopback end to end

`tools/loopback-e2e.sh` writes these files. `summary.md` names the source commit, Xcode and the
simulator; `local-api.txt` and `design-tour.txt` list each UI test with its result and seconds, then
the totals over the whole run, and the text Voxt returned for `testdata/jfk.wav`.

- `local-api.txt`: Try it and a dictation into the app's own text field, through the Voxt keyboard,
  against Voxt's Local API on this Mac with the models selected in Voxt.
- `design-tour.txt`: the design tour against `tools/design-tour-fixture.py`, which answers with the
  recorded contract and no model.

A freshly erased simulator differs from a used one in five ways, and the tests handle each:

- Settings lists the Voxt keyboard only after Voxt has been opened once, so the tests open it before
  turning the keyboard on.
- The first keyboard change shows iOS's one-time "Quickly Change Keyboards" tip over the keyboard;
  the tests close it with Continue.
- Safari's first load of the fixture's page can stay blank or keep loading; the tests reload it or
  open it again.
- Safari's one-time toolbar tip takes the next tap; the tests close it before tapping the field.
- The first system keyboard shows iOS's one-time slide-to-type tip over the keys, and switching
  keyboards does nothing until it is closed; the tests close it with Continue.

Run with `--erase` to start from that state.
TEXT
}
if ((dry)); then echo "+ write $evidence/README.md"; else readme > "$evidence/README.md"; fi

# The Local API tests run while it serves; they use the app's own fields, not Safari.
local_api_up || exit 2
test_group local-api "$VOXT_E2E_URL" "$VOXT_E2E_TOKEN" "$expect" \
  -only-testing:VoxtMobileUITests/TryItEndToEndUITests \
  -only-testing:VoxtMobileUITests/DictationFlowUITests/testInAppDictationGoesThroughMac
local_api=$?

# Then the design tour; on an erased simulator it opens Safari for the first time.
fixture_dir="$work/fixture"; rm -rf "$fixture_dir"
if ((dry)); then
  echo "+ python3 -I $mobile/tools/design-tour-fixture.py --output $fixture_dir"
  fixture_url="http://127.0.0.1:FIXTURE_PORT"
else
  python3 -I "$mobile/tools/design-tour-fixture.py" --output "$fixture_dir" > "$work/fixture.log" 2>&1 &
  fixture_pid=$!
  trap 'kill $fixture_pid 2>/dev/null' EXIT
  for _ in $(seq 100); do [[ -s "$fixture_dir/endpoint.txt" ]] && break; sleep 0.2; done
  [[ -s "$fixture_dir/endpoint.txt" ]] || { echo "loopback-e2e: design-tour fixture did not start" >&2; exit 1; }
  fixture_url=$(cat "$fixture_dir/endpoint.txt")
  # The fixture answers from the recorded contract; without it every design-tour test fails to pair.
  curl -fsS -m 5 -H "Authorization: Bearer design-review" "$fixture_url/v1/health" > /dev/null || {
    echo "loopback-e2e: design-tour fixture does not answer /v1/health; see $work/fixture.log" >&2; exit 1; }
fi
test_group design-tour "$fixture_url" design-review "design review dictation" \
  -only-testing:VoxtMobileUITests/DesignTourUITests
design=$?
run xcrun simctl shutdown "$simulator"
((dry)) && exit 0

{
  echo "# Loopback end to end"
  echo
  echo "- Source: $(git -C "$mobile" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "- Xcode: $(xcodebuild -version | head -1 | cut -d' ' -f2)"
  echo "- Simulator: $(xcrun simctl list devices -j | python3 -c '
import json, sys
for runtime, devices in json.load(sys.stdin)["devices"].items():
    for d in devices:
        if d["udid"] == sys.argv[1]:
            print(d["name"] + ", " + runtime.rsplit(".", 1)[-1].replace("-", " ", 1).replace("-", "."))
' "$simulator")$( ((erase)) && echo ", erased first")"
  echo "- Try it and the app's own field against Voxt's Local API, testdata/jfk.wav, expecting \"$expect\": exit $local_api (local-api.txt)"
  echo "- Design tour against tools/design-tour-fixture.py, after those: exit $design (design-tour.txt)"
} > "$evidence/summary.md"
check_evidence "$evidence" "$VOXT_E2E_TOKEN" || exit 1
echo "loopback-e2e: design tour exit $design, Local API exit $local_api; see docs/evidence/loopback-e2e/simulator/"
((design == 0 && local_api == 0))
