#!/bin/bash
# Checks tools/loopback-e2e.sh without a simulator or a Mac server: its refusals, the steps of a
# dry run, and the evidence check. Usage: tools/test-loopback-e2e.sh
set -uo pipefail
script="$(cd "$(dirname "$0")" && pwd)/loopback-e2e.sh"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: expected [$3], got [$2]"; fail=1; fi; }
has() { if grep -qF -- "$2" "$3"; then echo "ok   $1"; else echo "FAIL $1: no [$2]"; fail=1; fi; }

env -u VOXT_E2E_URL -u VOXT_E2E_TOKEN "$script" --dry-run --simulator SIM > "$tmp/out" 2>&1
check "no URL or token: exit" $? 2
has "no URL or token: says where they come from" "Settings › Local API" "$tmp/out"

export VOXT_E2E_WORK="$tmp/work"
VOXT_E2E_URL=http://127.0.0.1:47910 VOXT_E2E_TOKEN=fake-secret-token "$script" --dry-run --erase --simulator SIM > "$tmp/out" 2>&1
check "dry run: exit" $? 0
check "dry run never prints the token" "$(grep -c fake-secret-token "$tmp/out")" 0
has "erases the simulator first" "simctl erase SIM" "$tmp/out"
has "builds once" "xcodebuild build-for-testing" "$tmp/out"
has "signs to run locally" "DEVELOPMENT_TEAM= CODE_SIGN_IDENTITY=-" "$tmp/out"
has "design tour against the fixture" "-only-testing:VoxtMobileUITests/DesignTourUITests" "$tmp/out"
has "Try it against the Local API" "-only-testing:VoxtMobileUITests/TryItEndToEndUITests" "$tmp/out"
has "the app's own field against the Local API" "DictationFlowUITests/testInAppDictationGoesThroughMac" "$tmp/out"
has "a dry run names the README" "README.md" "$tmp/out"
has "the Local API run gets its URL" "TEST_RUNNER_VOXT_E2E_URL=http://127.0.0.1:47910" "$tmp/out"
# The Local API serves for a limited time: build first, then its tests, then the design tour.
order() { grep -n -e "$1" "$tmp/out" | head -1 | cut -d: -f1; }
check "build, then Local API tests, then design tour" \
  "$(( $(order build-for-testing) < $(order TryItEndToEndUITests) && $(order TryItEndToEndUITests) < $(order DesignTourUITests) ))" 1

printf 'VOXT_E2E_URL=http://127.0.0.1:47911\nVOXT_E2E_TOKEN=file-secret\n' > "$tmp/local-api.env"
env -u VOXT_E2E_URL -u VOXT_E2E_TOKEN VOXT_E2E_ENV_FILE="$tmp/local-api.env" "$script" --dry-run --simulator SIM > "$tmp/out" 2>&1
check "env file: exit" $? 0
has "env file: URL read" "TEST_RUNNER_VOXT_E2E_URL=http://127.0.0.1:47911" "$tmp/out"
check "env file: token not printed" "$(grep -c file-secret "$tmp/out")" 0

# With VOXT_E2E_READY_FILE the hand-off file is read only once that file exists.
rm -f "$tmp/ready"
( sleep 2; printf 'VOXT_E2E_URL=http://127.0.0.1:47912\nVOXT_E2E_TOKEN=late-secret\n' > "$tmp/late.env"; touch "$tmp/ready" ) &
env -u VOXT_E2E_URL -u VOXT_E2E_TOKEN VOXT_E2E_ENV_FILE="$tmp/late.env" VOXT_E2E_READY_FILE="$tmp/ready" \
  "$script" --dry-run --simulator SIM > "$tmp/out" 2>&1
check "ready file: exit" $? 0
has "ready file: URL read after it appears" "TEST_RUNNER_VOXT_E2E_URL=http://127.0.0.1:47912" "$tmp/out"
has "ready file: builds before waiting" "xcodebuild build-for-testing" "$tmp/out"
check "ready file: waits before reading the URL" "$(( $(order 'waiting for') < $(order 47912) ))" 1
wait
VOXT_E2E_WAIT=1 env -u VOXT_E2E_URL -u VOXT_E2E_TOKEN VOXT_E2E_ENV_FILE="$tmp/never.env" VOXT_E2E_READY_FILE="$tmp/never" \
  "$script" --dry-run --simulator SIM > "$tmp/out" 2>&1
check "ready file that never comes: exit" $? 2

mkdir -p "$tmp/evidence"; echo "Test passed (1.0 s)" > "$tmp/evidence/a.txt"
VOXT_E2E_TOKEN=fake-secret-token "$script" --check-evidence "$tmp/evidence" > /dev/null 2>&1
check "clean evidence passes" $? 0
# The user name counts only as part of a path: a short name such as "me" is in ordinary words.
echo "DesignTourUITests.testFirstHandoffInsertsIntoSameField passed" > "$tmp/evidence/b.txt"
USER=me VOXT_E2E_TOKEN=fake-secret-token "$script" --check-evidence "$tmp/evidence" > /dev/null 2>&1
check "a short user name inside a word passes" $? 0
s=/  # paths built from parts, so this file itself names none
for planted in "${s}Users${s}someone${s}x" "${s}Volumes${s}Disk${s}x" "token fake-secret-token" "~me/notes" "${s}home${s}me${s}x"; do
  echo "$planted" > "$tmp/evidence/b.txt"
  USER=me VOXT_E2E_TOKEN=fake-secret-token "$script" --check-evidence "$tmp/evidence" > /dev/null 2>&1
  check "evidence with '$planted' fails" $? 1
done
exit $fail
