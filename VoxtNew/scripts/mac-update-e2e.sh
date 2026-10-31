#!/usr/bin/env bash
# The update test (docs/release.md §6): two releases of VoxtNew, 0.1.0
# and 0.1.1, signed ad hoc, with a test key that signs their updates,
# built with the release configuration (release/mygo.release.json, its
# url pointing at the test's feed), checked as the release workflow
# publishes them (scripts/actnow-publish.sh --dry-run, the site's own
# script) and published as actnow.dev's back end does (update-server
# -publish, into a directory that stands for the R2 bucket), on a local
# actnow.dev: its update feed /updates/voxtnew/ served over HTTPS on the
# loopback interface (scripts/update-server -site). The installed 0.1.0 is then updated by
# the real updater, driven without a window (--update-check, -tags
# updatecheck: update_check.go):
#
#   A. the delta: found, downloaded alone, installed; the bundle on disk is
#      0.1.1 file for file (code signature included), its helper the new
#      one, and the supervisor of the old app starts no helper;
#   B. the archive, when the feed has no delta;
#   C. the archive after a delta that fails (an installed file changed);
#   D. 0.1.1 is up to date.
#
# Its own bundle ID (dev.hehehai.voxtnew.updtest) and directory, on a Mac
# that may be busy with other work: no window, no app activated, no key,
# no clipboard, no helper started, no keychain; niced, two jobs, and it
# stops when less than VOXTNEW_MIN_FREE percent of memory (40) is free.
# The test key, the builds and the feed are removed at the end
# (KEEP=1 keeps them in build/update-e2e).
#
#   MAC_DIR=code/VoxtNew-upd scripts/mac.sh scripts/mac-update-e2e.sh
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run it on a Mac: MAC_DIR=code/VoxtNew-upd scripts/mac.sh scripts/mac-update-e2e.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ "$ROOT" == "$HOME/code/VoxtNew" ]]; then
  echo "not in the app's own checkout: use MAC_DIR=code/VoxtNew-upd" >&2
  exit 1
fi
export GOMAXPROCS=2 GOFLAGS=-p=2 SWIFT_BUILD_JOBS=2
unset VOXTNEW_HELPER
ID=dev.hehehai.voxtnew.updtest
V0=0.1.0 V1=0.1.1
WORK="$ROOT/build/update-e2e"
FAILS=0

step() {
  local free
  free="$(memory_pressure -Q | sed -n 's/.*percentage: \([0-9]*\)%.*/\1/p')"
  echo "== $1 (free memory ${free}%)"
  if (( ${free:-0} < ${VOXTNEW_MIN_FREE:-40} )); then
    echo "less than ${VOXTNEW_MIN_FREE:-40}% of memory is free: stopping" >&2
    exit 3
  fi
}
check() { # check "what" command...: one line, ok or FAIL
  local what="$1"
  shift
  if "$@"; then
    echo "  ok   $what"
  else
    echo "  FAIL $what"
    FAILS=$((FAILS + 1))
  fi
}

server_pid=""
cleanup() {
  local rc=$?
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  for f in mygo.json helper/Support/Info.plist CHANGELOG.md; do
    [[ -f "$WORK/orig/$f" ]] && cp "$WORK/orig/$f" "$f"
  done
  rm -rf resources/darwin-arm64
  if [[ "${KEEP:-0}" != 1 ]]; then
    rm -rf "$WORK"
  fi
  exit "$rc"
}
trap cleanup EXIT

rm -rf "$WORK"
mkdir -p "$WORK"/{bucket,keys,orig/helper/Support}
cp mygo.json CHANGELOG.md "$WORK/orig/"
cp helper/Support/Info.plist "$WORK/orig/helper/Support/"

step "Go modules and the tools (before SSL_CERT_FILE, which trusts the test certificate only)"
nice -n 10 go mod download
nice -n 10 go build -o "$WORK/update-server" ./scripts/update-server
nice -n 10 go build -tags updatecheck -o /dev/null .

step "a test key and the feed"
go tool mygo keygen -o "$WORK/keys" >/dev/null
chmod 600 "$WORK/keys/mygo-update.key"
"$WORK/update-server" -site -dir "$WORK/bucket" -cert "$WORK/cert.pem" -url "$WORK/url.txt" >"$WORK/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 50); do [[ -s "$WORK/url.txt" ]] && break; sleep 0.1; done
SITE="$(cat "$WORK/url.txt")"
URL="$SITE/updates/voxtnew"
# The release's updates (url, s3, prereleases) with the test key and feed.
jq --arg id "$ID" --arg key "$(cat "$WORK/keys/mygo-update.pub")" --arg url "$URL" \
  '.identifier = $id | .updates.publicKey = $key | .updates.url = $url' release/mygo.release.json >"$WORK/overlay.json"
echo "  feed $URL"

# The release notes of both versions, as docs/release.md has them: English,
# then Chinese.
{
  sed -n '1,4p' "$WORK/orig/CHANGELOG.md"
  printf '## [%s] - 2026-10-16\n\n### Fixed\n\n- The update test.\n\n### 修复\n\n- 更新测试。\n\n' "$V1"
  printf '## [%s] - 2026-10-16\n\n### Added\n\n- The first test version.\n\n### 新增\n\n- 第一个测试版本。\n\n' "$V0"
  sed -n '5,$p' "$WORK/orig/CHANGELOG.md"
} >CHANGELOG.md

# build VERSION: the helper and the app of VERSION into $WORK/out-VERSION.
build() {
  local v="$1"
  cp "$WORK/orig/mygo.json" mygo.json
  scripts/release-config.sh --overlay "$WORK/overlay.json" >/dev/null
  # Without the model runtimes (the updater does not care what the
  # bundle holds): no voxtnew-llm to give its entitlements to.
  jq --arg v "$v" '.version = $v | del(.macos.helperEntitlements["llm/voxtnew-llm"])' mygo.json >"$WORK/mygo.json" &&
    cat "$WORK/mygo.json" >mygo.json
  /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $v" helper/Support/Info.plist
  nice -n 10 scripts/build-helper.sh >/dev/null
  # The CLI fetches the published version for its delta: the test
  # certificate, and no module downloads past this point.
  SSL_CERT_FILE="$WORK/cert.pem" GOPROXY=off GOFLAGS="$GOFLAGS -tags=updatecheck" \
    MYGO_UPDATER_PRIVATE_KEY="$(cat "$WORK/keys/mygo-update.key")" \
    nice -n 10 go tool mygo build -platform darwin/arm64 -skip-dmg -o "$WORK/out-$v" >"$WORK/build-$v.log" 2>&1 ||
    { tail -30 "$WORK/build-$v.log" >&2; exit 1; }
  grep -iE "delta|update" "$WORK/build-$v.log" | sed 's/^/  build: /' || true
}

# publish VERSION: checked by the site's script, then into the bucket as
# the site publishes it (no disk image: -skip-dmg).
publish() {
  local out="$WORK/out-$1/darwin-arm64"
  bash scripts/actnow-publish.sh --app voxtnew --dir "$out" --changelog CHANGELOG.md --min-macos 15.0 \
    --url "$SITE" --dry-run >"$WORK/publish-$1.json" 2>"$WORK/publish-$1.log" ||
    { cat "$WORK/publish-$1.log" >&2; exit 1; }
  # shellcheck disable=SC2016 # jq's variables
  check "actnow-publish.sh --dry-run takes $1 ($(jq -r '.channel' "$WORK/publish-$1.json"), $(jq -r '[.artifacts[].filename] | join(" ")' "$WORK/publish-$1.json"))" \
    jq -e --arg v "$1" '.version == $v and (.notes | length > 0) and (.notesZh | length > 0)' "$WORK/publish-$1.json"
  "$WORK/update-server" -publish "$out" -dir "$WORK/bucket" -app voxtnew | sed 's/^/  publish: /'
}

# install: 0.1.0-test from its archive, as a user has it.
install_v0() {
  rm -rf "$WORK/inst"
  mkdir -p "$WORK/inst"
  tar -xzf "$WORK/out-$V0/darwin-arm64/"*.tar.gz -C "$WORK/inst"
}

# run REPORT: the installed app's update check.
run_check() {
  SSL_CERT_FILE="$WORK/cert.pem" nice -n 10 "$WORK/inst/VoxtNew.app/Contents/MacOS/VoxtNew" --update-check "$1"
}

step "build $V0"
build "$V0"
publish "$V0"
step "build $V1 (with its delta from $V0)"
build "$V1"
NEW="$WORK/out-$V1/darwin-arm64/VoxtNew.app"
OLD="$WORK/out-$V0/darwin-arm64/VoxtNew.app"
check "$V1 has a delta from $V0" bash -c "compgen -G '$WORK/out-$V1/darwin-arm64/*-to-$V1-*.delta' >/dev/null"
check "the helpers of the two versions differ" \
  bash -c "! cmp -s '$OLD/Contents/Resources/helper/voxtnew-helper' '$NEW/Contents/Resources/helper/voxtnew-helper'"
check "the app ships LICENSE, NOTICE and LICENSES/ (mygo.json resources)" \
  bash -c "cmp -s LICENSE '$NEW/Contents/Resources/LICENSE' && cmp -s NOTICE '$NEW/Contents/Resources/NOTICE' && diff -r LICENSES '$NEW/Contents/Resources/LICENSES'"
check "the build passes release-verify.sh (ad hoc)" scripts/release-verify.sh "$WORK/out-$V1/darwin-arm64" ad-hoc --no-dmg --no-runtimes
publish "$V1"
check "the release is the beta channel's too" \
  test "$(jq -r .version "$WORK/bucket/voxtnew/beta/update-darwin-arm64.json")" = "$V1"
MANIFEST="$WORK/bucket/voxtnew/update-darwin-arm64.json"
cp "$MANIFEST" "$WORK/manifest-with-delta.json"
delta_size="$(jq -r --arg v "$V0" '.deltas[] | select(.from == $v) | .size' "$MANIFEST")"
archive_size="$(jq -r .size "$MANIFEST")"
echo "  delta $delta_size bytes, archive $archive_size bytes"

# verify REPORT TOTALS: the installed app is V1, exactly the published
# bundle; TOTALS the downloads the report must show.
verify() {
  local r="$1" totals="$2" app="$WORK/inst/VoxtNew.app"
  check "it found $V1" test "$(jq -r .found "$r")" = "$V1"
  check "it was enabled and ran $V0" test "$(jq -r '"\(.enabled) \(.running)"' "$r")" = "true $V0"
  check "the notes are $V1's, English and Chinese" bash -c "jq -r .notes '$r' | grep -q 'The update test' && jq -r .notes '$r' | grep -q '更新测试'"
  check "installed without an error" test -z "$(jq -r .installErr "$r")"
  check "downloads $totals" test "$(jq -c .totals "$r")" = "$totals"
  check "the bundle on disk says $V1" test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")" = "$V1"
  check "the bundle is the published one, file for file" diff -r "$app" "$NEW"
  check "its signature verifies (deep, strict)" codesign --verify --deep --strict "$app"
  check "the same code directory hash" test "$(codesign -dv "$app" 2>&1 | grep CDHash=)" = "$(codesign -dv "$NEW" 2>&1 | grep CDHash=)"
  check "the helper is the new one" cmp -s "$app/Contents/Resources/helper/voxtnew-helper" "$NEW/Contents/Resources/helper/voxtnew-helper"
  check "the helper's signature verifies" codesign --verify --strict "$app/Contents/Resources/helper/voxtnew-helper"
  check "the old app sees the update" test "$(jq -r '"\(.diskVersion) \(.stale)"' "$r")" = "$V1 true"
  check "its supervisor starts no helper (updated)" test "$(jq -r '"\(.helperState) \(.helperReason) \(.helperPID)"' "$r")" = "failed updated 0"
  check "nothing left next to the app" test -z "$(find "$WORK/inst" -maxdepth 1 -name '.VoxtNew.app.*' -print)"
}

step "A: the delta"
install_v0
: >"$WORK/server.log"
run_check "$WORK/report-delta.json"
verify "$WORK/report-delta.json" "[$delta_size]"
check "the feed served the delta, not the archive" \
  bash -c "grep -q '\.delta' '$WORK/server.log' && ! grep -q '\.tar\.gz' '$WORK/server.log'"

step "B: the archive (a feed without the delta)"
jq 'del(.deltas)' "$WORK/manifest-with-delta.json" >"$MANIFEST"
install_v0
run_check "$WORK/report-archive.json"
verify "$WORK/report-archive.json" "[$archive_size]"

step "C: the archive after a delta that fails (the installed app was changed)"
cp "$WORK/manifest-with-delta.json" "$MANIFEST"
install_v0
printf 'x' >>"$WORK/inst/VoxtNew.app/Contents/Resources/icon.png"
run_check "$WORK/report-fallback.json"
verify "$WORK/report-fallback.json" "[$delta_size,$archive_size]"

step "D: $V1 is up to date"
run_check "$WORK/report-current.json"
check "nothing found" test "$(jq -r '"\(.running) [\(.found)] [\(.checkErr)] \(.stale)"' "$WORK/report-current.json")" = "$V1 [] [] false"

if pgrep -f "$WORK/inst/VoxtNew.app/Contents/Resources/helper/voxtnew-helper" >/dev/null; then
  echo "  FAIL a helper is running" && FAILS=$((FAILS + 1))
fi
if (( FAILS > 0 )); then
  echo "== $FAILS checks failed" >&2
  exit 1
fi
echo "== all checks passed; the test key, the builds and the feed are removed"
