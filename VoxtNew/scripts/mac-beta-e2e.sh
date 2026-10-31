#!/usr/bin/env bash
# The beta channel's test (docs/release.md §6): five versions of VoxtNew,
# signed ad hoc with a test key, checked one after another as the release
# workflow publishes them (scripts/actnow-publish.sh --dry-run, the site's
# own script) and published as actnow.dev's back end does (update-server
# -publish, into a directory that stands for the R2 bucket: a release
# under voxtnew/, a beta under voxtnew/beta/, and the beta channel's
# manifest always the newer of both), on a local actnow.dev: its update feed
# /updates/voxtnew[/beta]/ served over HTTPS on the loopback interface
# (scripts/update-server -site, which serves the update files only):
#
#   0.1.0-beta.1, 0.1.0-beta.2   betas before any release
#   0.1.0                        the first release
#   0.2.0-beta.1                 a beta of the next version
#   0.1.1                        a fix released after that beta
#
# Each version is built as the release workflow builds it (the updates of
# release/mygo.release.json: url, s3 and prereleases, the url the test's),
# so its deltas come from what is published at that point. After each
# publication installed versions check for updates with Beta updates on,
# off or as by default (--update-check, -tags updatecheck:
# update_check.go), and install what they find with the real updater: a
# beta takes the next beta and the release, a release ignores betas unless
# asked, the beta channel offers the newer of both, nothing goes back, a
# delta when there is one and the archive when it fails, and with the beta
# channel's manifest broken betas fall back to the releases'.
#
# Its own bundle ID (dev.hehehai.voxtnew.betatest) and directory, on a Mac
# that may be busy with other work: no window, no app activated, no key, no
# clipboard, no helper started, no keychain; niced, two jobs, and it stops
# when less than VOXTNEW_MIN_FREE percent of memory (40) is free. The test
# key, the builds and the feed are removed at the end (KEEP=1 keeps them in
# build/beta-e2e).
#
#   MAC_DIR=code/VoxtNew-site scripts/mac.sh scripts/mac-beta-e2e.sh
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run it on a Mac: MAC_DIR=code/VoxtNew-site scripts/mac.sh scripts/mac-beta-e2e.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ "$ROOT" == "$HOME/code/VoxtNew" ]]; then
  echo "not in the app's own checkout: use MAC_DIR=code/VoxtNew-site" >&2
  exit 1
fi
export GOMAXPROCS=2 GOFLAGS=-p=2 SWIFT_BUILD_JOBS=2
unset VOXTNEW_HELPER
ID=dev.hehehai.voxtnew.betatest
B1=0.1.0-beta.1 B2=0.1.0-beta.2 R1=0.1.0 B3=0.2.0-beta.1 R2=0.1.1
WORK="$ROOT/build/beta-e2e"
BUCKET="$WORK/bucket"
FEED="$BUCKET/voxtnew"
FAILS=0 RUNS=0

step() {
  local free
  free="$(memory_pressure -Q | sed -n 's/.*percentage: \([0-9]*\)%.*/\1/p')"
  echo "== $1 (free memory ${free}%)"
  if ((${free:-0} < ${VOXTNEW_MIN_FREE:-40})); then
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
mkdir -p "$WORK"/{keys,orig/helper/Support} "$BUCKET"
cp mygo.json CHANGELOG.md "$WORK/orig/"
cp helper/Support/Info.plist "$WORK/orig/helper/Support/"

step "Go modules and the tools (before SSL_CERT_FILE, which trusts the test certificate only)"
nice -n 10 go mod download
nice -n 10 go build -o "$WORK/update-server" ./scripts/update-server
nice -n 10 go build -tags updatecheck -o /dev/null .

step "a test key and the feed"
go tool mygo keygen -o "$WORK/keys" >/dev/null
chmod 600 "$WORK/keys/mygo-update.key"
"$WORK/update-server" -site -dir "$BUCKET" -cert "$WORK/cert.pem" -url "$WORK/url.txt" >"$WORK/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 50); do [[ -s "$WORK/url.txt" ]] && break; sleep 0.1; done
SITE="$(cat "$WORK/url.txt")"
URL="$SITE/updates/voxtnew"
# The release's updates (url, s3, prereleases) with the test key and feed.
jq --arg id "$ID" --arg key "$(cat "$WORK/keys/mygo-update.pub")" --arg url "$URL" \
  '.identifier = $id | .updates.publicKey = $key | .updates.url = $url' release/mygo.release.json >"$WORK/overlay.json"
jq -e '.updates.prereleases == true' "$WORK/overlay.json" >/dev/null
echo "  feed $URL"

# The notes of every version, English then Chinese.
{
  sed -n '1,4p' "$WORK/orig/CHANGELOG.md"
  for v in "$R2" "$B3" "$R1" "$B2" "$B1"; do
    printf '## [%s] - 2026-10-16\n\n### Changed\n\n- Test version %s.\n\n### 变更\n\n- 测试版本 %s。\n\n' "$v" "$v" "$v"
  done
} >CHANGELOG.md

# local COMMAND: COMMAND trusting the test feed's certificate.
local_feed() { SSL_CERT_FILE="$WORK/cert.pem" "$@"; }

# build VERSION: the helper and the app of VERSION into $WORK/out-VERSION,
# its deltas from what is published.
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
  GOPROXY=off GOFLAGS="$GOFLAGS -tags=updatecheck" MYGO_UPDATER_PRIVATE_KEY="$(cat "$WORK/keys/mygo-update.key")" \
    local_feed nice -n 10 go tool mygo build -platform darwin/arm64 -skip-dmg -o "$WORK/out-$v" >"$WORK/build-$v.log" 2>&1 ||
    { tail -30 "$WORK/build-$v.log" >&2; exit 1; }
  grep -iE "delta" "$WORK/build-$v.log" | sed 's/^/  build: /' || true
}

# deltas VERSION: the versions VERSION's manifest has deltas from.
deltas() { jq -r '[.deltas[]?.from] | join(" ")' "$WORK/out-$1/darwin-arm64/update-darwin-arm64.json"; }

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
  "$WORK/update-server" -publish "$out" -dir "$BUCKET" -app voxtnew | sed 's/^/  publish: /'
}

# channels: the versions of the releases' and the beta channel's manifests.
channels() {
  local r b
  r="$(jq -r .version "$FEED/update-darwin-arm64.json" 2>/dev/null || echo none)"
  b="$(jq -r .version "$FEED/beta/update-darwin-arm64.json" 2>/dev/null || echo none)"
  echo "$r $b"
}

# expect NAME FROM BETA WANT HOW: installs FROM from its archive, checks
# for updates with Beta updates BETA (on, off, default), which must find
# WANT ("" for none) and install it by HOW: delta (its delta from FROM
# only), fallback (the delta, then the archive: the installed app was
# changed) or archive; NOTE is what a check that finds nothing says
# (notPublished, an error's words, or "" for nothing).
expect() {
  local name="$1" from="$2" beta="$3" want="$4" how="${5:-}" note="${6:-}"
  local r="$WORK/report-$name.json" app="$WORK/inst/VoxtNew.app" args=()
  RUNS=$((RUNS + 1))
  echo "-- $name: $from, beta $beta → ${want:-nothing}${how:+ ($how)}"
  rm -rf "$WORK/inst" && mkdir -p "$WORK/inst"
  tar -xzf "$WORK/out-$from/darwin-arm64/"*.tar.gz -C "$WORK/inst"
  [[ "$how" == fallback ]] && printf 'x' >>"$app/Contents/Resources/icon.png"
  [[ "$beta" != default ]] && args=(--update-beta "$beta")
  local before
  before="$(wc -l <"$WORK/server.log")"
  local_feed nice -n 10 "$app/Contents/MacOS/VoxtNew" --update-check "$r" ${args[@]+"${args[@]}"}
  local served
  served="$(tail -n +"$((before + 1))" "$WORK/server.log")"
  local on=false
  [[ "$beta" == on || "$beta" == default && "$from" == *-* ]] && on=true
  check "running $from, betas $on" test "$(jq -r '"\(.running) \(.beta)"' "$r")" = "$from $on"
  check "found ${want:-nothing}" test "$(jq -r .found "$r")" = "$want"
  if [[ -z "$want" ]]; then
    case "$note" in
      "") check "up to date, no error" test -z "$(jq -r .checkErr "$r")" ;;
      notPublished) check "no release on the channel (ErrNotPublished)" test "$(jq -r .notPublished "$r")" = true ;;
      *) check "the error says: $note" bash -c "jq -r .checkErr '$r' | grep -qF '$note'" ;;
    esac
    check "the bundle is still $from" test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")" = "$from"
    return
  fi
  local manifest="$WORK/out-$want/darwin-arm64/update-darwin-arm64.json" totals
  local dsize asize
  dsize="$(jq -r --arg v "$from" '.deltas[]? | select(.from == $v) | .size' "$manifest")"
  asize="$(jq -r .size "$manifest")"
  case "$how" in
    delta) totals="[$dsize]" ;;
    fallback) totals="[$dsize,$asize]" ;;
    archive) totals="[$asize]" ;;
  esac
  check "installed without an error" test -z "$(jq -r '.checkErr + .installErr' "$r")"
  check "downloaded $how $totals" test "$(jq -c .totals "$r")" = "$totals"
  check "the notes are $want's, English and Chinese" bash -c "jq -r .notes '$r' | grep -qF 'Test version $want.' && jq -r .notes '$r' | grep -qF '测试版本 ${want}。'"
  check "the bundle is $want's, file for file" diff -r "$app" "$WORK/out-$want/darwin-arm64/VoxtNew.app"
  check "its signature verifies (deep, strict)" codesign --verify --deep --strict "$app"
  check "CFBundleVersion is $want's numbers" test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$app/Contents/Info.plist")" = "${want%%-*}"
  check "the old app starts no helper (updated)" test "$(jq -r '"\(.diskVersion) \(.helperState) \(.helperReason) \(.helperPID)"' "$r")" = "$want failed updated 0"
  if [[ "$how" == delta ]]; then
    check "the feed served the delta, not the archive" bash -c "grep -q '\.delta' <<<\"\$1\" && ! grep -q '\.tar\.gz' <<<\"\$1\"" _ "$served"
  fi
  check "nothing left next to the app" test -z "$(find "$WORK/inst" -maxdepth 1 -name '.VoxtNew.app.*' -print)"
}

step "build $B1, the first beta: nothing published yet"
build "$B1"
check "$B1 has no delta" test -z "$(deltas "$B1")"
publish "$B1"
check "only the beta channel has $B1" test "$(channels)" = "none $B1"
step "build $B2 (only betas published: its delta from $B1)"
build "$B2"
check "$B2 has a delta from $B1" test "$(deltas "$B2")" = "$B1"
publish "$B2"
check "the build passes release-verify.sh (ad hoc)" scripts/release-verify.sh "$WORK/out-$B2/darwin-arm64" ad-hoc --no-dmg --no-runtimes

step "betas only"
expect a "$B1" default "$B2" delta
expect b "$B1" off "" "" notPublished

step "build $R1, the first release (deltas from the betas)"
build "$R1"
check "$R1 has deltas from $B2 and $B1" test "$(deltas "$R1")" = "$B2 $B1"
publish "$R1"
check "both channels have $R1" test "$(channels)" = "$R1 $R1"
expect c "$B2" default "$R1" delta
expect d "$B2" off "$R1" delta
expect e "$B1" on "$R1" delta
expect n "$B2" on "$R1" fallback

step "build $B3, a beta of the next version (deltas from both channels)"
build "$B3"
check "$B3 has deltas from $R1, $B2 and $B1" test "$(deltas "$B3")" = "$R1 $B2 $B1"
publish "$B3"
check "the beta channel has $B3" test "$(channels)" = "$R1 $B3"
expect f "$R1" default ""
expect g "$R1" on "$B3" delta
expect h "$B2" default "$B3" delta
expect i "$B2" off "$R1" delta

step "build $R2, a fix released after $B3"
build "$R2"
check "$R2 has deltas from $R1, $B2 and $B1" test "$(deltas "$R2")" = "$R1 $B2 $B1"
publish "$R2"
check "the beta channel keeps $B3, newer than $R2" test "$(channels)" = "$R2 $B3"
expect j "$B3" off ""
expect k "$B3" default ""
expect l "$R1" default "$R2" delta
expect m "$R1" on "$B3" delta

step "the beta channel's manifest broken: betas fall back to the releases'"
cp "$FEED/beta/update-darwin-arm64.json" "$WORK/beta-manifest.json"
echo '{"version":' >"$FEED/beta/update-darwin-arm64.json"
expect o "$R1" on "$R2" delta
expect p "$B3" on "" "" "checking for updates"
cp "$WORK/beta-manifest.json" "$FEED/beta/update-darwin-arm64.json"

step "the feed"
echo "  $(wc -l <"$WORK/server.log" | tr -d ' ') requests, $RUNS update checks"
check "nothing but /updates/voxtnew/ was asked" bash -c "! grep -E ' (GET|HEAD) /' '$WORK/server.log' | grep -v ' /updates/voxtnew/'"
check "no disk image was uploaded (-no-dmg) nor asked" bash -c "! find '$BUCKET' -name '*.dmg' | grep -q . && ! grep -q '\.dmg' '$WORK/server.log'"
if pgrep -f "$WORK/inst/VoxtNew.app/Contents/Resources/helper/voxtnew-helper" >/dev/null; then
  echo "  FAIL a helper is running" && FAILS=$((FAILS + 1))
fi
if ((FAILS > 0)); then
  echo "== $FAILS checks failed" >&2
  exit 1
fi
echo "== all checks passed; the test key, the builds and the feed are removed"
