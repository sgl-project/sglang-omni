#!/usr/bin/env bash
# The update test against the published feed (docs/release.md): a build
# of Voxt older than every release, with the release's update settings
# (the public key and the GitHub feed of scripts/release-appcast.sh), is
# updated by the real updater from the real GitHub releases, driven
# without a window (--update-check, -tags updatecheck: update_check.go).
# Once for the stable channel and once for the beta channel: the app
# finds the version the channel offers, downloads it, checks its
# signature against the release key and installs it; the bundle on disk
# is then that version and its code signature verifies.
#
# It needs no secret: the old build is signed ad hoc and makes no update
# files of its own (MyGo signs none without the private key). Its own
# bundle ID and directory, like scripts/mac-update-e2e.sh: no window, no
# helper started, no settings read. KEEP=1 keeps build/update-feed-e2e.
#
# With --from TAG it builds nothing: the published beta TAG itself, from
# its release's archive, updates to the newest beta (the release workflow
# after each beta: the app users have, not a test build). Betas are built
# with -tags updatecheck for this; a beta without it, or with another
# feed, is skipped. Only in CI: it runs the app under its real bundle ID.
#
#   scripts/mac-update-feed-e2e.sh [--from TAG] [OWNER/REPO]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FROM=""
if [[ "${1:-}" == --from ]]; then
  FROM="${2:?--from TAG}"
  shift 2
fi
REPO="${1:-sgl-project/sglang-omni}"
ID=dev.hehehai.voxtnew.updtest
V0=2.0.0
WORK="$ROOT/build/update-feed-e2e"
KEY="PPX1glcG71+Xyk/V2veU1p0QeyX1RKRFPeAG2++lUEE="
FAILS=0
unset VOXTNEW_HELPER

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
cleanup() {
  local rc=$?
  if [[ -z "$FROM" ]]; then
    for f in mygo.json helper/Support/Info.plist; do
      [[ -f "$WORK/orig/$f" ]] && cp "$WORK/orig/$f" "$f"
    done
    rm -rf resources/darwin-arm64
  fi
  if [[ "${KEEP:-0}" != 1 ]]; then
    rm -rf "$WORK"
  fi
  exit "$rc"
}
trap cleanup EXIT

rm -rf "$WORK"
if [[ -n "$FROM" ]]; then
  [[ -n "${CI:-}" ]] || { echo "--from runs the published app under its real bundle ID: only in CI" >&2; exit 1; }
  V0="${FROM#voxt-v}"
  echo "== the published $V0"
  mkdir -p "$WORK/out/darwin-arm64"
  gh release download "$FROM" --repo "$REPO" --pattern '*-darwin-arm64.tar.gz' --dir "$WORK"
  tar -xzf "$WORK/"*-darwin-arm64.tar.gz -C "$WORK/out/darwin-arm64"
  bin="$WORK/out/darwin-arm64/Voxt.app/Contents/MacOS/Voxt"
  if ! grep -aqF -- "--update-beta" "$bin" || ! grep -aqF "$(scripts/release-appcast.sh feed-url beta)" "$bin"; then
    echo "== $FROM has no update check or another feed: skipped"
    exit 0
  fi
else
mkdir -p "$WORK/orig/helper/Support"
cp mygo.json "$WORK/orig/"
cp helper/Support/Info.plist "$WORK/orig/helper/Support/"

echo "== build $V0 with the release's feed and key"
grep -q "\"$KEY\"" ../.github/workflows/voxt-release.yaml ||
  { echo "the release workflow has another public key than this test" >&2; exit 1; }
jq -n --arg id "$ID" --arg key "$KEY" --arg url "$(scripts/release-appcast.sh feed-base)" \
  '{identifier: $id, updates: {url: $url, publicKey: $key, prereleases: true}}' >"$WORK/overlay.json"
scripts/release-config.sh --overlay "$WORK/overlay.json" >/dev/null
jq --arg v "$V0" '.version = $v' mygo.json >"$WORK/mygo.json" && cat "$WORK/mygo.json" >mygo.json
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $V0" helper/Support/Info.plist
scripts/build-helper.sh >/dev/null
GOFLAGS="${GOFLAGS:-} -tags=updatecheck" MYGO_UPDATER_PRIVATE_KEY="" \
  go tool mygo build -platform darwin/arm64 -skip-dmg -o "$WORK/out" >"$WORK/build.log" 2>&1 ||
  { tail -30 "$WORK/build.log" >&2; exit 1; }
fi

curl -fsSL "https://api.github.com/repos/$REPO/releases?per_page=100" >"$WORK/releases.json"
# offered CHANNEL: the version the channel's feed must offer.
offered() {
  local v
  for v in $(jq -r '.[] | select(.draft | not) | .tag_name | select(startswith("voxt-v")) | ltrimstr("voxt-v")' "$WORK/releases.json"); do
    if scripts/release-appcast.sh feed-offers "$1" "$v" <"$WORK/releases.json" >/dev/null 2>&1; then
      echo "$v"
      return
    fi
  done
}

# update CHANNEL BETA: install 2.0.0, run its update check with Beta
# updates BETA (on or off), and check what it did.
update() {
  local channel="$1" want app="$WORK/inst-$1/Voxt.app" r="$WORK/report-$1.json"
  want="$(offered "$channel")"
  echo "== $channel: $V0 to ${want:-nothing offered}"
  mkdir -p "$WORK/inst-$channel"
  cp -R "$WORK/out/darwin-arm64/Voxt.app" "$WORK/inst-$channel/"
  "$app/Contents/MacOS/Voxt" --update-check "$r" --update-beta "$2"
  check "the feed offers a release" test -n "$want"
  check "it was enabled and ran $V0" test "$(jq -r '"\(.enabled) \(.running)"' "$r")" = "true $V0"
  check "it found $want" test "$(jq -r '"\(.found) [\(.checkErr)]"' "$r")" = "$want []"
  check "it has release notes" test -n "$(jq -r .notes "$r")"
  check "installed without an error" test -z "$(jq -r .installErr "$r")"
  check "the bundle on disk says $want" test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")" = "$want"
  check "the old app sees the update" test "$(jq -r '"\(.diskVersion) \(.stale)"' "$r")" = "$want true"
  check "its signature verifies (deep, strict)" codesign --verify --deep --strict "$app"
  check "nothing left next to the app" test -z "$(find "$WORK/inst-$channel" -maxdepth 1 -name '.Voxt.app.*' -print)"
}
[[ -n "$FROM" ]] || update stable off
update beta on

if (( FAILS > 0 )); then
  cat "$WORK"/report-*.json >&2 || true
  echo "== $FAILS checks failed" >&2
  exit 1
fi
echo "== all checks passed"
