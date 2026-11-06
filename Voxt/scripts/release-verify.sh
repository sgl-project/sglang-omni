#!/usr/bin/env bash
# Checks a build of VoxtNew (docs/release.md): its signatures, the
# entitlements of the app and of the helper, the model runtimes inside
# (every Mach-O file and framework signed, the Omni server and voxtnew-llm
# starting from the bundle) and, for a signed build, the hardened runtime
# and the timestamps of all its code; for a notarized Developer ID build
# also Gatekeeper's verdict, the stapled tickets and the update files.
# scripts/release.sh runs it after `go tool mygo build`.
#
#   scripts/release-verify.sh build/darwin-arm64 developer-id  # signed, notarized
#   scripts/release-verify.sh build/darwin-arm64 signed        # signed, not notarized yet
#   scripts/release-verify.sh build/darwin-arm64 ad-hoc
#   scripts/release-verify.sh DIR ad-hoc --no-dmg   # a build with -skip-dmg
#   scripts/release-verify.sh DIR ad-hoc --no-runtimes   # the update tests' builds
#
# EXPECT_TEAM, when set, is the Team ID a signed build must be signed by.
set -euo pipefail

dir="${1:?the build directory, e.g. build/darwin-arm64}"
mode="${2:?developer-id or ad-hoc}"
want_dmg=1 want_runtimes=1
for a in "${@:3}"; do
  case "$a" in
    --no-dmg) want_dmg=0 ;;
    --no-runtimes) want_runtimes=0 ;;
    *) echo "release-verify.sh: unknown argument $a" >&2; exit 2 ;;
  esac
done
case "$mode" in
  developer-id | signed | ad-hoc) ;;
  *) echo "release-verify.sh: the mode is developer-id, signed or ad-hoc, not $mode" >&2; exit 2 ;;
esac
app="$dir/VoxtNew.app"
res="$app/Contents/Resources"
helper="$app/Contents/Resources/helper/voxtnew-helper"
fails=0

check() { # check "what" command...
  local what="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "ok   $what"
  else
    echo "FAIL $what"
    "$@" 2>&1 | sed 's/^/     /' | tail -20 || true
    fails=$((fails + 1))
  fi
}
# details CODE: what codesign says about CODE's signature.
details() { codesign -dv --verbose=4 "$1" 2>&1; }
# (Read whole before grep -q, which would cut codesign off: pipefail.)
has() { # has CODE PATTERN: codesign's details of CODE match PATTERN
  local d
  d="$(details "$1")"
  grep -qE "$2" <<<"$d"
}
entitled() { # entitled CODE KEY: CODE is signed with the entitlement KEY
  local e
  e="$(codesign -d --entitlements - --xml "$1" 2>/dev/null)"
  grep -q "<key>$2</key><true/>" <<<"$e"
}

[[ -d "$app" ]] || { echo "no $app" >&2; exit 1; }
[[ -f "$helper" ]] || { echo "no $helper" >&2; exit 1; }
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")"
echo "VoxtNew $version ($mode) in $dir"

check "the app's signature (deep, strict)" codesign --verify --deep --strict --verbose=2 "$app"
check "the helper's signature (strict)" codesign --verify --strict --verbose=2 "$helper"
check "the helper is dev.hehehai.voxtnew.helper" has "$helper" '^Identifier=dev\.hehehai\.voxtnew\.helper$'
check "the app may use the microphone (audio-input)" entitled "$app" com.apple.security.device.audio-input
check "the helper may use the microphone (audio-input)" entitled "$helper" com.apple.security.device.audio-input
check "the helper is arm64" test "$(lipo -archs "$helper")" = arm64
dmg="$(compgen -G "$dir/*.dmg" | head -1 || true)"
((want_dmg)) && check "a disk image" test -n "$dmg"

# The code among the resources: the helper, the Omni server and MLX's
# libraries, voxtnew-llm and llama.framework. mygo signs them one by one
# (codesign --deep does not reach Resources), the frameworks after their
# contents.
code=("$app")
while IFS= read -r -d '' f; do
  if [[ "$(head -c 4 "$f" | xxd -p)" =~ ^(cffaedfe|cafebabe|feedfacf)$ ]]; then
    code+=("$f")
  fi
done < <(find "$res" -type f \( -perm -u+x -o -name '*.dylib' \) -not -path '*.framework/*' -print0)
while IFS= read -r -d '' f; do code+=("$f"); done < <(find "$res" -type d -name '*.framework' -print0)
echo "code inside: ${#code[@]} (app, executables, libraries, frameworks)"
if ((want_runtimes)); then
  server="$res/omni/bin/qwen3_asr_server"
  llm="$res/llm/voxtnew-llm"
  check "the Omni runtime is inside" test -x "$server" -a -f "$res/omni/bin/mlx.metallib"
  check "voxtnew-llm is inside" test -x "$llm" -a -f "$res/llm/mlx.metallib" -a -d "$res/llm/llama.framework"
  check "the Omni server starts from the app (its libraries load)" bash -c "'$server' --model-kind none 2>&1 | grep -q -- '--model-kind must be'"
  check "voxtnew-llm starts from the app (llm.hello)" bash -c "printf '%s\\n' '{\"id\":\"1\",\"method\":\"llm.hello\",\"params\":{\"protocol\":1,\"app\":\"release-verify\"}}' | '$llm' 2>/dev/null | grep -q '\"protocol\":1'"
fi
for c in "${code[@]:1}"; do
  check "${c#"$res"/}: signed (strict)" codesign --verify --strict "$c"
done

if [[ "$mode" != ad-hoc ]]; then
  team_app="$(details "$app" | sed -n 's/^TeamIdentifier=//p')"
  for c in "${code[@]}"; do
    name="${c#"$res"/}"
    name="${name##*/VoxtNew.app}"
    name="${name:-VoxtNew.app}"
    [[ "$mode" == developer-id ]] && check "$name: Developer ID Application" has "$c" '^Authority=Developer ID Application: '
    check "$name: hardened runtime" has "$c" '^CodeDirectory .*flags=0x[0-9a-f]*\(.*runtime.*\)'
    check "$name: secure timestamp" has "$c" '^Timestamp='
    check "$name: the app's team (${team_app:-none})" test "$(details "$c" | sed -n 's/^TeamIdentifier=//p')" = "$team_app"
  done
  [[ -n "${EXPECT_TEAM:-}" ]] && check "the team is $EXPECT_TEAM" test "$team_app" = "$EXPECT_TEAM"
  # Before notarization Gatekeeper rejects the app ("Unnotarized Developer
  # ID"); this says what it thinks.
  echo "Gatekeeper: $(spctl -a -vv -t exec "$app" 2>&1 | tr '\n' ' ')"
fi
if [[ "$mode" == developer-id ]]; then
  verdict="$(spctl -a -vv -t exec "$app" 2>&1 || true)"
  check "Gatekeeper accepts the app (notarized)" grep -q 'source=Notarized Developer ID' <<<"$verdict"
  check "the app's ticket is stapled" xcrun stapler validate "$app"
  [[ -n "$dmg" ]] && check "the disk image's ticket is stapled" xcrun stapler validate "$dmg"
  [[ -n "$dmg" ]] && check "Gatekeeper accepts the disk image" spctl -a -vv -t open --context context:primary-signature "$dmg"
  manifest="$dir/update-darwin-arm64.json"
  check "the update archive" bash -c "compgen -G '$dir/*-$version-darwin-arm64.tar.gz'"
  check "the update manifest says $version" bash -c "jq -e --arg v '$version' '.version == \$v and (.signature | length > 0)' '$manifest'"
elif [[ "$mode" == ad-hoc ]]; then
  check "signed ad hoc" has "$app" '^Signature=adhoc$'
fi
# CFBundleVersion is the version's numbers (MyGo patch 0007), as Apple wants.
build_version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$app/Contents/Info.plist")"
numbers() { [[ "$1" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; }
check "CFBundleVersion $build_version is numbers" numbers "$build_version"

if ((fails > 0)); then
  echo "$fails checks failed" >&2
  exit 1
fi
echo "all checks passed"
