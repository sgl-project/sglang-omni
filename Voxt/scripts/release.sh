#!/usr/bin/env bash
# The release build of VoxtNew, on an Apple silicon Mac (docs/release.md):
# the helper, the Omni runtime (speech) and voxtnew-llm (language models)
# are built and staged into resources/darwin-arm64/, then go tool mygo
# build makes build/darwin-arm64/VoxtNew.app and its disk image with all
# of them inside, and scripts/release-verify.sh checks the result.
#
#   scripts/release.sh                                  # ad hoc: a release-style test build
#   VOXTNEW_SIGN_IDENTITY="$(scripts/release-identity.sh 779M533DL5 | cut -f1)" \
#     scripts/release.sh [--notarize PROFILE] [--publish]
#   scripts/mac-gui.sh scripts/release.sh …              # the same over ssh (see below)
#
# Options:
#   --runtimes DIR       take the runtimes prebuilt from DIR/omni and DIR/llm
#                        (scripts/build-omni.sh --out DIR/omni, scripts/build-llm.sh
#                        --out DIR/llm; mac-air: ~/voxtnew-runtime) instead of
#                        building them (about 10 minutes and 3 GB of build trees)
#   --notarize PROFILE   notarize with notarytool's keychain profile PROFILE
#                        (xcrun notarytool store-credentials PROFILE …), then
#                        staple the app and the disk image; needs the identity
#   --keychain FILE      the keychain holding PROFILE (default: the login keychain)
#   --publish            publish on actnow.dev with scripts/actnow-publish.sh
#                        (ACTNOW_INGEST_TOKEN, from the environment only); needs
#                        a notarized Developer ID build with update files.
#                        Without it, a build with update files gets the
#                        script's --dry-run
#   --keep-runtimes      leave the staged runtimes in resources/darwin-arm64
#
# With VOXTNEW_SIGN_IDENTITY (a Developer ID Application identity of the
# team 779M533DL5, its SHA-1: scripts/release-identity.sh) the helper, the
# LLM and then every Mach-O file and framework inside the app are signed
# with it, with the hardened runtime and a secure timestamp (mygo build:
# signNestedCode), and the build takes the update feed of
# release/mygo.release.json (scripts/release-config.sh): actnow.dev's
# /updates/voxtnew, or /updates/voxtnew/beta for a version with a "-".
# The update files are signed with VoxtNew's update key:
# MYGO_UPDATER_PRIVATE_KEY (the key itself) or the file VOXTNEW_UPDATE_KEY
# (default ~/.config/voxtnew-release/update-keys/mygo-update.key), which
# must match the public key of release/mygo.release.json. Without an
# identity everything is signed ad hoc and the app has no update feed, as
# every development build.
#
# Over ssh, codesign cannot use a key of the login keychain
# (errSecInternalComponent): the script checks that first, and stops. Run
# it in the GUI session through scripts/mac-gui.sh (nothing to type), or in
# Terminal on the Mac (docs/release.md §1.1). mygo.json is put back as it
# was at the end, whatever happens.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != arm64 ]]; then
  echo "the release builds on an Apple silicon Mac: scripts/mac.sh scripts/release.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUNTIMES="" PROFILE="" KEYCHAIN="" PUBLISH=0 KEEP=0
while (($#)); do
  case "$1" in
    --runtimes) RUNTIMES="$(cd "${2:?--runtimes needs a directory}" && pwd)"; shift 2 ;;
    --notarize) PROFILE="${2:?--notarize needs a notarytool keychain profile}"; shift 2 ;;
    --keychain) KEYCHAIN="${2:?--keychain needs a keychain file}"; shift 2 ;;
    --publish) PUBLISH=1; shift ;;
    --keep-runtimes) KEEP=1; shift ;;
    -h | --help) sed -n '2,/^set -euo/{/^set -euo/d;p;}' "$0"; exit 0 ;;
    *) echo "release.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
done
IDENTITY="${VOXTNEW_SIGN_IDENTITY:-}"
if [[ -n "$PROFILE" && -z "$IDENTITY" ]]; then
  echo "notarization needs a Developer ID: set VOXTNEW_SIGN_IDENTITY" >&2
  exit 2
fi
command -v jq >/dev/null || { echo "release.sh needs jq" >&2; exit 1; }

# The update key: the key itself, or its file (never printed, never on a
# command line). It must be the pair of the public key installed apps check.
KEYFILE="${VOXTNEW_UPDATE_KEY:-$HOME/.config/voxtnew-release/update-keys/mygo-update.key}"
pub_of_key() { # the key's text on stdin → its public key (Ed25519: seed ‖ public)
  tr -d '[:space:]' | base64 --decode 2>/dev/null | tail -c 32 | base64
}
want_pub="$(jq -r .updates.publicKey release/mygo.release.json)"
HAVE_KEY=0 KEY_ARG=()
if [[ -n "$IDENTITY" ]]; then
  if [[ -n "${MYGO_UPDATER_PRIVATE_KEY:-}" ]]; then
    [[ "$(printf '%s' "$MYGO_UPDATER_PRIVATE_KEY" | pub_of_key)" == "$want_pub" ]] ||
      { echo "MYGO_UPDATER_PRIVATE_KEY is not the key of $want_pub (release/mygo.release.json)" >&2; exit 2; }
    HAVE_KEY=1
  elif [[ -f "$KEYFILE" ]]; then
    [[ "$(pub_of_key <"$KEYFILE")" == "$want_pub" ]] ||
      { echo "$KEYFILE is not the key of $want_pub (release/mygo.release.json)" >&2; exit 2; }
    HAVE_KEY=1 KEY_ARG=(--update-key "$KEYFILE")
  fi
fi
if ((PUBLISH)) && [[ -z "$PROFILE" || $HAVE_KEY == 0 || -z "${ACTNOW_INGEST_TOKEN:-}" ]]; then
  echo "--publish needs --notarize, the update key (MYGO_UPDATER_PRIVATE_KEY or $KEYFILE) and ACTNOW_INGEST_TOKEN" >&2
  exit 2
fi

# Can codesign use the identity here? Over ssh it cannot (the keychain is
# locked in that session): stop now rather than after the build.
if [[ -n "$IDENTITY" ]]; then
  probe="$(mktemp -d)"
  cp /usr/bin/true "$probe/probe"
  if ! err="$(codesign -f -s "$IDENTITY" --timestamp=none "$probe/probe" 2>&1)"; then
    rm -rf "$probe"
    echo "codesign cannot sign with $IDENTITY here: ${err##*: }" >&2
    if [[ "$err" == *errSecInternalComponent* ]]; then
      echo "Over ssh the login keychain is locked: run it in the GUI session, scripts/mac-gui.sh scripts/release.sh …," >&2
      echo "or in Terminal on the Mac (docs/release.md §1.1)." >&2
    fi
    exit 1
  fi
  rm -rf "$probe"
fi

# One version everywhere: the app, the helper, the LLM, the notes.
version="$(jq -r .version mygo.json)"
for plist in helper/Support/Info.plist llm/Support/Info.plist; do
  v="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$plist")"
  if [[ "$v" != "$version" ]]; then
    echo "$plist says $v, mygo.json $version: make them agree (docs/release.md §2.1)" >&2
    exit 1
  fi
done
notes="$(awk -v v="$version" '/^## /{f = index($0, "[" v "]") || $2 == v; next} f' CHANGELOG.md)"
if [[ -z "${notes//[[:space:]]/}" ]]; then
  if ((PUBLISH)); then
    echo "CHANGELOG.md has no section for $version" >&2
    exit 1
  else
    echo "note: CHANGELOG.md has no section for $version yet (the update window shows it)"
  fi
else
  :
fi
channel=stable feed="$(jq -r .updates.url release/mygo.release.json)"
if [[ "$version" == *-* ]]; then
  channel=beta feed="$feed/beta"
else
  :
fi
echo "== VoxtNew $version ($channel), signed ${IDENTITY:-ad hoc}${PROFILE:+, notarized with $PROFILE}"
[[ -n "$IDENTITY" ]] && echo "   update feed $feed$( ((HAVE_KEY)) && echo ", update files signed" || echo ", no update key: no update files")"

RES="$ROOT/resources/darwin-arm64"
saved="$(mktemp -d)"
cp mygo.json "$saved/mygo.json"
restore() {
  cp "$saved/mygo.json" "$ROOT/mygo.json"
  rm -rf "$saved"
  if ((KEEP == 0)); then
    rm -rf "$RES/omni" "$RES/llm"
  else
    :
  fi
}
trap restore EXIT

echo "== helper"
VOXTNEW_SIGN_IDENTITY="${IDENTITY:--}" scripts/build-helper.sh
if [[ -n "$RUNTIMES" ]]; then
  echo "== runtimes from $RUNTIMES"
  for part in omni llm; do
    [[ -d "$RUNTIMES/$part" ]] || { echo "no $RUNTIMES/$part" >&2; exit 1; }
    rm -rf "${RES:?}/$part"
    mkdir -p "$RES"
    # APFS clones: no copy on disk.
    cp -Rc "$RUNTIMES/$part" "$RES/$part"
  done
else
  echo "== Omni runtime"
  scripts/build-omni.sh --clean
  echo "== voxtnew-llm"
  VOXTNEW_SIGN_IDENTITY="${IDENTITY:--}" scripts/build-llm.sh
fi
[[ -x "$RES/omni/bin/qwen3_asr_server" ]] || { echo "no Omni runtime in $RES/omni" >&2; exit 1; }
[[ -x "$RES/llm/voxtnew-llm" ]] || { echo "no voxtnew-llm in $RES/llm" >&2; exit 1; }

echo "== configuration"
if [[ -n "$IDENTITY" ]]; then
  scripts/release-config.sh --sign "$IDENTITY" ${PROFILE:+--notary-profile "$PROFILE"} ${KEYCHAIN:+--notary-keychain "$KEYCHAIN"} \
    ${KEY_ARG[@]+"${KEY_ARG[@]}"}
  if ((HAVE_KEY == 0)); then
    echo "note: no update key (MYGO_UPDATER_PRIVATE_KEY or $KEYFILE): no update files (a release needs them)"
  else
    :
  fi
else
  echo "ad hoc: mygo.json as it is (no update feed, no notarization)"
fi

echo "== sounds"
# The family's cue sounds (React Bits Sound Lab's audio, from the private
# library; never in this repository): scripts/mac.sh syncs a copy made on
# the Linux side, else this fetches one; without access the build plays
# macOS's system sounds.
if [[ ! -f resources/sounds/catalog.json ]]; then
  scripts/sync-sounds.sh --optional || true
fi
if [[ -f resources/sounds/catalog.json ]]; then
  echo "resources/sounds: $(ls resources/sounds/sounds | wc -l | tr -d ' ') sounds"
else
  echo "warning: no resources/sounds: this build's cues are macOS's system sounds (run scripts/sync-sounds.sh where hehehai/actnow-sounds is readable)" >&2
fi

echo "== mygo build"
args=(-platform darwin/arm64)
[[ -n "$IDENTITY" ]] && args+=(-sign "$IDENTITY")
go tool mygo build "${args[@]}"

echo "== verify"
mode=ad-hoc
if [[ -n "$PROFILE" ]]; then
  mode=developer-id
elif [[ -n "$IDENTITY" ]]; then
  mode=signed
else
  :
fi
scripts/release-verify.sh build/darwin-arm64 "$mode"

if compgen -G "build/darwin-arm64/update-*.json" >/dev/null; then
  echo "== actnow.dev ($channel)"
  publish=(--app voxtnew --channel "$channel" --dir build/darwin-arm64 --changelog CHANGELOG.md --min-macos 15.0 --version "$version")
  if ((PUBLISH)); then
    bash scripts/actnow-publish.sh "${publish[@]}"
  else
    bash scripts/actnow-publish.sh "${publish[@]}" --dry-run
  fi
else
  echo "no update files: nothing to publish"
fi
du -sh build/darwin-arm64/VoxtNew.app build/darwin-arm64/*.dmg 2>/dev/null | sed 's/^/built /'
