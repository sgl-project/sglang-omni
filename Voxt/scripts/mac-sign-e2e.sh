#!/usr/bin/env bash
# The release's signing without a Developer ID (docs/release.md §2.2):
# a throwaway self-signed code signing identity in a keychain of its own
# stands in for the certificate, and scripts/release.sh signs the whole
# app with it, as it would with the Developer ID: every Mach-O file and
# framework among the resources with the hardened runtime and a secure
# timestamp, checked by release-verify.sh in its "signed" mode. Then the
# keychain is deleted and the keychain search list put back.
#
#   MAC_DIR=code/VoxtNew-sign scripts/mac.sh scripts/mac-sign-e2e.sh [--runtimes DIR]
#
# --runtimes DIR is release.sh's (default ~/voxtnew-runtime, as on
# mac-air). What it cannot show: Gatekeeper and notarization (a
# self-signed certificate is not Apple's), and library validation, which
# compares Team IDs that a self-signed certificate does not have, so the
# runtimes' start from the bundle is reported but not required.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run it on a Mac: MAC_DIR=code/VoxtNew-sign scripts/mac.sh scripts/mac-sign-e2e.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUNTIMES="$HOME/voxtnew-runtime"
[[ "${1:-}" == --runtimes ]] && RUNTIMES="${2:?--runtimes needs a directory}"
export PATH="/opt/homebrew/bin:$PATH"

WORK="$(mktemp -d)"
KC="$WORK/sign-e2e.keychain-db"
PASS="sign-e2e-$$"
read -r -a SEARCH <<<"$(security list-keychains -d user | tr -d '"' | xargs)"
cleanup() {
  security list-keychains -d user -s "${SEARCH[@]}"
  security delete-keychain "$KC" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "== a throwaway code signing identity"
openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj "/CN=VoxtNew sign e2e $$" \
  -addext "keyUsage=critical,digitalSignature" -addext "extendedKeyUsage=critical,codeSigning" \
  -addext "basicConstraints=critical,CA:false" -keyout "$WORK/key.pem" -out "$WORK/cert.pem" 2>/dev/null
legacy=()
openssl version | grep -q '^OpenSSL 3' && legacy=(-legacy)
openssl pkcs12 -export ${legacy[@]+"${legacy[@]}"} -inkey "$WORK/key.pem" -in "$WORK/cert.pem" \
  -out "$WORK/id.p12" -passout "pass:$PASS"
security create-keychain -p "$PASS" "$KC"
security set-keychain-settings -lut 3600 "$KC"
security unlock-keychain -p "$PASS" "$KC"
security import "$WORK/id.p12" -k "$KC" -P "$PASS" -T /usr/bin/codesign >/dev/null
security set-key-partition-list -S apple-tool:,apple: -s -k "$PASS" "$KC" >/dev/null
security list-keychains -d user -s "$KC" "${SEARCH[@]}"
identity="$(security find-identity -p codesigning "$KC" | sed -nE 's/^ *[0-9]+\) ([0-9A-F]{40}) .*/\1/p' | head -1)"
[[ -n "$identity" ]] || { echo "the identity is not there" >&2; exit 1; }
echo "signing as $identity (self-signed)"

echo "== release.sh, signed"
status=0
VOXTNEW_SIGN_IDENTITY="$identity" scripts/release.sh --runtimes "$RUNTIMES" 2>&1 | tee "$WORK/release.log" || status=$?
# Library validation needs a Team ID: the runtimes' start is not this test's.
fails="$(grep -E '^FAIL ' "$WORK/release.log" | grep -vE 'starts from the app' || true)"
if [[ -n "$fails" ]]; then
  echo "sign e2e: FAIL"
  echo "$fails"
  exit 1
fi
if ((status != 0)) && ! grep -q 'starts from the app' "$WORK/release.log"; then
  echo "sign e2e: FAIL (release.sh exited $status)"
  exit 1
fi
echo "sign e2e: every piece of code is signed with the hardened runtime and a timestamp"
