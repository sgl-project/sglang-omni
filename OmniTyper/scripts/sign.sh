#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IDENTITY="${CODE_SIGN_IDENTITY:-}"
SIGNING_OPTIONS=(--options runtime)
if [[ -z "$IDENTITY" ]]; then
  SIGNING_OPTIONS+=(--timestamp=none)
  NAME='OmniTyper Local Development'
  KEYCHAIN="$(security default-keychain -d user | sed 's/^ *"//; s/" *$//')"
  if ! CERTIFICATE="$(security find-certificate -c "$NAME" -p "$KEYCHAIN" 2>/dev/null)"; then
    umask 077
    SIGNING_TEMP="$(mktemp -d)"
    trap 'rm -rf "$SIGNING_TEMP"' EXIT
    cat > "$SIGNING_TEMP/certificate.cnf" <<'EOF'
[req]
distinguished_name=subject
x509_extensions=signing
prompt=no
[subject]
CN=OmniTyper Local Development
[signing]
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,codeSigning
EOF
    echo 'Creating a persistent local signing identity in your default Keychain.' >&2
    /usr/bin/openssl req -new -newkey rsa:2048 -x509 -sha256 -days 3650 -nodes \
      -config "$SIGNING_TEMP/certificate.cnf" -keyout "$SIGNING_TEMP/key.pem" -out "$SIGNING_TEMP/certificate.pem"
    security import "$SIGNING_TEMP/key.pem" -k "$KEYCHAIN" -x -T /usr/bin/codesign
    security import "$SIGNING_TEMP/certificate.pem" -k "$KEYCHAIN"
    CERTIFICATE="$(cat "$SIGNING_TEMP/certificate.pem")"
  fi
  # Note (Codex): Select self-signed identities by fingerprint; no root trust is required.
  IDENTITY="$(printf '%s\n' "$CERTIFICATE" | /usr/bin/openssl x509 -noout -fingerprint -sha1 | sed 's/.*=//; s/://g')"
elif [[ "$IDENTITY" == '-' ]]; then
  echo 'warning: explicit ad-hoc signing; updates can invalidate Microphone and Accessibility grants.' >&2
fi
codesign --force --sign "$IDENTITY" "${SIGNING_OPTIONS[@]}" \
  --entitlements "$APP_ROOT/Resources/Entitlements.plist" "$1"
codesign --verify --strict "$1"
