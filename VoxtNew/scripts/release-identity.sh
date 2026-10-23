#!/usr/bin/env bash
# Prints the Developer ID Application identity of a team that signs a
# release: its SHA-1 hash, a tab, its name, for VOXTNEW_SIGN_IDENTITY
# (scripts/release.sh, docs/release.md §1.1). VoxtNew is signed, like the
# other ActNow apps, by the team 779M533DL5 (Developer ID Application: wei
# guan, the account of york@actnow.dev). RY2TQTYSKQ signed it until 10-19.
#
# Sign with the hash: codesign refuses a name that two certificates share
# ("ambiguous"). A keychain may hold several valid certificates of one
# team (a renewed one beside the old) and the identities of other teams;
# the same certificate in several keychains is listed once per keychain
# (mac-air, 10-19: 779M533DL5's one certificate four times, from the login
# and the herenow-signing keychains) and counts once here. Of the team's
# valid identities (security find-identity -v: not expired, not revoked,
# with their private key) this takes the newest: the latest notBefore, then
# the latest notAfter, then the greatest SHA-1, so the choice never depends
# on the keychain order.
#
#   scripts/release-identity.sh [TEAMID] [KEYCHAIN]      # TEAMID: 779M533DL5
#
# It fails when the team has no valid Developer ID Application identity,
# listing those it saw (names only). The candidates and their dates go to
# stderr.
set -euo pipefail

team="${1:-779M533DL5}"
keychain="${2:-}"
if [[ ! "$team" =~ ^[A-Z0-9]{10}$ ]]; then
  echo "release-identity.sh: $team is not a Team ID (10 letters and digits)" >&2
  exit 2
fi
list="$(security find-identity -v -p codesigning ${keychain:+"$keychain"})"
# Lines such as:   1) 0123…ABCD "Developer ID Application: Name (TEAMID)"
match="$(sed -nE "s/^ *[0-9]+\) ([0-9A-F]{40}) \"(Developer ID Application: .* \($team\))\"\$/\1	\2/p" <<<"$list" | sort -u)"
if [[ -z "$match" ]]; then
  echo "release-identity.sh: no valid Developer ID Application identity of the team $team; found:" >&2
  sed -nE 's/^ *[0-9]+\) [0-9A-F]{40} (".*")$/  \1/p' <<<"$list" >&2
  exit 1
fi
if (($(wc -l <<<"$match") == 1)); then
  echo "$match"
  exit 0
fi

# Several: date each certificate (by its SHA-1) and take the newest.
pems="$(mktemp -d)"
trap 'rm -rf "$pems"' EXIT
name="$(head -1 <<<"$match" | cut -f2)"
security find-certificate -a -Z -p -c "$name" ${keychain:+"$keychain"} >"$pems/all.txt"
# The output repeats "SHA-1 hash: …" then the PEM block, per certificate.
awk -v dir="$pems" '
  /^SHA-1 hash: / { sha = $3; next }
  /-----BEGIN CERTIFICATE-----/ { out = dir "/" sha ".pem" }
  out != "" { print > out }
  /-----END CERTIFICATE-----/ { close(out); out = "" }
' "$pems/all.txt"
best="" best_key=""
while IFS=$'\t' read -r sha label; do
  pem="$pems/$sha.pem"
  [[ -f "$pem" ]] || continue
  start="$(openssl x509 -in "$pem" -noout -startdate | cut -d= -f2)"
  end="$(openssl x509 -in "$pem" -noout -enddate | cut -d= -f2)"
  key="$(date -j -f '%b %e %T %Y %Z' "$start" +%s) $(date -j -f '%b %e %T %Y %Z' "$end" +%s)"
  echo "release-identity.sh: $sha  from $start  until $end" >&2
  if [[ -z "$best" ]] || [[ "$(printf '%s\n%s\n' "$key" "$best_key" | sort -n -k1,1 -k2,2 | tail -1)" == "$key" ]]; then
    best="$sha	$label" best_key="$key"
  fi
done <<<"$match"
if [[ -z "$best" ]]; then
  echo "release-identity.sh: could not date the certificates of $team; signing with the first" >&2
  best="$(head -1 <<<"$match")"
fi
echo "release-identity.sh: $(wc -l <<<"$match" | tr -d ' ') identities of the team $team; the newest is ${best%%	*}" >&2
echo "$best"
