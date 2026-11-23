#!/usr/bin/env bash
# Checks a checkout before a release is tagged. It reads only: it builds,
# signs, uploads and tags nothing (docs/release.md §2.1).
#
#   scripts/release-preflight.sh 2.1.2
#
# One row per check; exit 1 when a row says FAIL. A "note" row does not
# fail the run:
#   version    mygo.json and the helper's Info.plist both say the version
#   changelog  CHANGELOG.md has "## [VERSION] - DATE" with English and 中文
#   links      the six pages that link the disk image link this version's
#   pulls      [Voxt] pull requests merged since the last stable tag that
#              CHANGELOG.md does not name (a note: a change no user can
#              notice needs no entry)
#   signing    the Developer ID certificate's last day (a note when this
#              machine has none; FAIL under 30 days)
#
# VOXT_PREFLIGHT_ROOT: the Voxt directory to check, for the script's test.
set -uo pipefail

version="${1:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: release-preflight.sh VERSION   (a stable version, like 2.1.2)" >&2
  exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${VOXT_PREFLIGHT_ROOT:-$HERE/..}"
cd "$ROOT" || exit 2
failed=0
row() { # row NAME ok|FAIL|note TEXT
  printf '%-10s %-5s %s\n' "$1" "$2" "$3"
  if [[ "$2" == FAIL ]]; then failed=1; fi
}

app="$(sed -n 's/^ *"version": *"\([^"]*\)".*/\1/p' mygo.json | head -1)"
helper="$(grep -A1 CFBundleShortVersionString helper/Support/Info.plist | sed -n 's/.*<string>\(.*\)<\/string>.*/\1/p')"
if [[ "$app" == "$version" && "$helper" == "$version" ]]; then
  row version ok "$version"
else
  row version FAIL "mygo.json has ${app:-nothing}, helper/Support/Info.plist has ${helper:-nothing}, want $version"
fi

section="$(awk -v v="## [$version] - " 'index($0, v) == 1 { on = 1; next } /^## \[/ { on = 0 } on' CHANGELOG.md)"
# Capture before grepping: in a pipeline grep -q exits at the first match and
# sed dies with SIGPIPE, which pipefail reads as a failure.
english="$(sed '/^### 中文$/,$d' <<<"$section")"
chinese="$(sed -n '/^### 中文$/,$p' <<<"$section")"
if [[ -z "$section" ]]; then
  row changelog FAIL "CHANGELOG.md has no \"## [$version] - DATE\" section"
elif ! grep -q '^### 中文$' <<<"$section"; then
  row changelog FAIL "the $version section has no \"### 中文\" part"
elif ! grep -q '^- ' <<<"$english"; then
  row changelog FAIL "the $version section has no English entry before \"### 中文\""
elif ! grep -q '^- ' <<<"$chinese"; then
  row changelog FAIL "the $version section has no Chinese entry after \"### 中文\""
else
  row changelog ok "English and 中文"
fi

dmg="https://github.com/sgl-project/sglang-omni/releases/download/voxt-v$version/Voxt-$version-arm64.dmg"
missing=""
for page in ../README.md README.md README.zh-CN.md ../docs/get_started/voxt.md \
  ../docs/get_started/voxt-tutorials/start.md ../docs/get_started/voxt-tutorials/en/start.md; do
  grep -qF "$dmg" "$page" 2>/dev/null || missing+=" $page"
done
if [[ -z "$missing" ]]; then
  row links ok "six pages link Voxt-$version-arm64.dmg"
else
  row links FAIL "no link to Voxt-$version-arm64.dmg in:$missing"
fi

last="$(git tag -l 'voxt-v*' | grep -E '^voxt-v[0-9]+\.[0-9]+\.[0-9]+$' | grep -vxF "voxt-v$version" | sort -V | tail -1)"
if [[ -z "$last" ]]; then
  row pulls note "no earlier stable tag here (fetch the tags)"
else
  unnamed=""
  while read -r n; do
    grep -Eq "#$n([^0-9]|$)" CHANGELOG.md || unnamed+=" #$n"
  done < <(git log "$last..HEAD" --format=%s -- . | sed -n 's/^\[Voxt\].*(#\([0-9]*\))$/\1/p')
  if [[ -z "$unnamed" ]]; then
    row pulls ok "CHANGELOG.md names every [Voxt] pull request since $last"
  else
    row pulls note "since $last, not in CHANGELOG.md:$unnamed"
  fi
fi

# release-identity.sh prints each certificate's dates, never the key.
until="$("$HERE/release-identity.sh" 2>&1 >/dev/null | sed -n 's/.*  until \(.*\)$/\1/p' | tail -1)"
if [[ -z "$until" ]]; then
  row signing note "no Developer ID certificate on this machine"
else
  days=$((($(date -j -f '%b %e %T %Y %Z' "$until" +%s) - $(date +%s)) / 86400))
  if ((days < 30)); then
    row signing FAIL "the certificate ends $until ($days days)"
  else
    row signing ok "the certificate ends $until ($days days)"
  fi
fi
exit "$failed"
