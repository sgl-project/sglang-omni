#!/usr/bin/env bash
# Vendors actnowkit with VoxtNew's patches into third_party/actnowkit, which
# the replace in go.mod builds against until upstream ships the Windows
# support: the module at the version go.mod requires, from the module cache,
# with third_party/actnowkit-patches/*.patch applied in name order. Tests and
# conformance fixtures stay vendored so every supported platform runs them.
#
#   scripts/vendor-actnowkit.sh          # make third_party/actnowkit anew
#   scripts/vendor-actnowkit.sh --check  # fail unless it is what this makes
#
# To move to another actnowkit: change the require in go.mod, run this, check
# the patches still apply and the tests pass. Drop patches upstream has; with
# none left, delete third_party/actnowkit and the replace.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCHES="$ROOT/third_party/actnowkit-patches"
DEST="$ROOT/third_party/actnowkit"
cd "$ROOT"

version="$(awk '$1 == "github.com/hehehai/actnowkit" && $2 ~ /^v/ { print $2; exit }' go.mod)"
[[ -n "$version" ]] || { echo "go.mod requires no github.com/hehehai/actnowkit" >&2; exit 1; }
src="$(go env GOMODCACHE)/github.com/hehehai/actnowkit@$version"
if [[ ! -f "$src/go.mod" ]]; then
  info="$(GOFLAGS=-mod=mod go mod download -json "github.com/hehehai/actnowkit@$version" || true)"
  src="$(awk -F'"' '$2 == "Dir" { print $4 }' <<<"$info")"
  [[ -d "$src" ]] || { echo "cannot download github.com/hehehai/actnowkit@$version: $info" >&2; exit 1; }
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
out="$work/actnowkit"
mkdir -p "$out"
(cd "$src" && tar cf - .) | (cd "$out" && tar xf -)
chmod -R u+w "$out"
for p in "$PATCHES"/*.patch; do
  [[ -e "$p" ]] || continue
  (cd "$out" && git -c core.autocrlf=false apply --whitespace=nowarn "$p") || { echo "patch $(basename "$p") does not apply to actnowkit $version" >&2; exit 1; }
done
unformatted="$(gofmt -l "$out")"
[[ -z "$unformatted" ]] || gofmt -w $unformatted
echo "github.com/hehehai/actnowkit $version + $(ls "$PATCHES"/*.patch 2>/dev/null | wc -l | tr -d ' ') patches (scripts/vendor-actnowkit.sh)" > "$out/VOXTNEW-VENDORED"

if [[ "${1:-}" == "--check" ]]; then
  if ! diff -r "$out" "$DEST" > "$work/diff"; then
    head -20 "$work/diff" >&2
    echo "third_party/actnowkit is not actnowkit $version with the patches: run scripts/vendor-actnowkit.sh" >&2
    exit 1
  fi
  echo "third_party/actnowkit is actnowkit $version with the patches"
  exit 0
fi
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$out" "$DEST"
go mod edit -replace github.com/hehehai/actnowkit=./third_party/actnowkit
echo "vendored actnowkit $version + patches in third_party/actnowkit"
