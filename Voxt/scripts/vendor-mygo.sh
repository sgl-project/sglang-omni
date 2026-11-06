#!/usr/bin/env bash
# Vendors MyGo with VoxtNew's patches into third_party/mygo, which the
# replace in go.mod builds against until upstream (egoist/mygo) ships the
# fixes: the module at the version go.mod requires, from the module cache,
# without its website, docs, examples, CI, tests and test data, with
# third_party/mygo-patches/*.patch applied in name order (each against the
# whole module, tests included, as sent upstream).
#
#   scripts/vendor-mygo.sh          # make third_party/mygo anew
#   scripts/vendor-mygo.sh --check  # fail unless third_party/mygo is what it makes
#
# To move to another MyGo: change the require in go.mod, run this, check
# the patches still apply and the tests pass (on the Mac too). Drop a patch
# once upstream has it; with none left, delete third_party/mygo and the
# replace (go mod edit -dropreplace github.com/egoist/mygo).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCHES="$ROOT/third_party/mygo-patches"
DEST="$ROOT/third_party/mygo"
cd "$ROOT"

version="$(awk '$1 == "github.com/egoist/mygo" && $2 ~ /^v/ { print $2; exit }' go.mod)"
[[ -n "$version" ]] || { echo "go.mod requires no github.com/egoist/mygo" >&2; exit 1; }
# The module cache has it once anything was built; else download it.
src="$(go env GOMODCACHE)/github.com/egoist/mygo@$version"
if [[ ! -f "$src/go.mod" ]]; then
  info="$(GOFLAGS=-mod=mod go mod download -json "github.com/egoist/mygo@$version" || true)"
  src="$(awk -F'"' '$2 == "Dir" { print $4 }' <<<"$info")"
  [[ -d "$src" ]] || { echo "cannot download github.com/egoist/mygo@$version: $info" >&2; exit 1; }
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
out="$work/mygo"
mkdir -p "$out"
(cd "$src" && tar cf - --exclude ./website --exclude ./docs --exclude ./examples --exclude ./.github .) | (cd "$out" && tar xf -)
chmod -R u+w "$out"
for p in "$PATCHES"/*.patch; do
  [[ -e "$p" ]] || continue
  # Hunks of the parts left out (a patch's docs) stay upstream with them.
  (cd "$out" && git apply --whitespace=nowarn --exclude='website/*' --exclude='docs/*' --exclude='examples/*' --exclude='.github/*' "$p") || { echo "patch $(basename "$p") does not apply to mygo $version" >&2; exit 1; }
done
# Tests and their data stay upstream (some hold sample credentials that
# look real to secret scanners); so do git's settings for the module.
find "$out" -name '*_test.go' -delete
find "$out" -type d -name testdata -prune -exec rm -rf {} +
find "$out" \( -name .gitignore -o -name .gitattributes \) -delete
# Directories that held only tests (internal/e2e) are left empty: git keeps
# no empty directories, so drop them, or --check sees one git cannot hold.
find "$out" -type d -empty -delete
# VoxtNew's checks run gofmt -l over the tree; a file upstream left
# unformatted is formatted here.
unformatted="$(gofmt -l "$out")"
[[ -z "$unformatted" ]] || gofmt -w $unformatted
echo "github.com/egoist/mygo $version + $(ls "$PATCHES"/*.patch 2>/dev/null | wc -l | tr -d ' ') patches (scripts/vendor-mygo.sh)" > "$out/VOXTNEW-VENDORED"

if [[ "${1:-}" == "--check" ]]; then
  if ! diff -r "$out" "$DEST" > "$work/diff"; then
    head -20 "$work/diff" >&2
    echo "third_party/mygo is not mygo $version with the patches: run scripts/vendor-mygo.sh" >&2
    exit 1
  fi
  echo "third_party/mygo is mygo $version with the patches"
  exit 0
fi
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$out" "$DEST"
go mod edit -replace github.com/egoist/mygo=./third_party/mygo
echo "vendored mygo $version + patches in third_party/mygo"
