#!/usr/bin/env bash
# Records the pre-refactor test, coverage and snapshot baselines in DIR.
# Run from Voxt/: scripts/baseline.sh DIR
set -euo pipefail

[ $# = 1 ] || { echo "usage: $0 DIR" >&2; exit 2; }
cd "$(dirname "$0")/.."
mkdir -p "$1/tests"
out="$(cd "$1" && pwd)"

{
	git rev-parse HEAD
	go version
	uname -srm
	sw_vers 2>/dev/null || true
	sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu 2>/dev/null || true
} >"$out/machine.txt"

export CGO_ENABLED=0
go test -list '.*' . >"$out/tests/main.txt"
for pkg in $(go list ./internal/...); do
	name="${pkg#github.com/sgl-project/sglang-omni/Voxt/internal/}"
	go test -list '.*' "$pkg" >"$out/tests/${name//\//_}.txt"
done

tmp="${TMPDIR:-$out/tmp}"
mkdir -p "$tmp"
TMPDIR="$tmp" go test -count=1 -coverprofile="$out/coverage-main.out" . >"$out/coverage-main.txt"
TMPDIR="$tmp" go test -count=1 -coverprofile="$out/coverage-internal.out" ./internal/... ./scripts/... >"$out/coverage-internal.txt"
python3 -c 'import pathlib, sys
out = pathlib.Path(sys.argv[1])
(out / "coverage-packages.txt").write_text("".join((out / name).read_text() for name in ("coverage-main.txt", "coverage-internal.txt")))
blocks = {}
for name in ("coverage-main.out", "coverage-internal.out"):
    for line in (out / name).read_text().splitlines():
        if line.startswith("mode: "):
            continue
        location, statements, count = line.rsplit(" ", 2)
        key = location, statements
        blocks[key] = max(blocks.get(key, 0), int(count))
lines = ["mode: set"] + [f"{location} {statements} {count}" for (location, statements), count in sorted(blocks.items())]
(out / "coverage.out").write_text("\n".join(lines) + "\n")
packages = {}
for (location, statements), count in blocks.items():
    package = location.rsplit("/", 1)[0]
    covered, total = packages.get(package, (0, 0))
    packages[package] = covered + int(statements) * (count > 0), total + int(statements)
summary = [f"{package}\t{100 * covered / total:.1f}%\t{covered}/{total}" for package, (covered, total) in sorted(packages.items())]
(out / "coverage-summary.txt").write_text("\n".join(summary) + "\n")
' "$out"
go tool cover -func="$out/coverage.out" >"$out/coverage-functions.txt"
TMPDIR="$tmp" scripts/w0-snapshots.sh render "$out/snapshots"
python3 -c 'import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(root).as_posix())
' "$out/snapshots" >"$out/snapshots.sha256"
