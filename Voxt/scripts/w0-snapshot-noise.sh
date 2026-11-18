#!/usr/bin/env bash
# Measures MyGo's run-to-run snapshot noise in five pairs on one clean commit
# and replaces scripts/w0-snapshots-noisy.list with the union of qualifying
# files. Run from Voxt/ with SNAP_NOISE_TMP set to a worktree-local evidence
# directory. Renders, logs and per-pair reports remain in that directory.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
if [ -n "$(git status --porcelain -- .)" ]; then
	echo "commit the Voxt tree before measuring snapshot noise" >&2
	exit 2
fi
tmp="${SNAP_NOISE_TMP:?set SNAP_NOISE_TMP to a worktree-local evidence directory}/w0-snapshot-noise"
rm -rf "$tmp"
mkdir -p "$tmp"
sets=(main min home overlay ink)

for iteration in {1..5}; do
	pair="$tmp/run-$iteration"
	mkdir -p "$pair"
	scripts/w0-snapshots.sh render "$pair/before"
	scripts/w0-snapshots.sh render "$pair/after"
	for set_name in "${sets[@]}"; do
		go run ./internal/tools/w0snapcmp -measure -set "$set_name" \
			-report "$pair/$set_name.list" "$pair/before/$set_name" "$pair/after/$set_name" \
			>"$pair/$set_name.log" 2>&1
	done
done

{
	printf '%s\n' \
		'# Measured MyGo raster noise. Each entry is an exact SET/FILE.png path.' \
		'# Regenerate from Voxt/: SNAP_NOISE_TMP="$PWD/../.hmz-evidence/w0/snapshot-noise" scripts/w0-snapshot-noise.sh' \
		'# The command takes the union from five pairs; only equal dimensions, equal alpha and RGB deltas <= 1 qualify.'
	for iteration in {1..5}; do
		for set_name in "${sets[@]}"; do
			cat "$tmp/run-$iteration/$set_name.list"
		done
	done | sort -u
} >"$tmp/w0-snapshots-noisy.list"
cp "$tmp/w0-snapshots-noisy.list" scripts/w0-snapshots-noisy.list
echo "wrote scripts/w0-snapshots-noisy.list from five pairs in $tmp"
