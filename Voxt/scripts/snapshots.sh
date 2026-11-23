#!/usr/bin/env bash
# Renders every set of UI snapshots and compares them, for refactors that
# must not change a pixel (refactor R0, docs/architecture.md "Code structure
# rules"). The snapshots are reproducible: two runs of one commit write the
# same bytes (snapshot_env_test.go pins the clock, the data directories and
# the free disk space).
#
#   scripts/snapshots.sh render DIR   render into DIR (emptied first)
#   scripts/snapshots.sh check [BASE] render BASE (default: the merge base
#                                     with origin/main) in a temporary
#                                     worktree and the working tree, and
#                                     compare them
#   scripts/snapshots.sh twice        render the working tree twice and
#                                     compare (is it reproducible?)
#
# The sets, a directory each: main (TestSnapshots*, the window's default
# size), min (the same at the smallest size), home (TestHomeSnapshots),
# overlay (TestOverlayShots), ink (the overlay's ink checks' pictures). The
# overlay's flows (VOXTNEW_FLOW_SHOTS) are left out: they run real
# goroutines and the fake helper's process, so how many pieces of a
# streamed draft a frame shows varies; their tests check the steps.
#
# check and twice exit 1 when a snapshot differs and leave the pictures of
# the differences (before | after | red pixels) in $SNAP_TMP/diff/<set>;
# they delete the renders (about 100 MB each) unless SNAP_KEEP=1. SNAP_TMP
# defaults to /tmp/voxtnew-snapcheck-$USER. Rendering runs under
# nice -n 19 (the Linux host is shared) and takes about half a minute.
#
# The comparison is go run ./internal/tools/snapcmp BEFORE AFTER per set,
# with the snapshots allowed to differ a little listed, with the reason,
# in scripts/snapshots-volatile.txt (none today).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="${SNAP_TMP:-/tmp/voxtnew-snapcheck-${USER:-$(id -u)}}"
sets=(main min home overlay ink)

render() { # render SRC_DIR OUT_DIR
	local src="$1" out="$2"
	rm -rf "$out"
	mkdir -p "${sets[@]/#/$out/}"
	echo "rendering $(git -C "$src" rev-parse --short HEAD)$(git -C "$src" diff --quiet HEAD -- . 2>/dev/null || echo '+changes') into $out" >&2
	run() { (cd "$src" && nice -n 19 env GOMAXPROCS=4 CGO_ENABLED=0 "$@" >&2); }
	run env VOXTNEW_SNAPSHOTS="$out/main" go test -count=1 -timeout 30m -run 'TestSnapshots' .
	run env VOXTNEW_SNAPSHOTS="$out/min" VOXTNEW_SNAPSHOTS_SIZE=min go test -count=1 -timeout 30m -run 'TestSnapshots' .
	run env VOXTNEW_HOME_SHOTS="$out/home" go test -count=1 -timeout 30m -run 'TestHomeSnapshots' .
	run env VOXTNEW_OVERLAY_SHOTS="$out/overlay" go test -count=1 -timeout 30m -run 'TestOverlayShots' .
	run env VOXTNEW_INK="$out/ink" go test -count=1 -timeout 30m -run 'TestOverlayInk|TestAnswerCardInk' .
	echo "$(find "$out" -type f | wc -l) files" >&2
}

compare() { # compare BEFORE AFTER
	rm -rf "$tmp/diff"
	local status=0 s
	for s in "${sets[@]}"; do
		echo "== $s" >&2
		(cd "$root" && go run ./internal/tools/snapcmp -volatile scripts/snapshots-volatile.txt -out "$tmp/diff/$s" "$1/$s" "$2/$s") || {
			local st=$?
			[ "$st" -gt "$status" ] && status=$st
		}
	done
	if [ "$status" = 1 ]; then
		echo "differences: $tmp/diff" >&2
	fi
	return "$status"
}

cleanup() {
	if [ "${SNAP_KEEP:-}" != 1 ]; then
		rm -rf "$tmp/before" "$tmp/after"
	fi
	rmdir "$tmp/diff"/* "$tmp/diff" "$tmp" 2>/dev/null || true
	if [ -n "${base_tree:-}" ]; then
		git -C "$root" worktree remove --force "$base_tree" || true
	fi
}

case "${1:-}" in
render)
	[ $# = 2 ] || { echo "usage: $0 render DIR" >&2; exit 2; }
	render "$root" "$(realpath -m "$2")"
	;;
check)
	base="${2:-$(git -C "$root" merge-base HEAD origin/main)}"
	mkdir -p "$tmp"
	base_tree="$tmp/base"
	trap cleanup EXIT
	git -C "$root" worktree remove --force "$base_tree" 2>/dev/null || rm -rf "$base_tree"
	git -C "$root" worktree add --detach "$base_tree" "$base" >&2
	# The worktree is the whole repository: VoxtNew is a directory of it.
	prefix="$(git -C "$root" rev-parse --show-prefix)"
	[ -f "$base_tree/${prefix}go.mod" ] || { echo "$base has no ${prefix%/}: pass a BASE that has (e.g. voxtnew/dev)" >&2; exit 2; }
	render "$base_tree/$prefix" "$tmp/before"
	render "$root" "$tmp/after"
	compare "$tmp/before" "$tmp/after"
	;;
twice)
	mkdir -p "$tmp"
	trap cleanup EXIT
	render "$root" "$tmp/before"
	render "$root" "$tmp/after"
	compare "$tmp/before" "$tmp/after"
	;;
*)
	sed -n '2,30p' "$0" >&2
	exit 2
	;;
esac
