#!/bin/sh
# Copies the ActNow family's cue sounds VoxtNew offers from the family's
# private sound library (github.com/hehehai/actnow-sounds) into
# resources/sounds, which ships in the app (Contents/Resources/sounds).
# The audio is React Bits Sound Lab's (NOTICE): it may be used inside the
# app, not redistributed on its own. This repository is public, so
# the copy is not committed (.gitignore): official builds run this first
# (scripts/release.sh does), and a build without it plays macOS's system
# sounds.
#
#   scripts/sync-sounds.sh [--optional]
#
# The library: $ACTNOW_SOUNDS, else ../actnow-sounds beside the repository,
# else a clone in ~/.cache/actnow-sounds (needs access to the private
# repository). --optional: without access, say so and leave the build
# without the sounds (it plays macOS's system sounds).
set -eu
cd "$(dirname "$0")/.."
optional=0
[ "${1:-}" = --optional ] && optional=1
top=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
src=${ACTNOW_SOUNDS:-}
if [ -z "$src" ]; then
	if [ -f "$top/../actnow-sounds/catalog.json" ]; then
		src="$top/../actnow-sounds"
	else
		src="${XDG_CACHE_HOME:-$HOME/.cache}/actnow-sounds"
		[ -d "$src/.git" ] && git -C "$src" pull -q --ff-only 2>/dev/null || true
	fi
fi
if [ ! -f "$src/catalog.json" ]; then
	rm -rf "$src"
	if ! git clone -q --depth 1 https://github.com/hehehai/actnow-sounds "$src" 2>/dev/null; then
		if [ $optional = 1 ]; then
			echo "sync-sounds: no access to hehehai/actnow-sounds: this build plays macOS's system sounds" >&2
			exit 0
		fi
		echo "sync-sounds: cannot clone hehehai/actnow-sounds (private; set ACTNOW_SOUNDS to a copy)" >&2
		exit 1
	fi
fi
# The cues of a recording: taps, selections, toggles, feedback, system
# and media sounds, and every sound meant to start or stop something.
go run github.com/hehehai/actnowkit/cmd/actnow-sounds subset -from "$src" -to resources/sounds \
	-groups tap,select,toggle,feedback,system,media -uses record.start,record.stop,start,stop
