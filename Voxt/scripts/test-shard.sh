#!/usr/bin/env bash
# Voxt CI runs the app's package in parallel shards. This prints the -run
# pattern of one shard: the names `go test -list` prints, dealt in turn, so
# each name is in exactly one shard.
#   go test -list . . | scripts/test-shard.sh <shard, from 0> <shards>
set -euo pipefail
grep -E '^(Test|Example|Fuzz)' | awk -v s="$1" -v n="$2" 'NR % n == s' | paste -sd'|' - | sed 's/.*/^(&)$/'
