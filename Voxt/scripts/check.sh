#!/usr/bin/env bash
# The fast checks, the same locally and in Voxt CI: gofmt, MyGo's vet and
# internal/archtest (file sizes, import layering, view files).
# Run from anywhere: scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

bad="$(git ls-files -z '*.go' ':!:third_party/**' | xargs -0 gofmt -l)"
if [ -n "$bad" ]; then
	echo "::error::not gofmt-ed (run gofmt -w): $bad"
	exit 1
else
	echo "gofmt: clean"
fi

go tool mygo vet .
echo "mygo vet: clean"

CGO_ENABLED=0 go test -count=1 ./internal/archtest
