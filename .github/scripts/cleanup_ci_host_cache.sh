#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <ci-home>" >&2
  exit 1
fi

echo "cleanup_ci_host_cache.sh is deprecated; use cleanup_ci_pr_home.sh" >&2
exec bash "$(dirname "$0")/cleanup_ci_pr_home.sh" "$1"
