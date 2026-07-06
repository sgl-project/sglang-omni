#!/usr/bin/env bash
# Remove a PR- or run-scoped CI home directory on the self-hosted runner.
#
# Omni CI preserves /data/omni-ci/pr-* across workflow runs so pushes within the
# runner TTL (e.g. 3 days) can reuse venv/cache; this script is used on PR close
# and for ephemeral workflow_dispatch run-* homes—not after every PR CI run.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <ci-home>" >&2
  exit 1
fi

CI_HOME="$1"

if [[ "${CI_HOME}" == *".."* ]]; then
  echo "refusing to remove unsafe CI home path: ${CI_HOME}" >&2
  exit 1
fi

if [[ "${CI_HOME}" != /data/omni-ci/pr-* ]] && [[ "${CI_HOME}" != /data/omni-ci/run-* ]]; then
  echo "refusing to remove CI home outside /data/omni-ci/pr-* or /data/omni-ci/run-*: ${CI_HOME}" >&2
  exit 1
fi

if [ -e "${CI_HOME}" ]; then
  echo "Removing ${CI_HOME}..."
  rm -rf "${CI_HOME}"
else
  echo "CI home already absent: ${CI_HOME}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -f ".github/scripts/delete_gpu_process.sh" ]; then
    bash .github/scripts/delete_gpu_process.sh || true
  fi
fi

echo "PR CI home cleanup complete: ${CI_HOME}"
