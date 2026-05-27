#!/usr/bin/env bash
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

if [[ "${CI_HOME}" != /github/home/pr-* ]] && [[ "${CI_HOME}" != /github/home/run-* ]]; then
  echo "refusing to remove CI home outside /github/home/pr-* or /github/home/run-*: ${CI_HOME}" >&2
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
