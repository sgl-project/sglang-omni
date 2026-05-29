#!/usr/bin/env bash
set -euo pipefail

if [ -z "${OMNI_CI_HOME:-}" ]; then
  echo "OMNI_CI_HOME is not set" >&2
  exit 1
fi

if [[ "${OMNI_CI_HOME}" == *".."* ]]; then
  echo "unsafe OMNI_CI_HOME: ${OMNI_CI_HOME}" >&2
  exit 1
fi

if [[ "${OMNI_CI_HOME}" != /github/home/pr-* ]] && [[ "${OMNI_CI_HOME}" != /github/home/run-* ]]; then
  echo "OMNI_CI_HOME must be under /github/home/pr-* or /github/home/run-*: ${OMNI_CI_HOME}" >&2
  exit 1
fi
