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

# Actions runners use /data/omni-ci/{pr-*,run-*} (a persistent host bind mount,
# safe from the runner's per-job /github/home temp wipe). The manual repro host
# (sglang-h100-ci) keeps its calibration slice at /github/home/calibration.
if [[ "${OMNI_CI_HOME}" != /data/omni-ci/pr-* ]] \
  && [[ "${OMNI_CI_HOME}" != /data/omni-ci/run-* ]] \
  && [[ "${OMNI_CI_HOME}" != /data/omni-ci/calibration ]] \
  && [[ "${OMNI_CI_HOME}" != /github/home/calibration ]]; then
  echo "OMNI_CI_HOME must be under /data/omni-ci/{pr-*,run-*,calibration} or /github/home/calibration: ${OMNI_CI_HOME}" >&2
  exit 1
fi
