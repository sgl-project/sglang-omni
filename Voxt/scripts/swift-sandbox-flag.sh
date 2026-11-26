#!/usr/bin/env bash
# Prints SwiftPM's --disable-sandbox when the calling agent already isolates
# the build (VOXTNEW_SWIFT_DISABLE_SANDBOX=1), nothing otherwise. Shared by
# build-helper.sh and mac-e2e.sh so the opt-in lives in exactly one place.
set -euo pipefail
if [[ "${VOXTNEW_SWIFT_DISABLE_SANDBOX:-0}" == "1" ]]; then
  echo "--disable-sandbox"
fi
