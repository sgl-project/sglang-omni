#!/usr/bin/env bash
# Shared dependency fingerprint for Omni CI venv reuse, including source pins.
set -euo pipefail

omni_ci_deps_hash() {
  if [ ! -f pyproject.toml ]; then
    echo "pyproject.toml not found in $(pwd)" >&2
    return 1
  fi
  cat pyproject.toml "$(dirname "${BASH_SOURCE[0]}")/prepare_omni_venv.sh" \
    | sha256sum | awk '{print $1}'
}
