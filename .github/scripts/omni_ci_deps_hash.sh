#!/usr/bin/env bash
# Shared dependency fingerprint for Omni CI venv reuse (full pyproject.toml).
set -euo pipefail

omni_ci_deps_hash() {
  if [ ! -f pyproject.toml ]; then
    echo "pyproject.toml not found in $(pwd)" >&2
    return 1
  fi
  sha256sum pyproject.toml | awk '{print $1}'
}
