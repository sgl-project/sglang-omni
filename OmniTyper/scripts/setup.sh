#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "$APP_ROOT/.." && pwd)"
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'OmniTyper local inference requires macOS 14+ on Apple Silicon.' >&2
  exit 1
fi
export SGLANG_OMNI_VENV="${OMNITYPER_VENV:-$APP_ROOT/.venv}"
bash "$REPO_ROOT/install.sh" --non-interactive
uv pip install --python "$SGLANG_OMNI_VENV/bin/python" -r "$APP_ROOT/backend/requirements.txt"
OMNITYPER_PYTHON="$SGLANG_OMNI_VENV/bin/python" bash "$APP_ROOT/scripts/build.sh"
echo "Ready. Launch with: open '$APP_ROOT/dist/OmniTyper.app'"
