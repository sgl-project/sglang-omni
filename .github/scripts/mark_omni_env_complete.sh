#!/usr/bin/env bash
# Record that setup finished successfully for this PR-scoped CI home.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME="$1"

bash "${SCRIPT_DIR}/validate_omni_env_reusable.sh" "${VENV_NAME}"

# shellcheck source=omni_ci_deps_hash.sh
source "${SCRIPT_DIR}/omni_ci_deps_hash.sh"
DEPS_HASH="$(omni_ci_deps_hash)"
MARKER="${OMNI_CI_HOME}/.omni-env-complete"
MARKED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SETUP_GIT_SHA="${GITHUB_SHA:-unknown}"

cat > "${MARKER}" <<EOF
deps_hash=${DEPS_HASH}
venv_name=${VENV_NAME}
marked_at=${MARKED_AT}
setup_git_sha=${SETUP_GIT_SHA}
EOF

echo "Marked OMNI CI environment complete: ${MARKER}"
