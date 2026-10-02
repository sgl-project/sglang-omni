#!/bin/bash
# No third-party Python packages, installation, model downloads or real app launches.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
for script in local_runtime install_local start_local; do
    /bin/bash -n "$project_dir/$script.sh"
done
test_python="${SGLANG_OMNI_VENV:-$project_dir/../../.venv-apple}/bin/python"
if [[ ! -x "$test_python" ]]; then test_python="$(command -v python3)"; fi
"$test_python" "$project_dir/Tests/LocalSetup_test.py" -v
