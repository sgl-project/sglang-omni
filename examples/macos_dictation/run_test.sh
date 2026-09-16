#!/bin/bash
# Deterministic local regression suite: no models, microphone, or real key posting.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf 'This suite requires macOS on Apple Silicon.\n' >&2
  exit 1
fi
tests=(
  client_regression_test.sh
  insertion_test.sh
  feedback_test.sh
  hover_feedback_test.sh
  fallback_feedback_test.sh
  personalization_test.sh
  polish_fidelity_test.sh
  service_configuration_test.sh
  settings_test.sh
  target_observation_test.sh
  general_target_test.sh
  clipboard_paste_test.sh
  current_cursor_test.sh
  current_cursor_feedback_test.sh
)
for test_script in "${tests[@]}"; do
  printf '\nRunning %s\n' "$test_script"
  bash "$project_dir/$test_script"
done
printf '\nPASS: all deterministic dictation regressions. Live model and UI checks are separate.\n'
