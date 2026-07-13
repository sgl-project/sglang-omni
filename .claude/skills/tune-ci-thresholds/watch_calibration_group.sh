#!/usr/bin/env bash
# Tab A: summarize total calibration progress for one GPU group.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: watch_calibration_group.sh <gpu-group> <run-dir> [<run-dir> ...]" >&2
  exit 2
fi

GPU_GROUP="$1"
shift
RUN_DIRS=("$@")
POLL_S="${CALIBRATION_WATCH_POLL_S:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_PY="${SCRIPT_DIR}/tune.py"
last=""

echo "[Tab A][$GPU_GROUP] calibration group progress"
echo "[Tab A][$GPU_GROUP] runs: ${RUN_DIRS[*]}"

snapshot() {
  local run output summary
  for run in "${RUN_DIRS[@]}"; do
    if [[ ! -f "$run/plan.json" ]]; then
      printf '%s: waiting for plan.json\n' "$run"
      continue
    fi
    output="$(python "$TUNE_PY" strict-audit --run-dir "$run" 2>&1 || true)"
    summary="$(printf '%s\n' "$output" | grep -E '^(STRICT READY:|GIT PROVENANCE:)' | paste -sd' ' -)"
    if [[ -z "$summary" ]]; then
      summary="audit unavailable"
    fi
    if pgrep -af "[p]ython -m pytest.*$(basename "$run")" >/dev/null 2>&1; then
      summary="$summary ACTIVE"
    fi
    printf '%s: %s\n' "$run" "$summary"
  done
}

while true; do
  current="$(snapshot)"
  if [[ "$current" != "$last" ]]; then
    printf '\n[%s][Tab A][%s]\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU_GROUP" "$current"
    last="$current"
  fi
  sleep "$POLL_S"
done
