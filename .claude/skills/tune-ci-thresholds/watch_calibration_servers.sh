#!/usr/bin/env bash
# Tab B: follow every server log launched by the active pytest for one GPU group.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: watch_calibration_servers.sh <gpu-group> <run-dir> [<run-dir> ...]" >&2
  exit 2
fi

GPU_GROUP="$1"
shift
RUN_DIRS=("$@")
POLL_S="${CALIBRATION_SERVER_WATCH_POLL_S:-1}"
# On attach, show recent context then follow (avoid dumping multi-MB history).
TAIL_N="${CALIBRATION_SERVER_WATCH_TAIL_N:-40}"
# IDE terminals cap around ~1MiB; filter ultra-chatty scheduler lines by default.
# Set CALIBRATION_SERVER_WATCH_VERBOSE=1 to keep Decode/Prefill spam.
VERBOSE="${CALIBRATION_SERVER_WATCH_VERBOSE:-0}"
# Always tee a durable copy (not subject to IDE terminal truncation).
TEE_LOG="${CALIBRATION_SERVER_WATCH_TEE_LOG:-/tmp/calibration_tabB_${GPU_GROUP//,/_}.log}"

declare -A TAIL_PIDS=()
declare -A ACTIVE_LOGS=()

clear_active_logs() {
  local key
  for key in "${!ACTIVE_LOGS[@]}"; do
    unset "ACTIVE_LOGS[$key]"
  done
}

echo "[Tab B][$GPU_GROUP] dynamic server logs"
echo "[Tab B][$GPU_GROUP] prefers server.log under basetemp; falls back to pytest runN.log (local non-CI)"
echo "[Tab B][$GPU_GROUP] tee -> $TEE_LOG (full durable stream)"
if [[ "$VERBOSE" != "1" ]]; then
  echo "[Tab B][$GPU_GROUP] filtering Decode/Prefill batch spam (VERBOSE=1 to disable)"
fi
: > "$TEE_LOG"

stop_tail() {
  local log="$1" pid="${TAIL_PIDS[$1]:-}"
  if [[ -n "$pid" ]]; then
    # Kill process group if setsid child; never `wait` a non-child (can hang).
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    sleep 0.05
  fi
  unset 'TAIL_PIDS[$log]'
}

cleanup() {
  local log
  for log in "${!TAIL_PIDS[@]}"; do
    stop_tail "$log"
  done
}
trap cleanup EXIT INT TERM

active_basetemps() {
  local run line
  for run in "${RUN_DIRS[@]}"; do
    while IFS= read -r line; do
      [[ "$line" == *" -m pytest "* ]] || continue
      if [[ "$line" =~ --basetemp=([^[:space:]]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
      fi
    done < <(pgrep -af "[p]ython -m pytest.*$(basename "$run")" 2>/dev/null || true)
  done
}

# Discover attachable logs for one basetemp.
# On GitHub Actions / force_log fixtures: server.log under basetemp.
# Locally: server_log_file() returns None, so router/worker stdout is
# multiplexed into the sibling pytest runN.log next to basetemp_runN.
discover_logs() {
  local basetemp="$1"
  local found=0
  local log base parent runlog

  while IFS= read -r log; do
    printf '%s\n' "$log"
    found=1
  done < <(find "$basetemp" -type f -name 'server.log' -print 2>/dev/null | sort)

  if [[ "$found" -eq 0 ]]; then
    base="$(basename "$basetemp")"
    if [[ "$base" =~ ^basetemp_run([0-9]+)$ ]]; then
      parent="$(dirname "$basetemp")"
      runlog="$parent/run${BASH_REMATCH[1]}.log"
      if [[ -f "$runlog" ]]; then
        printf '%s\n' "$runlog"
      fi
    fi
  fi
}

log_label() {
  local log="$1"
  local name parent
  name="$(basename "$log")"
  if [[ "$name" == "server.log" ]]; then
    basename "$(dirname "$log")"
  else
    parent="$(basename "$(dirname "$log")")"
    printf '%s/%s' "$parent" "$name"
  fi
}

attach_log() {
  local log="$1"
  local label filter_cmd
  label="$(log_label "$log")"
  echo "[Tab B][$GPU_GROUP] attach -> $log" | tee -a "$TEE_LOG"
  if [[ "$VERBOSE" == "1" ]]; then
    filter_cmd="cat"
  else
    # Drop the lines that blow the IDE terminal budget in seconds.
    filter_cmd="grep -E -v 'scheduler_metrics_mixin: (Decode|Prefill) batch'"
  fi
  # New session so stop_tail can kill the whole group.
  # Line-buffer everything; tee durable full/filtered stream for operators.
  setsid bash -c "
    stdbuf -oL -eL tail -n '${TAIL_N}' -F '${log}' \
      | stdbuf -oL -eL sed -u \"s|^|[${label}] |\" \
      | stdbuf -oL -eL ${filter_cmd} \
      | stdbuf -oL -eL tee -a '${TEE_LOG}'
  " &
  TAIL_PIDS["$log"]=$!
}

while true; do
  clear_active_logs
  while IFS= read -r basetemp; do
    [[ -d "$basetemp" ]] || continue
    while IFS= read -r log; do
      ACTIVE_LOGS["$log"]=1
      pid="${TAIL_PIDS[$log]:-}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        attach_log "$log"
      fi
    done < <(discover_logs "$basetemp")
  done < <(active_basetemps | sort -u)

  for log in "${!TAIL_PIDS[@]}"; do
    if [[ -z "${ACTIVE_LOGS[$log]:-}" ]]; then
      echo "[Tab B][$GPU_GROUP] detach old server -> $log" | tee -a "$TEE_LOG"
      stop_tail "$log"
    fi
  done
  sleep "$POLL_S"
done
