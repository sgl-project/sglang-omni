#!/usr/bin/env bash
# Tab A supervision for `tune.py run` — always tail the *active* pytest log.
#
# Resolves the log from the running pytest process (--basetemp=.../basetemp_runK),
# NOT from `ls -t` mtime (stale completed logs like mmmu run1 stay frozen forever).
# DO NOT tail <run-dir>/run.log — that duplicates Tab B (tune.py milestones).
set -euo pipefail

RUN_DIR="${1:?usage: tail_calibration_pytest.sh <run-dir>}"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RUN_DIR_BASENAME="$(basename "$RUN_DIR")"
POLL_S="${TAIL_CALIBRATION_POLL_S:-1}"

echo "[Tab A] pytest/server verbose — run-dir: $RUN_DIR"
echo "[Tab A] tracks ACTIVE pytest via process + run.log (never stale ls -t tail)"

# Return path to the pytest log for the currently running calibration pytest.
resolve_active_log() {
  local line basetemp test_name k log

  while IFS= read -r line; do
    [[ "$line" == *" -m pytest "* ]] || continue
    [[ "$line" == *"${RUN_DIR_BASENAME}"* ]] || continue
    if [[ "$line" =~ --basetemp=([^[:space:]]+) ]]; then
      basetemp="${BASH_REMATCH[1]}"
      if [[ "$basetemp" =~ _pytest/([^/]+)/basetemp_run([0-9]+)$ ]]; then
        test_name="${BASH_REMATCH[1]}"
        k="${BASH_REMATCH[2]}"
        log="${RUN_DIR}/_pytest/${test_name}/run${k}.log"
        if [[ -f "$log" ]]; then
          echo "$log"
          return 0
        fi
      fi
    fi
  done < <(pgrep -af "[p]ython -m pytest.*${RUN_DIR_BASENAME}" 2>/dev/null || true)

  # Between tests: use the last "running…" milestone (not ok/failed).
  if [[ -f "${RUN_DIR}/run.log" ]]; then
    local milestone test_name run_k
    milestone="$(grep -E "^\[test_.+\] run [0-9]+/[0-9]+" "${RUN_DIR}/run.log" \
      | grep " running" | tail -1 || true)"
    if [[ -n "$milestone" ]] \
        && [[ "$milestone" =~ ^\[([^]]+)\]\ run\ ([0-9]+)/ ]]; then
      test_name="${BASH_REMATCH[1]}"
      run_k="${BASH_REMATCH[2]}"
      log="${RUN_DIR}/_pytest/${test_name}/run${run_k}.log"
      if [[ -f "$log" ]]; then
        echo "$log"
        return 0
      fi
    fi
  fi

  return 1
}

current=""
tail_pid=""

stop_tail() {
  if [[ -n "${tail_pid:-}" ]] && kill -0 "$tail_pid" 2>/dev/null; then
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
  fi
  tail_pid=""
}

while true; do
  if ! active="$(resolve_active_log)"; then
    if [[ -n "$current" ]]; then
      echo "[Tab A] pytest idle — waiting for next stage (stopped tail on $current)"
      stop_tail
      current=""
    else
      echo "[Tab A] waiting for calibration pytest to start…"
    fi
    sleep "$POLL_S"
    continue
  fi

  if [[ "$active" == "$current" ]]; then
    if [[ -n "$tail_pid" ]] && ! kill -0 "$tail_pid" 2>/dev/null; then
      tail_pid=""
    fi
    if [[ -z "$tail_pid" ]]; then
      echo "[Tab A] tail -f $active"
      tail -f "$active" &
      tail_pid=$!
    fi
    sleep "$POLL_S"
    continue
  fi

  stop_tail
  if [[ -n "$current" ]]; then
    echo "[Tab A] switch → $active"
  else
    echo "[Tab A] tail -f $active"
  fi
  current="$active"
  tail -f "$current" &
  tail_pid=$!
  sleep "$POLL_S"
done
