#!/bin/bash
# Launch N serving replicas on ONE GPU behind a private CUDA MPS daemon.
# Companion to docs/basic_usage/mps_dp.md.
# This is a tested example, not a production process supervisor.
#
# Usage:
#   MODEL=bosonai/higgs-tts-3-4b GPU_ID=0 N=3 MAX_TOTAL_TOKENS=100000 \
#     CORE_BLOCKS="0-9 10-19 20-29" \
#     bash examples/mps_dp/launch.sh up
#   bash examples/mps_dp/launch.sh list
#   bash examples/mps_dp/launch.sh verify [RUN_ID]
#   bash examples/mps_dp/launch.sh down [RUN_ID]
#
# Environment for `up` (defaults in parentheses):
#   MODEL (bosonai/higgs-tts-3-4b), MODEL_NAME (higgs), GPU_ID (0), N (3),
#   BASE_PORT (8801),
#   CORE_BLOCKS: N non-overlapping CPU blocks on the GPU's NUMA node, required.
#   NUMA_NODE: explicit override when the PCI-derived NUMA node is unavailable.
#   MAX_TOTAL_TOKENS: common positive --max-total-tokens cap; required for N > 1.
#   MF: optional explicit --mem-fraction-static override (unset = Higgs default).
set -euo pipefail

STATE_ROOT=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$UID}
CMD=${1:-}
RUN_ARG=${2:-}
HEALTH_TRIES=${HEALTH_TRIES:-50}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-6}
DRAIN_TRIES=${DRAIN_TRIES:-40}
DRAIN_INTERVAL=${DRAIN_INTERVAL:-3}
readonly MPS_STARTUP_TIMEOUT_SECONDS=5
readonly MPS_STARTUP_QUERY_TIMEOUT_SECONDS=1
readonly MPS_STARTUP_POLL_INTERVAL_SECONDS=0.2
readonly MPS_QUERY_TIMEOUT_SECONDS=10
readonly MPS_SHUTDOWN_TIMEOUT_SECONDS=10
readonly MPS_SHUTDOWN_POLL_INTERVAL_SECONDS=1
startup_state=""

die() { echo "error: $*" >&2; exit 1; }

cleanup_failed_startup() {
  echo "startup failed; stopping this run only" >&2
  teardown_state "$startup_state" --keep-state || true
}

pid_is_live() {
  local pid=$1 status
  kill -0 "$pid" 2>/dev/null || return 1
  status=$(ps -o stat= -p "$pid" 2>/dev/null || true)
  case "$status" in Z*|"") return 1;; esac
}

pid_start_time() {
  # A live PID alone does not prove that retained state still owns the process.
  local start_time
  start_time=$(LC_ALL=C ps -o lstart= -p "$1" 2>/dev/null) || return 1
  [ -n "${start_time// /}" ] || return 1
  printf '%s\n' "$start_time"
}

leader_identity_matches() {
  local pid=$1 expected_start=$2 actual_start
  pid_is_live "$pid" || return 1
  actual_start=$(pid_start_time "$pid") || return 1
  [ "$actual_start" = "$expected_start" ]
}

mps_query() {
  local state=$1 cmd=$2 query_timeout=${3:-$MPS_QUERY_TIMEOUT_SECONDS}
  CUDA_MPS_PIPE_DIRECTORY=$state/mps/pipe CUDA_MPS_LOG_DIRECTORY=$state/mps/log \
    timeout "$query_timeout" nvidia-cuda-mps-control <<< "$cmd" 2>> "$state/mps_ctl.err"
}

mps_alive() {
  mps_query "$1" get_default_active_thread_percentage "${2:-$MPS_QUERY_TIMEOUT_SECONDS}" > /dev/null 2>&1
}

mps_control_pid() {
  local pid_file=$1/mps/pipe/nvidia-cuda-mps-control.pid pid
  [ -r "$pid_file" ] || return 1
  read -r pid < "$pid_file"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "$pid"
}

mps_quit() {
  local state=$1 control_pid=$2
  mps_query "$state" quit > /dev/null || {
    echo "error: failed to send quit to the MPS control daemon" >&2
    return 1
  }
  local deadline=$((SECONDS + MPS_SHUTDOWN_TIMEOUT_SECONDS))
  while pid_is_live "$control_pid"; do
    if ((SECONDS >= deadline)); then
      echo "error: MPS control daemon PID $control_pid is still alive after quit" >&2
      return 1
    fi
    sleep "$MPS_SHUTDOWN_POLL_INTERVAL_SECONDS"
  done
}

resolve_numa() {
  if [ -n "${NUMA_NODE:-}" ]; then echo "$NUMA_NODE"; return 0; fi
  # Note (jiaxin): /sys/class/drm ordinals are not guaranteed to match nvidia-smi
  # ordinals, so the NUMA node is derived from the GPU's PCI bus id instead.
  local bus node
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1")
  bus=${bus,,}; bus=${bus:4}
  node=$(cat "/sys/bus/pci/devices/$bus/numa_node" 2>/dev/null || echo "")
  { [ -n "$node" ] && [ "$node" -ge 0 ]; } \
    || die "cannot resolve NUMA node for GPU $1 (pci '$bus'); set NUMA_NODE explicitly"
  echo "$node"
}

find_runs() { ls -d "$STATE_ROOT"/gpu-*/run-* 2>/dev/null || true; }

resolve_state() {
  local arg=$1 matches="" d
  if [ -n "$arg" ]; then
    for d in $(find_runs); do
      [ "$(basename "$d")" = "$arg" ] && matches+="$d"$'\n'
    done
    matches=${matches%$'\n'}
    [ -n "$matches" ] || die "no run state named '$arg' under $STATE_ROOT"
    [ "$(echo "$matches" | wc -l)" -eq 1 ] \
      || { echo "run id '$arg' is ambiguous:" >&2; echo "$matches" >&2; exit 1; }
    echo "$matches"
    return 0
  fi
  matches=$(find_runs)
  if [ -z "$matches" ]; then
    echo "No launcher state found under $STATE_ROOT — refusing to guess." >&2
    echo "Inspect manually before signalling anything:" >&2
    echo "  nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv" >&2
    echo "  ps -o pid,pgid,cmd -p <pid>" >&2
    exit 1
  fi
  [ "$(echo "$matches" | wc -l)" -eq 1 ] \
    || { echo "Multiple runs found; pass a RUN_ID:" >&2; echo "$matches" >&2; exit 1; }
  echo "$matches"
}

tracked_pids() {
  # Note (jiaxin): zombies hold no resources and can never be reaped by this
  # script in init-less containers, so they do not count as live.
  local pgid out="" p
  while IFS=$'\t' read -r _ _ pgid _ _; do
    for p in $(pgrep -g "$pgid" 2>/dev/null || true); do
      pid_is_live "$p" && out+=" $p"
    done
  done < "$1/replicas.tsv"
  echo "$out"
}

run_is_active() {
  local state=$1 port live
  live=$(tracked_pids "$state")
  [ -n "${live// /}" ] && return 0
  mps_alive "$state" && return 0
  while IFS=$'\t' read -r _ _ _ port _; do
    (exec 3<> "/dev/tcp/127.0.0.1/$port") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
  done < "$state/replicas.tsv"
  return 1
}

mps_clients() {
  local state=$1 servers s clients="" out
  if ! out=$(mps_query "$state" get_server_list); then
    return 1
  fi
  servers=$(echo "$out" | grep -E '^[0-9]+$' || true)
  for s in $servers; do
    out=$(mps_query "$state" "get_client_list $s") || return 1
    clients+=" $s:$(echo "$out" | grep -E '^[0-9]+$' | tr '\n' ',' || true)"
  done
  echo "$clients"
}

verify_attach() {
  local state=$1
  [ -n "$state" ] && [ -f "$state/replicas.tsv" ] || die "invalid or missing run state '$state'"
  local art="$state/mps_attach.txt" fail=0 raw entry srv cl all=" " idx pid pgid port log
  : > "$art"
  if ! raw=$(mps_clients "$state"); then
    echo "FAIL: MPS control query failed (see $state/mps_ctl.err)" | tee -a "$art" >&2
    return 1
  fi
  if [ -z "${raw// /}" ]; then
    echo "FAIL: no MPS server under $state/mps/pipe" | tee -a "$art" >&2
    return 1
  fi
  for entry in $raw; do
    srv=${entry%%:*}
    echo "mps_server $srv" >> "$art"
    for cl in $(echo "${entry#*:}" | tr ',' ' '); do
      all+="$cl "
      local owner="UNMATCHED" opgid
      while IFS=$'\t' read -r idx _ opgid oport _; do
        case " $(pgrep -g "$opgid" 2>/dev/null || true) " in
          *" $cl "*) owner="replica $idx (pgid $opgid, port $oport)";;
        esac
      done < "$state/replicas.tsv"
      echo "  client $cl -> $owner" >> "$art"
    done
  done
  while IFS=$'\t' read -r idx pid pgid port log; do
    local expected matched="" p
    expected=$(pgrep -g "$pgid" 2>/dev/null || true)
    for p in $expected; do
      case "$all" in *" $p "*) matched+="$p ";; esac
    done
    if [ -z "$matched" ]; then
      echo "replica $idx (port $port): no attached MPS client; group members without client match: $(echo $expected)" >> "$art"
      echo "attach verification FAILED: replica $idx (port $port) has no process in the MPS client list" >&2
      fail=1
    else
      echo "replica $idx (port $port): attached clients: $matched" >> "$art"
    fi
  done < "$state/replicas.tsv"
  [ "$fail" = 0 ] && echo "RESULT: PASS" >> "$art" || echo "RESULT: FAIL" >> "$art"
  echo "attach mapping written to $art"
  return $fail
}

teardown_state() {
  # Note (jiaxin): these GPUs are shared; teardown only signals processes recorded
  # in this run's state, never scans the whole GPU, and keeps the state directory
  # whenever cleanup cannot be confirmed, so nothing is hidden from inspection.
  local state=$1 keep=${2:-} leader_pid pgid leader_start t live raw control_pid=""
  [ -n "$state" ] && [ -f "$state/replicas.tsv" ] || die "invalid or missing run state '$state'"
  control_pid=$(mps_control_pid "$state" || true)
  while IFS=$'\t' read -r _ leader_pid pgid _ _ leader_start; do
    leader_identity_matches "$leader_pid" "$leader_start" || continue
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done < "$state/replicas.tsv"
  for ((t=1; t<=DRAIN_TRIES; t++)); do
    live=$(tracked_pids "$state")
    [ -z "${live// /}" ] && break
    sleep "$DRAIN_INTERVAL"
  done
  # Note (jiaxin): the pipe is private to this run, so ANY client the daemon still
  # reports is outstanding even if its PID left the tracked groups; quitting around
  # live clients can wedge the MPS server with RPC failures that outlast this run.
  if raw=$(mps_clients "$state"); then
    if [ -z "$control_pid" ]; then
      echo "error: MPS is responding but its control PID is missing or invalid; state kept at $state" >&2
      return 1
    fi
    local entry cl clients="" tracked blocked="" unowned=""
    for entry in $raw; do
      clients+=" $(echo "${entry#*:}" | tr ',' ' ')"
    done
    tracked=" $(tracked_pids "$state") "
    for cl in $clients; do
      case "$tracked" in
        *" $cl "*) blocked+="$cl " ;;
        *) unowned+="$cl " ;;
      esac
    done
    if [ -n "$blocked" ]; then
      echo "error: this run's MPS clients are still alive after TERM+drain: $blocked" >&2
      echo "state kept at $state — inspect (ps -o pid,pgid,cmd -p $blocked), then re-run down" >&2
      return 1
    fi
    if [ -n "$unowned" ]; then
      echo "error: MPS daemon still reports client(s) outside this run's tracked groups: $unowned" >&2
      echo "state kept at $state — inspect (ps -o pid,pgid,cmd -p $unowned), then re-run down" >&2
      return 1
    fi
    mps_quit "$state" "$control_pid" || { echo "state kept at $state" >&2; return 1; }
  elif [ -n "$control_pid" ] && pid_is_live "$control_pid"; then
    echo "error: MPS control PID $control_pid is alive but its control interface is unavailable" >&2
    echo "state kept at $state — inspect $state/mps_ctl.err and retry down" >&2
    return 1
  fi
  live=$(tracked_pids "$state")
  if [ -n "${live// /}" ]; then
    echo "warning: tracked non-client processes survived TERM; last-resort SIGKILL on tracked groups only" >&2
    while IFS=$'\t' read -r _ leader_pid pgid _ _ leader_start; do
      leader_identity_matches "$leader_pid" "$leader_start" || continue
      kill -KILL -- "-$pgid" 2>/dev/null || true
    done < "$state/replicas.tsv"
    sleep 2
  fi
  live=$(tracked_pids "$state")
  if [ -n "${live// /}" ]; then
    echo "error: tracked pids still alive:$live — state kept at $state" >&2
    return 1
  fi
  if [ "$keep" = "--keep-state" ]; then
    echo "processes cleaned; state kept for diagnostics at $state"
  else
    rm -rf -- "$state"
    echo "down: run state $state cleaned; only this run's processes were touched"
  fi
}

up() {
  local model=${MODEL:-bosonai/higgs-tts-3-4b} model_name=${MODEL_NAME:-higgs}
  local gpu=${GPU_ID:-0} n=${N:-3} base_port=${BASE_PORT:-8801} mf=${MF:-}
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer, got '$gpu'"
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "N must be a positive integer, got '$n'"
  [[ "$base_port" =~ ^[1-9][0-9]*$ ]] \
    || die "BASE_PORT must be a positive integer, got '$base_port'"
  ((base_port + n - 1 <= 65535)) \
    || die "ports $base_port through $((base_port+n-1)) exceed 65535"
  [ -n "${CORE_BLOCKS:-}" ] || {
    echo "CORE_BLOCKS is required: N non-overlapping blocks on the GPU's NUMA node." >&2
    echo "Cores on that node: numactl -H" >&2
    exit 1
  }
  local blocks=()
  read -r -a blocks <<< "$CORE_BLOCKS"
  [ "${#blocks[@]}" = "$n" ] || die "CORE_BLOCKS must contain exactly $n blocks"

  local extra_args=() mem_args=()
  if [ "$n" -gt 1 ] && [ -z "${MAX_TOTAL_TOKENS:-}" ]; then
    die "MAX_TOTAL_TOKENS is required for N=$n so every replica has the same KV capacity"
  fi
  if [ -n "${MAX_TOTAL_TOKENS:-}" ]; then
    [[ "$MAX_TOTAL_TOKENS" =~ ^[1-9][0-9]*$ ]] \
      || die "MAX_TOTAL_TOKENS must be a positive integer, got '$MAX_TOTAL_TOKENS'"
    extra_args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
  fi
  if [ -n "$mf" ]; then
    mem_args+=(--mem-fraction-static "$mf")
  fi

  local d
  for d in $(ls -d "$STATE_ROOT/gpu-$gpu"/run-* 2>/dev/null || true); do
    if run_is_active "$d"; then
      die "an active run already exists on GPU $gpu: $d — bring it down first"
    fi
    die "stale run state exists on GPU $gpu: $d — inspect it, then 'down $(basename "$d")' before starting a new run"
  done

  local i port
  for ((i=0; i<n; i++)); do
    port=$((base_port+i))
    if (exec 3<> "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      exec 3>&- 3<&-
      die "port $port is already in use; pick another BASE_PORT"
    fi
  done

  local uuid node run state
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")
  node=$(resolve_numa "$gpu")
  run="run-$(date +%Y%m%d-%H%M%S)-$$"
  state=$STATE_ROOT/gpu-$gpu/$run
  mkdir -p "$state/logs" "$state/mps/pipe" "$state/mps/log"
  : > "$state/replicas.tsv"

  # note(ratish): Without this trap, a later replica failure leaves earlier
  # replicas and the private MPS daemon running; keep the state for diagnosis.
  startup_state=$state
  trap cleanup_failed_startup EXIT

  chmod 700 "$state/mps" "$state/mps/pipe" "$state/mps/log"
  {
    echo "run_id=$run"; echo "gpu_id=$gpu"; echo "gpu_uuid=$uuid"; echo "numa_node=$node"
    echo "model=$model"; echo "model_name=$model_name"; echo "n=$n"
    echo "mem_fraction_static=${mf:-default}"
    echo "base_port=$base_port"; echo "core_blocks=$CORE_BLOCKS"
    echo "max_total_tokens=${MAX_TOTAL_TOKENS:-auto/profiled}"
  } > "$state/manifest"

  export CUDA_MPS_PIPE_DIRECTORY=$state/mps/pipe CUDA_MPS_LOG_DIRECTORY=$state/mps/log
  local mps_launch_status=0
  env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT \
    CUDA_VISIBLE_DEVICES="$uuid" nvidia-cuda-mps-control -d \
    2>> "$state/mps_ctl.err" || mps_launch_status=$?
  # note(ratish): Daemonization can return before the control socket accepts commands.
  local mps_ready=0
  local mps_deadline=$((SECONDS + MPS_STARTUP_TIMEOUT_SECONDS))
  while ((SECONDS < mps_deadline)); do
    if mps_alive "$state" "$MPS_STARTUP_QUERY_TIMEOUT_SECONDS"; then
      mps_ready=1
      break
    fi
    sleep "$MPS_STARTUP_POLL_INTERVAL_SECONDS"
  done
  [ "$mps_ready" = 1 ] \
    || die "MPS control daemon did not become ready (launch status $mps_launch_status; see $state/mps_ctl.err)"
  local control_pid
  control_pid=$(mps_control_pid "$state") \
    || die "MPS control daemon is ready but its PID file is missing or invalid"
  pid_is_live "$control_pid" \
    || die "MPS control daemon PID $control_pid exited during startup"

  local pid leader_start log resolved_tokens
  for ((i=0; i<n; i++)); do
    port=$((base_port+i))
    log=$state/logs/replica_$i.log
    # Note (jiaxin): concurrent colocated launches raced on CUDA-graph capture and
    # memory profiling in testing, so replicas start sequentially behind a health
    # gate; setsid gives each replica its own process group so teardown can signal
    # exactly this run's process trees.
    CUDA_VISIBLE_DEVICES="$uuid" \
    setsid numactl --cpunodebind="$node" --membind="$node" -C "${blocks[$i]}" \
      sgl-omni serve --model-path "$model" --model-name "$model_name" \
        "${mem_args[@]}" "${extra_args[@]}" \
        --host 127.0.0.1 --port "$port" > "$log" 2>&1 < /dev/null &
    pid=$!
    leader_start=$(pid_start_time "$pid") \
      || die "replica $i exited before its process identity could be recorded"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$i" "$pid" "$pid" "$port" "$log" "$leader_start" >> "$state/replicas.tsv"
    local healthy=0 t code
    for ((t=1; t<=HEALTH_TRIES; t++)); do
      if ! pid_is_live "$pid"; then
        echo "replica $i exited during startup; last log lines:" >&2
        tail -n 8 "$log" >&2
        exit 1
      fi
      code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "127.0.0.1:$port/health" || true)
      [ "$code" = 200 ] && { healthy=1; break; }
      sleep "$HEALTH_INTERVAL"
    done
    if [ "$healthy" != 1 ]; then
      echo "replica $i health timeout after $((HEALTH_TRIES*HEALTH_INTERVAL))s; last log lines:" >&2
      tail -n 8 "$log" >&2
      exit 1
    fi
    echo "replica $i healthy on port $port (cores ${blocks[$i]}, mem fraction ${mf:-default})"
    resolved_tokens=""
    resolved_tokens=$(grep -m1 -oE '#tokens:[[:space:]]*[0-9]+' "$log" \
      | grep -oE '[0-9]+$' || true)
    if [ "$n" -gt 1 ]; then
      [ -n "$resolved_tokens" ] \
        || die "replica $i is healthy but its resolved KV capacity is missing from $log"
      [ "$resolved_tokens" = "$MAX_TOTAL_TOKENS" ] \
        || die "replica $i resolved $resolved_tokens KV tokens; expected $MAX_TOTAL_TOKENS"
    fi
    echo "replica $i KV #tokens: ${resolved_tokens:-not found}"
  done

  verify_attach "$state" || exit 1
  if [ "$(cat "$state"/logs/replica_*.log 2>/dev/null | grep -c MpsRpc)" != 0 ]; then
    echo "warning: MpsRpc errors present in replica logs; bring the run down and restart" >&2
    exit 1
  fi
  trap - EXIT
  echo "up: $n replicas on GPU $gpu; token cap ${MAX_TOTAL_TOKENS:-auto/profiled}; state: $state"
  echo "tear down with: bash $0 down $run"
}

case "$CMD" in
  up) up ;;
  down) st=$(resolve_state "$RUN_ARG") || exit 1; teardown_state "$st" ;;
  verify) st=$(resolve_state "$RUN_ARG") || exit 1; verify_attach "$st" ;;
  list) find_runs ;;
  *) die "usage: launch.sh up|down [RUN_ID]|verify [RUN_ID]|list" ;;
esac
