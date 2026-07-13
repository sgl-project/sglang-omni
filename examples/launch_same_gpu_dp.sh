#!/bin/bash
# Launch N serving replicas on ONE GPU behind a private CUDA MPS daemon.
# Companion to docs/basic_usage/same_gpu_dp.md.
# This is a tested example, not a production process supervisor.
#
# Usage:
#   MODEL=bosonai/higgs-tts-3-4b GPU_ID=0 N=2 CORE_BLOCKS="0-15 16-31" \
#     bash examples/launch_same_gpu_dp.sh up
#   bash examples/launch_same_gpu_dp.sh list
#   bash examples/launch_same_gpu_dp.sh verify [RUN_ID]
#   bash examples/launch_same_gpu_dp.sh down [RUN_ID]
#
# Environment for `up` (defaults in parentheses):
#   MODEL (bosonai/higgs-tts-3-4b), MODEL_NAME (higgs), GPU_ID (0), N (2),
#   MF (0.42 for N=2, 0.27 for N=3), BASE_PORT (8801),
#   CORE_BLOCKS: N non-overlapping CPU blocks on the GPU's NUMA node, required.
#   NUMA_NODE: explicit override when the PCI-derived NUMA node is unavailable.
#   MAX_TOTAL_TOKENS: optional positive integer; when set, every replica is launched
#     with the same --max-total-tokens cap (unset = engine auto/profiled).
set -euo pipefail

STATE_ROOT=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$USER}
CMD=${1:-}
RUN_ARG=${2:-}
HEALTH_TRIES=${HEALTH_TRIES:-50}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-6}
DRAIN_TRIES=${DRAIN_TRIES:-40}
DRAIN_INTERVAL=${DRAIN_INTERVAL:-3}

die() { echo "error: $*" >&2; exit 1; }

mps_query() {
  local state=$1 cmd=$2
  CUDA_MPS_PIPE_DIRECTORY=$state/mps/pipe CUDA_MPS_LOG_DIRECTORY=$state/mps/log \
    timeout 10 nvidia-cuda-mps-control <<< "$cmd" 2>> "$state/mps_ctl.err"
}

mps_alive() { mps_query "$1" get_default_active_thread_percentage > /dev/null 2>&1; }

mps_quit() {
  local state=$1
  if ! mps_alive "$state"; then
    return 0
  fi
  mps_query "$state" quit > /dev/null || true
  local t
  for ((t=1; t<=5; t++)); do
    mps_alive "$state" || return 0
    sleep 2
  done
  echo "error: MPS control daemon still responding after quit ($state/mps/pipe)" >&2
  return 1
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
    if [ -d "$arg" ] && [ -f "$arg/replicas.tsv" ]; then echo "$arg"; return 0; fi
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
  local pgid out="" p st
  while IFS=$'\t' read -r _ _ pgid _ _; do
    for p in $(pgrep -g "$pgid" 2>/dev/null || true); do
      st=$(ps -o stat= -p "$p" 2>/dev/null || true)
      case "$st" in Z*|"") ;; *) out+=" $p" ;; esac
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
  local state=$1 keep=${2:-} pgid t live raw
  [ -n "$state" ] && [ -f "$state/replicas.tsv" ] || die "invalid or missing run state '$state'"
  while IFS=$'\t' read -r _ _ pgid _ _; do
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
  if mps_alive "$state"; then
    raw=$(mps_clients "$state") || { echo "error: MPS client query failed; state kept at $state" >&2; return 1; }
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
    mps_quit "$state" || { echo "state kept at $state" >&2; return 1; }
  fi
  live=$(tracked_pids "$state")
  if [ -n "${live// /}" ]; then
    echo "warning: tracked non-client processes survived TERM; last-resort SIGKILL on tracked groups only" >&2
    while IFS=$'\t' read -r _ _ pgid _ _; do
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
    rm -rf "$state"
    echo "down: run state $state cleaned; only this run's processes were touched"
  fi
}

up() {
  local model=${MODEL:-bosonai/higgs-tts-3-4b} model_name=${MODEL_NAME:-higgs}
  local gpu=${GPU_ID:-0} n=${N:-2} base_port=${BASE_PORT:-8801} mf=${MF:-}
  if [ -z "$mf" ]; then
    case "$n" in 2) mf=0.42 ;; 3) mf=0.27 ;; *) die "set MF explicitly for N=$n" ;; esac
  fi
  [ -n "${CORE_BLOCKS:-}" ] || {
    echo "CORE_BLOCKS is required: N non-overlapping blocks on the GPU's NUMA node." >&2
    echo "Cores on that node: numactl -H" >&2
    exit 1
  }
  local blocks=($CORE_BLOCKS)
  [ "${#blocks[@]}" = "$n" ] || die "CORE_BLOCKS must contain exactly $n blocks"

  local extra_args=()
  if [ -n "${MAX_TOTAL_TOKENS:-}" ]; then
    [[ "$MAX_TOTAL_TOKENS" =~ ^[1-9][0-9]*$ ]] \
      || die "MAX_TOTAL_TOKENS must be a positive integer, got '$MAX_TOTAL_TOKENS'"
    extra_args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
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
  {
    echo "run_id=$run"; echo "gpu_id=$gpu"; echo "gpu_uuid=$uuid"; echo "numa_node=$node"
    echo "model=$model"; echo "model_name=$model_name"; echo "n=$n"; echo "mf=$mf"
    echo "base_port=$base_port"; echo "core_blocks=$CORE_BLOCKS"
    echo "max_total_tokens=${MAX_TOTAL_TOKENS:-auto/profiled}"
  } > "$state/manifest"
  : > "$state/replicas.tsv"

  local up_done=0
  # Note (jiaxin): on startup failure only this run's processes are stopped, and
  # the state directory is kept so the failure can be diagnosed from its logs.
  trap '[ "$up_done" = 1 ] || { echo "startup failed; stopping this run only" >&2; teardown_state "'"$state"'" --keep-state || true; }' EXIT

  export CUDA_MPS_PIPE_DIRECTORY=$state/mps/pipe CUDA_MPS_LOG_DIRECTORY=$state/mps/log
  nvidia-cuda-mps-control -d 2>> "$state/mps_ctl.err" || true
  mps_alive "$state" || die "MPS control daemon did not start (pipe $state/mps/pipe; see $state/mps_ctl.err)"

  local pid log
  for ((i=0; i<n; i++)); do
    port=$((base_port+i))
    log=$state/logs/replica_$i.log
    # Note (jiaxin): concurrent colocated launches raced on CUDA-graph capture and
    # memory profiling in testing, so replicas start sequentially behind a health
    # gate; setsid gives each replica its own process group so teardown can signal
    # exactly this run's process trees.
    CUDA_VISIBLE_DEVICES=$gpu \
    setsid numactl --cpunodebind="$node" --membind="$node" -C "${blocks[$i]}" \
      sgl-omni serve --model-path "$model" --model-name "$model_name" \
        --mem-fraction-static "$mf" "${extra_args[@]}" \
        --host 127.0.0.1 --port "$port" > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$pid" "$pid" "$port" "$log" >> "$state/replicas.tsv"
    local healthy=0 t code
    for ((t=1; t<=HEALTH_TRIES; t++)); do
      if ! kill -0 "$pid" 2>/dev/null; then
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
    echo "replica $i healthy on port $port (cores ${blocks[$i]}, mf $mf)"
    # Note (jiaxin): per-replica KV pools are not additive shares of the device;
    # later replicas can receive much smaller pools, so surface each allocation.
    grep -m1 -oE '#tokens: [0-9]+' "$log" | sed "s/^/replica $i KV /" || true
  done

  verify_attach "$state" || exit 1
  if [ "$(cat "$state"/logs/replica_*.log 2>/dev/null | grep -c MpsRpc)" != 0 ]; then
    echo "warning: MpsRpc errors present in replica logs; bring the run down and restart" >&2
    exit 1
  fi
  up_done=1
  trap - EXIT
  echo "up: $n replicas on GPU $gpu; token cap ${MAX_TOTAL_TOKENS:-auto/profiled}; state: $state"
  echo "tear down with: bash $0 down $run"
}

case "$CMD" in
  up) up ;;
  down) st=$(resolve_state "$RUN_ARG") || exit 1; teardown_state "$st" ;;
  verify) st=$(resolve_state "$RUN_ARG") || exit 1; verify_attach "$st" ;;
  list) find_runs ;;
  *) die "usage: launch_same_gpu_dp.sh up|down [RUN_ID]|verify [RUN_ID]|list" ;;
esac
