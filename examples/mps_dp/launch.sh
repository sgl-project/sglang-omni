#!/bin/bash
# Launch N serving replicas on ONE GPU behind a private CUDA MPS daemon.
# Companion to docs/basic_usage/mps_dp.md.
# The optional lifecycle supervisor is intended for tested single-host
# deployments. It does not replace an external cluster orchestrator.
#
# Usage:
#   CONFIG=examples/mps_dp/configs/higgs_h100_dp3.yaml GPU_ID=0 N=3 \
#     CORE_BLOCKS="0-9 10-19 20-29" \
#     bash examples/mps_dp/launch.sh up
#   MODEL=bosonai/higgs-tts-3-4b GPU_ID=0 N=3 MAX_TOTAL_TOKENS=100000 \
#     CORE_BLOCKS="0-9 10-19 20-29" \
#     bash examples/mps_dp/launch.sh up
#   bash examples/mps_dp/launch.sh list
#   bash examples/mps_dp/launch.sh verify [RUN_ID] [REPLICA_INDICES]
#     REPLICA_INDICES: optional comma-separated subset to gate on. Replicas
#     outside the subset are still mapped into the artifact but cannot fail the
#     check, so a restart can be gated without members that are known dead.
#   bash examples/mps_dp/launch.sh down [RUN_ID]
#
# Environment for `up` (defaults in parentheses):
#   CONFIG: optional pipeline config. For N > 1, it must contain one SGLang
#     engine stage so the launcher can identify that stage's KV log. When unset,
#     MODEL is used.
#   MODEL (bosonai/higgs-tts-3-4b; unavailable with CONFIG),
#   MODEL_NAME (higgs without CONFIG; pipeline name with CONFIG), GPU_ID (0), N (3),
#   BASE_PORT (8801), PYTHON_BIN (python),
#   CORE_BLOCKS: N non-overlapping CPU blocks on the GPU's NUMA node, required.
#   NUMA_NODE: explicit override when the PCI-derived NUMA node is unavailable.
#   MAX_TOTAL_TOKENS: optional common positive token-cap override. For N > 1,
#     set it here or in CONFIG's generation-stage server arguments. The environment
#     value takes precedence when both are set.
#   MF: optional explicit --mem-fraction-static override (unset = pipeline default).
#   WEIGHT_SHARE (0): 1 = replicas share one copy of the AR backbone weights
#     over CUDA IPC. Requires CONFIG (the validated-config preflight runs
#     before any resource is created). Replica 0 is the weight LEADER (loads the checkpoint and
#     publishes IPC handles under $state/ipc_weights); replicas 1..N-1 attach
#     zero-copy instead of loading their own copy. The leader owns the shared
#     storage: if replica 0 dies, followers hold dangling mappings — always
#     bring the whole run down and restart it together (down + up).
#   ROUTER_PORT (8799), ROUTER_POLICY (least_request), ROUTER_ENABLED (1):
#     launch a local Omni Router over the replica pool after all startup gates.
#   SUPERVISE: restart failed replicas through the router/KV/MPS gates. Defaults
#     to 1 without weight sharing and 0 with WEIGHT_SHARE=1. Individual restart
#     is intentionally forbidden for a shared-weight group.
#   SUPERVISOR_INTERVAL (5), SUPERVISOR_FAILURE_THRESHOLD (3): runtime health
#     polling controls.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STATE_ROOT=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$UID}
PYTHON_BIN=${PYTHON_BIN:-python}
CMD=${1:-}
RUN_ARG=${2:-}
REPLICAS_ARG=${3:-}
HEALTH_TRIES=${HEALTH_TRIES:-50}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-6}
DRAIN_TRIES=${DRAIN_TRIES:-40}
DRAIN_INTERVAL=${DRAIN_INTERVAL:-3}
ROUTER_HEALTH_TRIES=${ROUTER_HEALTH_TRIES:-30}
ROUTER_HEALTH_INTERVAL=${ROUTER_HEALTH_INTERVAL:-1}
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

pid_identity_matches() {
  local pid=$1 expected_start=$2 actual_start
  pid_is_live "$pid" || return 1
  actual_start=$(pid_start_time "$pid") || return 1
  [ "$actual_start" = "$expected_start" ]
}

group_is_owned() {
  # note (Junnan Li): liveness is deliberately not required, because the leader
  # may already be a zombie. A zombie still occupies its PID, and a recorded
  # PGID *is* the leader's PID, so while that entry exists no unrelated process
  # can lead a group with this PGID. It is NOT sufficient by itself: once the
  # leader is reaped this returns 1 forever, which is what left orphaned groups
  # unsignallable. Callers must use group_ownership_proven, which also accepts
  # the two leaderless proofs below. Mirror of supervisor.py's
  # _record_group_is_owned.
  local pid=$1 expected_start=$2 expected_pgid=$3 actual_start actual_pgid
  actual_start=$(pid_start_time "$pid") || return 1
  [ "$actual_start" = "$expected_start" ] || return 1
  actual_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null) || return 1
  [ "${actual_pgid// /}" = "$expected_pgid" ]
}

pid_pgid() {
  local out
  out=$(ps -o pgid= -p "$1" 2>/dev/null) || return 1
  printf '%s\n' "${out// /}"
}

group_members_file() {
  # note (Junnan Li): the kind prefix is part of the key because replica indices
  # and service names share one namespace on disk; without it a service could
  # overwrite the membership proof of a replica.
  printf '%s/group-members/%s_%s.tsv\n' "$1" "$2" "$3"
}

persist_group_members() {
  # note (Junnan Li): the PGID goes on every row because the file name only
  # identifies the replica or service, and a list left over from an earlier
  # generation of that name must never be readable as proof about the current
  # one. The write is atomic so no reader can see a half-written list and
  # conclude the group is gone. Mirror of supervisor.py's
  # _persist_group_members.
  local file=$1 pgid=$2 tmp
  mkdir -p "$(dirname "$file")" || return 1
  tmp=$(mktemp "$(dirname "$file")/.$(basename "$file").XXXXXX") || return 1
  frozen_group_members "$pgid" | sed "s/^/$pgid\t/" > "$tmp"
  mv -f "$tmp" "$file"
}

persisted_group_members() {
  local file=$1 expected_pgid=$2 pgid p start
  [ -s "$file" ] || return 0
  while IFS=$'\t' read -r pgid p start; do
    [ "$pgid" = "$expected_pgid" ] || continue
    printf '%s\t%s\n' "$p" "$start"
  done < "$file"
}

members_prove_ownership() {
  # note (Junnan Li): the kernel cannot recycle a PGID while any process still
  # belongs to that group, so one persisted member that both matches its
  # identity AND is still in the recorded group proves the whole group is this
  # run's, with no reference to the leader at all. This is the proof that keeps
  # a reaped leader from making its own live group unsignallable.
  local file=$1 expected_pgid=$2 p start
  [ -s "$file" ] || return 1
  while IFS=$'\t' read -r p start; do
    pid_identity_matches "$p" "$start" || continue
    [ "$(pid_pgid "$p" || true)" = "$expected_pgid" ] || continue
    return 0
  done < <(persisted_group_members "$file" "$expected_pgid")
  return 1
}

group_token_matches() {
  # note (Junnan Li): independent second factor, needed because the persisted
  # list can be stale or unreadable — a live process in the recorded group
  # carrying this run's launch token belongs to this run whatever the process
  # table says about the leader. This is the identity check an operator
  # otherwise has to perform by hand.
  local pgid=$1 token=${2:-} p
  [ -n "$token" ] || return 1
  for p in $(pgrep -g "$pgid" 2>/dev/null || true); do
    pid_is_live "$p" || continue
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null \
      | grep -qxF "SGLANG_OMNI_MPS_DP_RUN_TOKEN=$token" && return 0
  done
  return 1
}

run_token() {
  local file=$1/run_token
  [ -r "$file" ] || return 0
  head -n 1 "$file" 2>/dev/null | tr -d '[:space:]'
}

group_ownership_proven() {
  # Usage: group_ownership_proven <state> <members-file> <pid> <start> <pgid>
  # note (Junnan Li): three states have to be told apart and only the last may
  # be treated as gone, or a live group loses its last signaller:
  #   zombie leader  -> group_is_owned (PID still taken, PGID still ours)
  #   reaped leader  -> a persisted member still in the group, or a token match
  #   all gone       -> no proof at all; the group no longer exists
  local state=$1 file=$2 pid=$3 start=$4 pgid=$5
  group_is_owned "$pid" "$start" "$pgid" && return 0
  members_prove_ownership "$file" "$pgid" && return 0
  group_token_matches "$pgid" "$(run_token "$state")"
}

frozen_group_members() {
  # note (Junnan Li): the caller must have just proven ownership of the group
  # (group_ownership_proven) — that is what rules out scanning a PGID that has
  # been recycled to a process this run never owned.
  local pgid=$1 p start
  for p in $(pgrep -g "$pgid" 2>/dev/null || true); do
    pid_is_live "$p" || continue
    start=$(pid_start_time "$p") || continue
    printf '%s\t%s\n' "$p" "$start"
  done
}

live_frozen_members() {
  local file=$1 out="" p start
  [ -s "$file" ] || { echo "$out"; return 0; }
  while IFS=$'\t' read -r p start; do
    pid_identity_matches "$p" "$start" && out+=" $p"
  done < "$file"
  echo "$out"
}

kill_frozen_members() {
  # note (Junnan Li): per-PID KILL rather than killpg, because by this point a
  # reaped leader's PID may have been reused by an unrelated process that now
  # leads a group carrying the same PGID number — a group-wide KILL could reach
  # a process this run never owned. Re-checking each frozen start time just
  # before the signal is the only check that excludes that.
  local file=$1 p start
  [ -s "$file" ] || return 0
  while IFS=$'\t' read -r p start; do
    pid_identity_matches "$p" "$start" && { kill -KILL "$p" 2>/dev/null || true; }
  done < "$file"
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
  # Note (Jiaxin Deng): /sys/class/drm ordinals are not guaranteed to match nvidia-smi
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
  # Note (Jiaxin Deng): zombies hold no resources and can never be reaped by this
  # script in init-less containers, so they do not count as live.
  local pgid out="" p
  while IFS=$'\t' read -r _ _ pgid _ _; do
    for p in $(pgrep -g "$pgid" 2>/dev/null || true); do
      pid_is_live "$p" && out+=" $p"
    done
  done < "$1/replicas.tsv"
  echo "$out"
}

tracked_service_pids() {
  # note (Junnan Li): a reaped leader must not hide its own live group from the
  # drain checks, so the persisted membership list is accepted as proof whenever
  # the leader identity is already gone — see members_prove_ownership.
  local state=$1 out="" p name pid pgid start
  [ -f "$state/services.tsv" ] || { echo "$out"; return 0; }
  while IFS=$'\t' read -r name pid pgid _ start; do
    pid_identity_matches "$pid" "$start" \
      || members_prove_ownership "$(group_members_file "$state" service "$name")" "$pgid" \
      || continue
    for p in $(pgrep -g "$pgid" 2>/dev/null || true); do
      pid_is_live "$p" && out+=" $p"
    done
  done < "$state/services.tsv"
  echo "$out"
}

run_is_active() {
  local state=$1 port live
  live=$(tracked_pids "$state")
  [ -n "${live// /}" ] && return 0
  live=$(tracked_service_pids "$state")
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
  # note (Junnan Li): the optional index subset scopes the pass/fail decision
  # only — every replica is still mapped into the artifact, because a member that
  # is killed, disabled, or mid-restart has no process group left to match an MPS
  # client and must not fail a check it structurally cannot pass.
  local state=$1 scope=${2:-}
  [ -n "$state" ] && [ -f "$state/replicas.tsv" ] || die "invalid or missing run state '$state'"
  local scoped=" "
  [ -n "$scope" ] && scoped=" $(echo "$scope" | tr ',' ' ') "
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
    if [ "$scoped" != " " ]; then
      case "$scoped" in
        *" $idx "*) ;;
        *) echo "replica $idx (port $port): out of verification scope" >> "$art"; continue;;
      esac
    fi
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

stop_services() {
  # note (Junnan Li): the supervisor must die before the router and replicas or
  # teardown races an automatic restart, which is why the rows are consumed in
  # reverse: services.tsv is written router-first, supervisor-last.
  local state=$1 name pid pgid log start t live members=$state/service-members.tsv
  [ -f "$state/services.tsv" ] || return 0
  : > "$members"
  while IFS=$'\t' read -r name pid pgid log start; do
    group_ownership_proven "$state" \
      "$(group_members_file "$state" service "$name")" "$pid" "$start" "$pgid" \
      || continue
    echo "stopping $name (pid $pid, log $log)"
    # note (Junnan Li): the persisted list is merged in because a live scan
    # misses a member that has just become a zombie, which must still be waited
    # on rather than assumed gone. Group-wide TERM is safe here only because
    # ownership was proven from a process still *in* the group — see
    # members_prove_ownership.
    frozen_group_members "$pgid" >> "$members"
    persisted_group_members \
      "$(group_members_file "$state" service "$name")" "$pgid" >> "$members"
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done < <(tac "$state/services.tsv")
  for ((t=1; t<=DRAIN_TRIES; t++)); do
    live="$(tracked_service_pids "$state")$(live_frozen_members "$members")"
    [ -z "${live// /}" ] && return 0
    sleep "$DRAIN_INTERVAL"
  done
  echo "warning: tracked router/supervisor services survived TERM; using SIGKILL on their recorded groups" >&2
  while IFS=$'\t' read -r name pid pgid _ start; do
    group_ownership_proven "$state" \
      "$(group_members_file "$state" service "$name")" "$pid" "$start" "$pgid" \
      || continue
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done < <(tac "$state/services.tsv")
  kill_frozen_members "$members"
  sleep 2
  live="$(tracked_service_pids "$state")$(live_frozen_members "$members")"
  [ -z "${live// /}" ] || {
    echo "error: tracked service pids still alive:$live" >&2
    return 1
  }
}

teardown_state() {
  # Note (Jiaxin Deng): these GPUs are shared; teardown only signals processes recorded
  # in this run's state, never scans the whole GPU, and keeps the state directory
  # whenever cleanup cannot be confirmed, so nothing is hidden from inspection.
  local state=$1 keep=${2:-} idx leader_pid pgid leader_start t live raw control_pid=""
  local members=$state/replica-members.tsv
  [ -n "$state" ] && [ -f "$state/replicas.tsv" ] || die "invalid or missing run state '$state'"
  stop_services "$state" || {
    echo "state kept at $state — refusing to stop replicas while their supervisor may still be active" >&2
    return 1
  }
  "$PYTHON_BIN" "$SCRIPT_DIR/supervisor.py" cleanup-pending --state "$state" || {
    echo "state kept at $state — pending replacement cleanup failed" >&2
    return 1
  }
  control_pid=$(mps_control_pid "$state" || true)
  : > "$members"
  while IFS=$'\t' read -r idx leader_pid pgid _ _ leader_start; do
    group_ownership_proven "$state" \
      "$(group_members_file "$state" replica "$idx")" \
      "$leader_pid" "$leader_start" "$pgid" || continue
    # note (Junnan Li): the freeze is not what makes the drain wait correct —
    # tracked_pids already counts every member of a recorded group, so the wait
    # cannot be satisfied by the leader alone. It exists so the last-resort KILL
    # has per-PID identities to re-verify, and the persisted list is merged in
    # because a live scan misses a member that has just become a zombie.
    frozen_group_members "$pgid" >> "$members"
    persisted_group_members \
      "$(group_members_file "$state" replica "$idx")" "$pgid" >> "$members"
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done < "$state/replicas.tsv"
  for ((t=1; t<=DRAIN_TRIES; t++)); do
    live=$(tracked_pids "$state")
    [ -z "${live// /}" ] && break
    sleep "$DRAIN_INTERVAL"
  done
  # Note (Jiaxin Deng): the pipe is private to this run, so ANY client the daemon still
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
  live="$(tracked_pids "$state")$(live_frozen_members "$members")"
  if [ -n "${live// /}" ]; then
    echo "warning: tracked non-client processes survived TERM; last-resort SIGKILL on tracked groups only" >&2
    while IFS=$'\t' read -r idx leader_pid pgid _ _ leader_start; do
      group_ownership_proven "$state" \
        "$(group_members_file "$state" replica "$idx")" \
        "$leader_pid" "$leader_start" "$pgid" || continue
      kill -KILL -- "-$pgid" 2>/dev/null || true
    done < "$state/replicas.tsv"
    kill_frozen_members "$members"
    sleep 2
  fi
  live="$(tracked_pids "$state")$(live_frozen_members "$members")"
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
  local config=${CONFIG:-} model=${MODEL:-bosonai/higgs-tts-3-4b}
  local model_name=${MODEL_NAME:-}
  local gpu=${GPU_ID:-0} n=${N:-3} base_port=${BASE_PORT:-8801} mf=${MF:-}
  local weight_share=${WEIGHT_SHARE:-0}
  local router_enabled=${ROUTER_ENABLED:-1}
  local router_port=${ROUTER_PORT:-8799}
  local router_policy=${ROUTER_POLICY:-least_request}
  local supervise=${SUPERVISE:-}
  if [ -z "$supervise" ]; then
    if [ "$weight_share" = 1 ]; then supervise=0; else supervise=1; fi
  fi
  [[ "$weight_share" =~ ^[01]$ ]] || die "WEIGHT_SHARE must be 0 or 1, got '$weight_share'"
  [[ "$router_enabled" =~ ^[01]$ ]] || die "ROUTER_ENABLED must be 0 or 1, got '$router_enabled'"
  [[ "$supervise" =~ ^[01]$ ]] || die "SUPERVISE must be 0 or 1, got '$supervise'"
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative integer, got '$gpu'"
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "N must be a positive integer, got '$n'"
  [[ "$base_port" =~ ^[1-9][0-9]*$ ]] \
    || die "BASE_PORT must be a positive integer, got '$base_port'"
  [[ "$router_port" =~ ^[1-9][0-9]*$ ]] \
    || die "ROUTER_PORT must be a positive integer, got '$router_port'"
  ((base_port + n - 1 <= 65535)) \
    || die "ports $base_port through $((base_port+n-1)) exceed 65535"
  ((router_port <= 65535)) || die "ROUTER_PORT must not exceed 65535"
  if [ "$router_enabled" = 1 ] && ((router_port >= base_port && router_port < base_port+n)); then
    die "ROUTER_PORT $router_port overlaps replica ports $base_port through $((base_port+n-1))"
  fi
  [ "$supervise" = 0 ] || [ "$router_enabled" = 1 ] \
    || die "SUPERVISE=1 requires ROUTER_ENABLED=1"
  [ "$supervise" = 0 ] || [ "$weight_share" = 0 ] \
    || die "SUPERVISE=1 is unsafe with WEIGHT_SHARE=1; restart the whole shared-weight run"
  case "$router_policy" in
    round_robin|least_request|random) ;;
    *) die "ROUTER_POLICY must be round_robin, least_request, or random, got '$router_policy'" ;;
  esac
  [ -n "${CORE_BLOCKS:-}" ] || {
    echo "CORE_BLOCKS is required: N non-overlapping blocks on the GPU's NUMA node." >&2
    echo "Cores on that node: numactl -H" >&2
    exit 1
  }
  local blocks=()
  read -r -a blocks <<< "$CORE_BLOCKS"
  [ "${#blocks[@]}" = "$n" ] || die "CORE_BLOCKS must contain exactly $n blocks"

  local serve_cmd=(sgl-omni serve) source_args=() model_name_args=()
  local extra_args=() mem_args=()
  local expected_max_total_tokens=${MAX_TOTAL_TOKENS:-}
  local model_path_manifest=$model
  if [ -n "$config" ]; then
    [ -z "${MODEL:-}" ] || die "MODEL cannot be combined with CONFIG"
    [ -f "$config" ] || die "config file not found: $config"
    config=$(cd -- "$(dirname -- "$config")" && pwd)/$(basename -- "$config")
    serve_cmd=("$PYTHON_BIN" -m sglang_omni.cli serve)
    source_args=(--config "$config")
    model_path_manifest=from_config
    if [ -n "$model_name" ]; then
      model_name_args=(--model-name "$model_name")
    fi
    local config_resolver_args=("$config")
    if [ -n "$expected_max_total_tokens" ]; then
      config_resolver_args+=(--max-total-tokens "$expected_max_total_tokens")
    fi
    if [ "$n" -gt 1 ]; then
      config_resolver_args+=(--require-single-sglang-engine)
    fi
    if [ "$weight_share" = 1 ]; then
      config_resolver_args+=(--weight-share)
    fi
    expected_max_total_tokens=$("$PYTHON_BIN" "$SCRIPT_DIR/config.py" \
      "${config_resolver_args[@]}") \
      || die "could not resolve max_total_tokens from $config"
  else
    # Note (Jiaxin Deng): without a pipeline config the supported-model check
    # cannot run until engine startup, which is after the MPS daemon and state
    # dir exist; sharing therefore requires CONFIG so unsupported models are
    # rejected before any resource is created.
    [ "$weight_share" = 1 ] \
      && die "WEIGHT_SHARE=1 requires CONFIG (support is checked per pipeline config before any resource is created)"
    source_args=(--model-path "$model")
    model_name=${MODEL_NAME:-higgs}
    model_name_args=(--model-name "$model_name")
  fi
  if [ "$n" -gt 1 ] && [ -z "$expected_max_total_tokens" ]; then
    die "MAX_TOTAL_TOKENS is required for N=$n so every replica has the same KV capacity"
  fi
  if [ -n "$expected_max_total_tokens" ]; then
    [[ "$expected_max_total_tokens" =~ ^[1-9][0-9]*$ ]] \
      || die "max_total_tokens must be a positive integer, got '$expected_max_total_tokens'"
  fi
  if [ -n "${MAX_TOTAL_TOKENS:-}" ]; then
    extra_args+=(--max-total-tokens "$expected_max_total_tokens")
  fi
  if [ -n "${SERVE_EXTRA_ARGS:-}" ]; then
    # Extra sgl-omni serve flags, word-split intentionally (e.g.
    # "--max-running-requests 32"). Applied identically to every replica.
    # shellcheck disable=SC2206
    extra_args+=($SERVE_EXTRA_ARGS)
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
  if [ "$router_enabled" = 1 ] \
    && (exec 3<> "/dev/tcp/127.0.0.1/$router_port") 2>/dev/null; then
    exec 3>&- 3<&-
    die "router port $router_port is already in use; pick another ROUTER_PORT"
  fi

  local uuid node run state driver_version host_timezone started_at token
  # Note (Jiaxin Deng): a caller (autodp) may pin RUN_ID so it can tear down exactly
  # the run it started, instead of rediscovering the newest dir.
  # Note (Yueying Li): RUN_ID becomes a single directory component under
  # gpu-$gpu; a separator or traversal sequence would relocate run state into
  # another GPU's namespace (or out of STATE_ROOT) and bypass the
  # active/stale-run guards above, so restrict it to a run-* basename before
  # probing hardware or creating any resource.
  run="${RUN_ID:-run-$(date +%Y%m%d-%H%M%S)-$$}"
  [[ "$run" =~ ^run-[A-Za-z0-9_-]+$ ]] \
    || die "RUN_ID must be a single 'run-<suffix>' path component ([A-Za-z0-9_-]), got '$run'"
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")
  driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$gpu")
  host_timezone=$(date +'%Z %:z')
  started_at=$(date -Is)
  node=$(resolve_numa "$gpu")
  state=$STATE_ROOT/gpu-$gpu/$run
  mkdir -p "$state/logs" "$state/mps/pipe" "$state/mps/log"
  : > "$state/replicas.tsv"
  : > "$state/services.tsv"
  printf '[]\n' > "$state/replica_specs.json"
  mkdir -p "$state/group-members"
  # note (Junnan Li): the per-run token exists because it is the one ownership
  # proof that needs neither the recorded leader nor a frozen membership list —
  # both can be gone while the group itself is still alive. It is readable back
  # off any member through /proc/<pid>/environ.
  token=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
  [ -n "$token" ] || die "could not generate a run token"
  printf '%s\n' "$token" > "$state/run_token"
  chmod 600 "$state/run_token"

  # note(ratish): Without this trap, a later replica failure leaves earlier
  # replicas and the private MPS daemon running; keep the state for diagnosis.
  startup_state=$state
  trap cleanup_failed_startup EXIT

  chmod 700 "$state/mps" "$state/mps/pipe" "$state/mps/log"
  {
    echo "run_id=$run"; echo "gpu_id=$gpu"; echo "gpu_uuid=$uuid"; echo "numa_node=$node"
    echo "driver_version=$driver_version"; echo "host_timezone=$host_timezone"
    echo "started_at=$started_at"
    echo "config=${config:-none}"; echo "model_path=$model_path_manifest"
    echo "model_name=${model_name:-from_config}"; echo "n=$n"
    echo "mem_fraction_static_cli_override=${mf:-none}"
    echo "base_port=$base_port"; echo "core_blocks=$CORE_BLOCKS"
    echo "max_total_tokens=${expected_max_total_tokens:-auto/profiled}"
    echo "weight_share=$weight_share"
    echo "router_enabled=$router_enabled"; echo "router_port=$router_port"
    echo "router_policy=$router_policy"; echo "supervise=$supervise"
  } > "$state/manifest"
  if [ "$weight_share" = 1 ]; then mkdir -p "$state/ipc_weights"; chmod 700 "$state/ipc_weights"; fi

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

  local pid leader_start log resolved_tokens ws_env
  for ((i=0; i<n; i++)); do
    port=$((base_port+i))
    log=$state/logs/replica_$i.log
    # Note (Jiaxin Deng): replica 0 leads (loads + exports IPC handles); later
    # replicas attach. The sequential health gate below already guarantees the
    # leader has exported (export completes during model load, well before
    # /health turns 200) by the time any follower boots, so followers never
    # block on the handle file in this launcher. Empty value = feature off.
    ws_env=""
    if [ "$weight_share" = 1 ]; then
      if [ "$i" = 0 ]; then ws_env="leader:$state/ipc_weights"
      else ws_env="follower:$state/ipc_weights"; fi
    fi
    # Note (Jiaxin Deng): concurrent colocated launches raced on CUDA-graph capture and
    # memory profiling in testing, so replicas start sequentially behind a health
    # gate; setsid gives each replica its own process group so teardown can signal
    # exactly this run's process trees.
    local replica_cmd=(
      numactl "--cpunodebind=$node" "--membind=$node" -C "${blocks[$i]}"
      "${serve_cmd[@]}" "${source_args[@]}" "${model_name_args[@]}" \
        "${mem_args[@]}" "${extra_args[@]}" \
        --host 127.0.0.1 --port "$port"
    )
    local spec_args=(
      add-replica --state "$state" --index "$i" --port "$port" --log "$log"
      --cwd "$PWD"
      --env "CUDA_VISIBLE_DEVICES=$uuid"
      --env "CUDA_MPS_PIPE_DIRECTORY=$state/mps/pipe"
      --env "CUDA_MPS_LOG_DIRECTORY=$state/mps/log"
      --env "SGLANG_OMNI_WEIGHT_SHARE=$ws_env"
      --env "SGLANG_OMNI_WEIGHT_SHARE_RUN_ID=$run"
      --env "SGLANG_OMNI_STRICT_PORT=1"
      --env "SGLANG_OMNI_MPS_DP_RUN_TOKEN=$token"
    )
    if [ -n "$expected_max_total_tokens" ]; then
      spec_args+=(--expected-tokens "$expected_max_total_tokens")
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/supervisor.py" \
      "${spec_args[@]}" -- "${replica_cmd[@]}"
    CUDA_VISIBLE_DEVICES="$uuid" \
    SGLANG_OMNI_WEIGHT_SHARE="$ws_env" \
    SGLANG_OMNI_WEIGHT_SHARE_RUN_ID="$run" \
    SGLANG_OMNI_STRICT_PORT=1 \
    SGLANG_OMNI_MPS_DP_RUN_TOKEN="$token" \
    setsid "${replica_cmd[@]}" > "$log" 2>&1 < /dev/null &
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
    # note (Junnan Li): frozen here rather than at spawn time because only the
    # leader exists then, and the group is fully populated only once the health
    # gate passes. With SUPERVISE=0 nothing ever refreshes this list, so this
    # snapshot is the run's only leaderless ownership proof.
    persist_group_members "$(group_members_file "$state" replica "$i")" "$pid" \
      || die "replica $i is healthy but its group membership could not be recorded"
    echo "replica $i healthy on port $port (cores ${blocks[$i]})"
    resolved_tokens=""
    resolved_tokens=$(grep -m1 -oE '#tokens:[[:space:]]*[0-9]+' "$log" \
      | grep -oE '[0-9]+$' || true)
    if [ "$n" -gt 1 ]; then
      [ -n "$resolved_tokens" ] \
        || die "replica $i is healthy but its resolved KV capacity is missing from $log"
      [ "$resolved_tokens" = "$expected_max_total_tokens" ] \
        || die "replica $i resolved $resolved_tokens KV tokens; expected $expected_max_total_tokens"
    fi
    echo "replica $i KV #tokens: ${resolved_tokens:-not found}"
  done

  verify_attach "$state" || exit 1
  if [ "$(cat "$state"/logs/replica_*.log 2>/dev/null | grep -c MpsRpc)" != 0 ]; then
    echo "warning: MpsRpc errors present in replica logs; bring the run down and restart" >&2
    exit 1
  fi

  local router_url=""
  if [ "$router_enabled" = 1 ]; then
    router_url="http://127.0.0.1:$router_port"
    local worker_urls=()
    for ((i=0; i<n; i++)); do
      worker_urls+=("http://127.0.0.1:$((base_port+i))")
    done
    local router_log=$state/logs/router.log
    setsid "$PYTHON_BIN" -m sglang_omni_router.serve \
      --host 127.0.0.1 --port "$router_port" \
      --worker-urls "${worker_urls[@]}" \
      --policy "$router_policy" \
      --health-failure-threshold 1 \
      --health-success-threshold 1 \
      --health-check-interval-secs 2 \
      --health-check-timeout-secs 2 \
      > "$router_log" 2>&1 < /dev/null &
    pid=$!
    leader_start=$(pid_start_time "$pid") \
      || die "router exited before its process identity could be recorded"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      router "$pid" "$pid" "$router_log" "$leader_start" >> "$state/services.tsv"
    local router_ready=0
    for ((t=1; t<=ROUTER_HEALTH_TRIES; t++)); do
      if ! pid_is_live "$pid"; then
        echo "router exited during startup; last log lines:" >&2
        tail -n 12 "$router_log" >&2
        exit 1
      fi
      code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$router_url/ready" || true)
      [ "$code" = 200 ] && { router_ready=1; break; }
      sleep "$ROUTER_HEALTH_INTERVAL"
    done
    [ "$router_ready" = 1 ] || {
      echo "router did not report a routable worker; last log lines:" >&2
      tail -n 12 "$router_log" >&2
      exit 1
    }
    persist_group_members "$(group_members_file "$state" service router)" "$pid" \
      || die "router is healthy but its group membership could not be recorded"
    echo "router healthy on $router_url (policy $router_policy)"
  fi

  if [ "$supervise" = 1 ]; then
    local supervisor_log=$state/logs/supervisor.log
    setsid "$PYTHON_BIN" "$SCRIPT_DIR/supervisor.py" run \
      --state "$state" \
      --launch-script "$SCRIPT_DIR/launch.sh" \
      --router-url "$router_url" \
      --interval-secs "${SUPERVISOR_INTERVAL:-5}" \
      --health-failure-threshold "${SUPERVISOR_FAILURE_THRESHOLD:-3}" \
      --health-tries "$HEALTH_TRIES" \
      --health-interval-secs "$HEALTH_INTERVAL" \
      > "$supervisor_log" 2>&1 < /dev/null &
    pid=$!
    leader_start=$(pid_start_time "$pid") \
      || die "supervisor exited before its process identity could be recorded"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      supervisor "$pid" "$pid" "$supervisor_log" "$leader_start" >> "$state/services.tsv"
    sleep 1
    pid_is_live "$pid" || {
      echo "supervisor exited during startup; last log lines:" >&2
      tail -n 12 "$supervisor_log" >&2
      exit 1
    }
    persist_group_members "$(group_members_file "$state" service supervisor)" "$pid" \
      || die "supervisor is running but its group membership could not be recorded"
    echo "replica supervisor active (interval ${SUPERVISOR_INTERVAL:-5}s, failure threshold ${SUPERVISOR_FAILURE_THRESHOLD:-3})"
  fi

  trap - EXIT
  echo "up: $n replicas on GPU $gpu; token cap ${expected_max_total_tokens:-auto/profiled}; weight_share=$weight_share; state: $state"
  if [ "$router_enabled" = 1 ]; then
    echo "shared ingress: $router_url (policy $router_policy)"
  fi
  if [ "$weight_share" = 1 ]; then
    echo "weight sharing is ON: replica 0 owns the shared weights — never restart replicas individually; use down + up"
  fi
  echo "tear down with: bash $0 down $run"
}

case "$CMD" in
  up) up ;;
  down) st=$(resolve_state "$RUN_ARG") || exit 1; teardown_state "$st" ;;
  verify) st=$(resolve_state "$RUN_ARG") || exit 1; verify_attach "$st" "$REPLICAS_ARG" ;;
  list) find_runs ;;
  *) die "usage: launch.sh up|down [RUN_ID]|verify [RUN_ID] [REPLICA_INDICES]|list" ;;
esac
