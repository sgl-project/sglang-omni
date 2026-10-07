#!/bin/bash
# gate.sh: MiniCPM-o duplex gates. See README.md in this directory.
#
# Local (inside a git checkout; NODE=<ssh destination>):
#   gate.sh push <tag>:<sha>...          copy this directory and `git archive` of each tree (and $BENCH_TREE) to the node
#   gate.sh pull <name> [dest]           copy a results directory back without recorder traces (TRACES=1 keeps them; large)
# Node host (each command starts detached work in the containers and returns; results in $OMNI_ROOT/logs/gates/<name>):
#   gate.sh unit <tree> [--gpu] [--full]
#   gate.sh ladder <tree>[,<tree>...] <sessions> <runs> [--speech tuned|default] [--samples N] [--frames-per-unit N] [--no-warm] [--name NAME]
#   gate.sh perception <tree>[,<tree>...] [sessions=48] [runs=3]
#   gate.sh serving <tree> <c-list> <reps> [--graphs] [--card I]
#   gate.sh agree <tree> [--card I]
#   gate.sh sweep <tree> [--sessions 1,2,4,8,16,32,48,64] [--runs 2] [--runs-above-target R] [--miss-threshold 1.0] [--stop-miss 30] [--frames-per-unit N]
#   gate.sh status <name> | report <name> | stop
# A tree is <tag>:<sha> of a tree pushed with `gate.sh push`; the sha may be abbreviated.
# Environment: HOSTLOAD=1 (host CPU sampler next to each ladder run), WAVE_GAP (s between two cards' client waves, 30),
# STAGE_TIMING_LIGHT=1 (hook without synchronization), MPS=0 (ladder without MPS), plus everything in config.sh.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=config.sh source-path=SCRIPTDIR
source "$HERE/config.sh"
read -r -a CARD_LIST <<< "$CARDS"
read -r -a CONTAINER_LIST <<< "$CONTAINERS"
read -r -a CPUS_LIST <<< "$CLIENT_CPUS_LIST"
TUNED_SPEECH='{"dtype": "float32", "enable_dit_torch_compile": true, "n_timesteps": 5}'
WAVE_GAP=${WAVE_GAP:-30}
SAMPLES_PER_SESSION=2
FRAMES=0     # JPEG frames attached to every unit by the recorders (frame_hook/); set by --frames-per-unit
RUNS_ABOVE=  # sweep: runs per level above the first unsolved level (empty: all); set by --runs-above-target
die() { echo "gate.sh: $*" >&2; exit 1; }
stamp() { date -u +%T; }

# ---------------------------------------------------------------- local side

push() {
  [ -n "${NODE:-}" ] || die "set NODE to the ssh destination of the node"
  local remote=${NODE_OMNI_ROOT:-omni} tmp spec tag sha
  tmp=$(mktemp -d)
  # shellcheck disable=SC2029  # the paths are meant to expand locally
  ssh "$NODE" "mkdir -p $remote/src/heads $remote/src/harness $remote/logs/gates" || die "ssh $NODE"
  rsync -a --delete --exclude __pycache__ "$HERE/" "$NODE:$remote/src/harness/" || die "rsync harness"
  for spec in "$@" "$BENCH_TREE"; do
    tag=${spec%%:*}
    sha=$(git rev-parse --verify "${spec#*:}^{commit}") || die "unknown commit in $spec"
    # shellcheck disable=SC2029
    if [ "$(ssh "$NODE" "cat $remote/src/heads/$tag.SHA 2>/dev/null")" = "$sha" ]; then
      echo "$tag $sha already on the node"
      continue
    fi
    git archive --format=tar.gz -o "$tmp/$tag.tgz" "$sha" || die "git archive $sha"
    echo "$sha" > "$tmp/$tag.SHA"
    scp -q "$tmp/$tag.tgz" "$tmp/$tag.SHA" "$NODE:$remote/src/heads/" || die "scp $tag"
    echo "$tag $sha pushed"
  done
  rm -rf "$tmp"
}

pull() {
  [ -n "${NODE:-}" ] || die "set NODE to the ssh destination of the node"
  local name=${1:?results name} dest=${2:-gate-results}
  mkdir -p "$dest"
  local -a exclude=(--exclude '*.wav' --exclude '*.pcm' --exclude '*.npy')
  [ "${TRACES:-0}" = 1 ] || exclude+=(--exclude 'samples/')
  rsync -a "${exclude[@]}" "$NODE:${NODE_OMNI_ROOT:-omni}/logs/gates/$name" "$dest/"
}

# ---------------------------------------------------------------- node host side

resolve_tree() { # <tag>:<sha> -> "<tag> <full sha>" of a pushed tree
  local tag=${1%%:*} want=${1#*:} have
  [ "$tag" != "$1" ] || die "tree must be <tag>:<sha>, got $1"
  have=$(cat "$OMNI_ROOT/src/heads/$tag.SHA" 2>/dev/null) || die "tree $tag not pushed (gate.sh push $1)"
  [ "${have#"$want"}" != "$have" ] || die "pushed $tag is $have, not $want"
  echo "$tag $have"
}

new_results() { # <name> <kind> <args...>: create the results directory and its env file, print its container path
  local name=$1 kind=$2 dir
  shift 2
  dir=$OMNI_ROOT/logs/gates/$name
  [ -e "$dir" ] && die "results $dir exist; pass another --name"
  mkdir -p "$dir"
  {
    for var in CARDS CONTAINERS CLIENT_CPUS_LIST UNIT_CPUS BENCH_TREE CLIENT_PROFILE MODEL_REPO MODEL_REVISION MODEL_NAME FDB_DIGEST WAVE_GAP; do
      printf '%s=%q\n' "$var" "${!var}"
    done
    printf '%s=%q\n' HOSTLOAD "${HOSTLOAD:-0}" STAGE_TIMING_LIGHT "${STAGE_TIMING_LIGHT:-0}" MPS "${MPS:-1}"
  } > "$dir/env.sh"
  printf '%s\n' "kind=$kind" "started=$(date -u +%FT%TZ)" "args=$*" > "$dir/gate.txt"
  echo "$RESULTS_ROOT/$name"
}

exec_detached() { # <slot> <args...>: run gate.sh <args> inside the container of card slot <slot>
  local slot=$1
  shift
  podman exec -d "${CONTAINER_LIST[$slot]}" bash "$HARNESS_DIR/gate.sh" "$@" > /dev/null
}

default_name() { date -u +"$1-$2-%m%d-%H%M"; }

ladder_like() { # <kind> <trees> <sessions-list> <runs> <speech> <samples> <warm> <name> <miss-threshold> <stop-miss>
  local kind=$1 trees=$2 levels=$3 runs=$4 speech=$5 samples=$6 warm=$7 name=$8 threshold=$9 stop_miss=${10}
  local results spec run level tag sha label n queue=() slot resolved
  local -a specs
  IFS=, read -r -a specs <<< "$trees"
  for level in ${levels//,/ }; do
    for run in $(seq 1 "$runs"); do
      for spec in "${specs[@]}"; do
        resolved=$(resolve_tree "$spec") || exit 1
        read -r tag sha <<< "$resolved"
        label=$tag-n$level-r$run
        [ "$FRAMES" -gt 0 ] && label=$tag-f$FRAMES-n$level-r$run
        n=$samples
        [ "$n" = auto ] && n=$((level * SAMPLES_PER_SESSION < 16 ? 16 : (level * SAMPLES_PER_SESSION > 96 ? 96 : level * SAMPLES_PER_SESSION)))
        queue+=("$label $tag $sha $level $speech $n $FRAMES $run")
      done
    done
  done
  results=$(new_results "$name" "$kind" "$trees sessions=$levels runs=$runs speech=$speech threshold=$threshold stop_miss=$stop_miss") || exit 1
  printf '%s\n' "${queue[@]}" > "$OMNI_ROOT/logs/gates/$name/queue"
  printf '%s\n' "threshold=$threshold" "stop_miss=$stop_miss" "speech=$speech" "frames=$FRAMES" "runs=$runs" "runs_above=$RUNS_ABOVE" "workers=${#CARD_LIST[@]}" "warm_tree=${queue[0]}" >> "$OMNI_ROOT/logs/gates/$name/gate.txt"
  for slot in "${!CARD_LIST[@]}"; do
    exec_detached "$slot" run-worker "$results" "$slot" "$warm" || die "start worker on ${CONTAINER_LIST[$slot]}"
  done
  echo "started $kind: ${#queue[@]} runs on cards ${CARD_LIST[*]}; results $OMNI_ROOT/logs/gates/$name (DONE when finished, report.md)"
}

cmd_ladder() {
  local trees=${1:?tree} sessions=${2:?sessions} runs=${3:?runs} speech=tuned samples="" warm=1 name=""
  shift 3
  while [ $# -gt 0 ]; do
    case $1 in
      --speech) speech=$2; shift 2 ;;
      --samples) samples=$2; shift 2 ;;
      --frames-per-unit) FRAMES=$2; shift 2 ;;
      --no-warm) warm=0; shift ;;
      --name) name=$2; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  # 16 sessions over 48 samples and 32 or more over 96 are the conventions of the earlier gates
  [ -n "$samples" ] || samples=$((sessions > 16 ? 96 : 48))
  ladder_like ladder "$trees" "$sessions" "$runs" "$speech" "$samples" "$warm" "${name:-$(default_name ladder "${trees%%:*}")}" 1.0 101
}

cmd_perception() {
  local trees=${1:?tree} sessions=${2:-48} runs=${3:-3}
  ladder_like perception "$trees" "$sessions" "$runs" default 96 1 "$(default_name perception "${trees%%:*}")" 1.0 101
}

cmd_sweep() {
  local tree=${1:?tree} levels=1,2,4,8,16,32,48,64 runs=2 threshold=1.0 stop_miss=30 warm=1 name=""
  shift
  while [ $# -gt 0 ]; do
    case $1 in
      --sessions) levels=$2; shift 2 ;;
      --runs) runs=$2; shift 2 ;;
      --miss-threshold) threshold=$2; shift 2 ;;
      --stop-miss) stop_miss=$2; shift 2 ;;
      --frames-per-unit) FRAMES=$2; shift 2 ;;
      --runs-above-target) RUNS_ABOVE=$2; shift 2 ;;
      --no-warm) warm=0; shift ;;
      --name) name=$2; shift 2 ;;
      *) die "unknown option $1" ;;
    esac
  done
  [ "${tree//,/}" = "$tree" ] || die "sweep takes one tree"
  levels=$(tr ',' '\n' <<< "$levels" | sort -n | paste -sd,)
  ladder_like sweep "$tree" "$levels" "$runs" tuned auto "$warm" "${name:-$(default_name sweep "${tree%%:*}")}" "$threshold" "$stop_miss"
}

cmd_single() { # unit | serving | agree: one container
  local kind=$1 spec=$2 slot=0 name="" tag sha results resolved
  shift 2
  local -a rest=()
  while [ $# -gt 0 ]; do
    case $1 in
      --card) slot=$2; shift 2 ;;
      --name) name=$2; shift 2 ;;
      *) rest+=("$1"); shift ;;
    esac
  done
  resolved=$(resolve_tree "$spec") || exit 1
  read -r tag sha <<< "$resolved"
  results=$(new_results "${name:-$(default_name "$kind" "$tag")}" "$kind" "$spec ${rest[*]}") || exit 1
  exec_detached "$slot" "run-$kind" "$results" "$slot" "$tag" "$sha" "${rest[@]}" || die "start $kind"
  echo "started $kind on ${CONTAINER_LIST[$slot]}; results $OMNI_ROOT/logs/gates/${results##*/} (DONE when finished, report.md)"
}

cmd_status() {
  local dir=$OMNI_ROOT/logs/gates/${1:?name}
  cat "$dir/gate.txt" "$dir/chain.log" 2>/dev/null
  [ -f "$dir/DONE" ] && echo DONE || echo "running ($(wc -l < "$dir/queue" 2>/dev/null || echo 0) queued)"
}

cmd_stop() {
  local c
  for c in "${CONTAINER_LIST[@]}"; do podman exec "$c" bash "$HARNESS_DIR/gate.sh" run-stop; done
}

# ---------------------------------------------------------------- container side

load_results_env() { # <results>
  RESULTS=$1
  # shellcheck source=/dev/null
  source "$RESULTS/env.sh"
  read -r -a CARD_LIST <<< "$CARDS"
  read -r -a CPUS_LIST <<< "$CLIENT_CPUS_LIST"
  BENCH_TAG=${BENCH_TREE%%:*}
  BENCH_SHA=${BENCH_TREE#*:}
}

prep_tree() { # <tag> <sha>: extract the pushed tree to /work-<tag> unless it is already there
  local dir=/work-$1
  [ "$(cat "$dir/SHA" 2>/dev/null)" = "$2" ] && return 0
  [ "$(cat "$HEADS_DIR/$1.SHA" 2>/dev/null)" = "$2" ] || { echo "pushed $1 is not $2"; return 1; }
  (
    exec 8> "/tmp/prep-$1.lock"
    flock 8
    [ "$(cat "$dir/SHA" 2>/dev/null)" = "$2" ] && exit 0
    rm -rf "$dir" && mkdir -p "$dir" && tar -xzf "$HEADS_DIR/$1.tgz" -C "$dir" && echo "$2" > "$dir/SHA"
  )
}

write_yaml() { # <tree> <out> <max_sessions> <speech tuned|default> <graphs 1|0>
  MODEL_DIR=$MODEL_DIR TUNED_SPEECH=$TUNED_SPEECH python - "$@" << 'PY'
import json
import os
import sys

import yaml

tree, out, sessions, speech, graphs = sys.argv[1:]
config = yaml.safe_load(open(f"{tree}/examples/full_duplex/minicpmo-parity.yaml"))
config["model_path"] = os.environ["MODEL_DIR"]
config["max_sessions"] = int(sessions)
if graphs == "1":
    for stage in ("thinker", "talker"):
        engine = config["stages"].get(stage, {}).get("engine")
        if engine is not None:
            engine.pop("disable_cuda_graph", None)
            if not engine:
                config["stages"][stage].pop("engine")
if speech == "tuned":
    config["speech"] = json.loads(os.environ["TUNED_SPEECH"])
yaml.safe_dump(config, open(out, "w"), sort_keys=False)
PY
}

start_server() { # <tree> <yaml> <port> <log> <timing-dir or ""> -> prints seconds to ready; fails on a crash or after 30 min
  local tree=$1 yaml=$2 port=$3 log=$4 timing=$5 t0 pythonpath=$1
  local -a hook_env=()
  if [ -n "$timing" ]; then
    pythonpath=$HARNESS_DIR/timing_hook:$tree
    hook_env=("STAGE_TIMING_DIR=$timing" "STAGE_TIMING_LIGHT=${STAGE_TIMING_LIGHT:-0}")
  fi
  t0=$(date +%s)
  # the whole background group writes to the log, so it does not hold the caller's $(...) pipe open
  (cd "$tree" && exec env HF_HUB_OFFLINE=1 PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES=0 "${hook_env[@]}" \
    setsid nohup python -m sglang_omni.cli serve --config "$yaml" --enable-realtime --host 127.0.0.1 --port "$port") > "$log" 2>&1 < /dev/null &
  for _ in $(seq 1 600); do
    if curl -sf -m 3 "http://127.0.0.1:$port/v1/realtime/capabilities" > /dev/null; then
      echo $(($(date +%s) - t0))
      return 0
    fi
    grep -qE "Traceback|died during startup" "$log" && return 1
    sleep 3
  done
  return 1
}

stop_server() { # <yaml>: kill the server's session (main process and stage processes)
  local pid sid
  for pid in $(pgrep -f "[s]glang_omni.cli serve --config $1"); do
    sid=$(ps -o sid= -p "$pid" | tr -d ' ')
    [ -n "$sid" ] && pkill -9 -s "$sid"
  done
  sleep 8
}

mps_start() {
  [ "${MPS:-1}" = 1 ] || return 0
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d
  sleep 3
}

mps_stop() {
  [ "${MPS:-1}" = 1 ] || return 0
  echo quit | nvidia-cuda-mps-control
}

record() { # <out> <port> <tree-sha> <cpus> <frames per unit> <dataset-root> <sample-id>...: one recorder client, one thread, pinned
  local out=$1 port=$2 sha=$3 cpus=$4 frames=$5 root=$6 id
  shift 6
  local -a ids=()
  for id in "$@"; do ids+=(--sample-id "$id"); done
  (cd /work-bench && env PYTHONPATH="$HARNESS_DIR/frame_hook:/work-bench" DUPLEX_FRAMES_PER_UNIT="$frames" DUPLEX_FRAME_DIR="$HARNESS_DIR/frames" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$cpus" \
    python -m benchmarks.eval.benchmark_duplex_v10 record --dataset-root "$root" --dataset-revision "zips-sha256-$FDB_DIGEST" \
    --url "ws://127.0.0.1:$port/v1/realtime" --profile "$CLIENT_PROFILE" --model "$MODEL_REPO" --model-revision "$MODEL_REVISION" \
    --server-revision "$sha" --timeout 180 "${ids[@]}" --output "$out" > "$out-record.log" 2>&1)
}

wait_wave_slot() { # serialize the client waves of all cards at least WAVE_GAP s apart (shared stamp under /logs)
  local last
  exec 9> "$RESULTS_ROOT/wave.lock"
  flock 9
  while last=$(cat "$RESULTS_ROOT/wave.last" 2>/dev/null || echo 0); [ $(($(date +%s) - last)) -lt "$WAVE_GAP" ]; do sleep 1; done
  date +%s > "$RESULTS_ROOT/wave.last"
  flock -u 9
  exec 9>&-
}

select_ids() { # <count>: <count> ids spread evenly over ids96.txt (48 gives the odd lines)
  awk -v m="$1" '{a[NR] = $0} END {for (i = 0; i < m && i < NR; i++) print a[int(i * NR / m) + 1]}' "$HARNESS_DIR/ids96.txt"
}

run_ladder_job() { # <slot> <port> <label> <tag> <sha> <sessions> <speech> <samples> <frames>
  local slot=$1 port=$2 label=$3 tag=$4 sha=$5 sessions=$6 speech=$7 samples=$8 frames=$9
  local run=$RESULTS/$label cpus startup k i sampler hostload=""
  local -a pids=() ids=() mine=()
  cpus=${CPUS_LIST[$slot]}
  rm -rf "$run" && mkdir -p "$run/timing"
  prep_tree "$tag" "$sha" > "$run/prep.log" 2>&1 || { echo "$(stamp) $label NO_TREE $(cat "$run/prep.log")" >> "$RESULTS/chain.log"; return 1; }
  write_yaml "/work-$tag" "$run/server.yaml" "$sessions" "$speech" 1
  printf '%s\n' "label=$label" "tag=$tag" "sha=$sha" "sessions=$sessions" "speech=$speech" "samples=$samples" "frames=$frames" "card=${CARD_LIST[$slot]}" > "$run/meta.txt"
  if ! startup=$(start_server "/work-$tag" "$run/server.yaml" "$port" "$run/server.log" "$run/timing"); then
    echo "NOT_READY" > "$run/status"
    echo "$(stamp) $label card ${CARD_LIST[$slot]} NOT_READY $(grep -m1 -E 'Error' "$run/server.log" | cut -c1-300)" >> "$RESULTS/chain.log"
    stop_server "$run/server.yaml"
    return 0
  fi
  echo "startup_s=$startup" >> "$run/meta.txt"
  grep -o "StageGroup [a-z_]*: spawned 1 process(es) (pids=\[[0-9]*\])" "$run/server.log" > "$run/stage_pids.txt"
  (while [ ! -f "$run/clients.done" ]; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 >> "$run/mem.log"; sleep 1; done) &
  sampler=$!
  if [ "${HOSTLOAD:-0}" = 1 ]; then
    bash "$HARNESS_DIR/hostload.sh" > "$run/hostload.log" 2>&1 &
    hostload=$!
  fi
  sleep 5
  mapfile -t ids < <(select_ids "$samples")
  wait_wave_slot
  date -u +%s.%N > "$run/t_clients_start"
  for k in $(seq 0 $((sessions - 1))); do
    mine=()
    for i in "${!ids[@]}"; do [ $((i % sessions)) = "$k" ] && mine+=("${ids[$i]}"); done
    [ ${#mine[@]} = 0 ] && continue
    record "$run/w$k" "$port" "$sha" "$cpus" "$frames" "$FDB_ROOT" "${mine[@]}" &
    pids+=($!)
  done
  wait "${pids[@]}"
  date -u +%s.%N > "$run/t_clients_end"
  touch "$run/clients.done"
  wait "$sampler"
  [ -n "$hostload" ] && kill "$hostload"
  grep -c Traceback "$run/server.log" > "$run/tracebacks.txt"
  stop_server "$run/server.yaml"
  echo READY > "$run/status"
  echo "$(stamp) $(python "$HARNESS_DIR/report.py" run "$run")" >> "$RESULTS/chain.log"
}

stop_requested() { # <level>: a sweep stops at levels above the first level that failed to start or missed more than stop_miss
  local stop_level
  stop_level=$(cat "$RESULTS/STOP" 2>/dev/null) || return 1
  [ "$1" -gt "$stop_level" ]
}

note_stop() { # <run dir> <level>
  local stop_miss miss
  stop_miss=$(sed -n 's/^stop_miss=//p' "$RESULTS/gate.txt")
  if [ "$(cat "$1/status" 2>/dev/null)" = READY ]; then
    miss=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['miss_pct'])" "$1/run.json" 2>/dev/null || echo 100)
    python -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$miss" "$stop_miss" || return 0
  fi
  (
    flock 7
    old=$(cat "$RESULTS/STOP" 2>/dev/null || echo 999999)
    [ "$2" -lt "$old" ] && echo "$2" > "$RESULTS/STOP"
  ) 7> "$RESULTS/stop.lock"
}

note_unsolved() { # <level>: once every run of a level has finished, record the smallest level whose mean miss exceeds the threshold
  (
    flock 7
    python - "$RESULTS" "$1" << 'PY'
import glob
import json
import os
import sys

results, level = sys.argv[1], int(sys.argv[2])
gate = dict(line.rstrip("\n").split("=", 1) for line in open(f"{results}/gate.txt") if "=" in line)
misses = []
for meta in glob.glob(f"{results}/*/meta.txt"):
    run = os.path.dirname(meta)
    fields = dict(line.rstrip("\n").split("=", 1) for line in open(meta) if "=" in line)
    if os.path.basename(run).startswith("warm-") or int(fields.get("sessions", -1)) != level or not os.path.isfile(f"{run}/status"):
        continue
    ready = open(f"{run}/status").read().strip() == "READY" and os.path.isfile(f"{run}/run.json")
    summary = json.load(open(f"{run}/run.json")) if ready else {}
    # a run with session errors or server tracebacks counts as unsolved whatever its miss
    misses.append(summary["miss_pct"] if ready and not (summary.get("sessions_with_error") or summary.get("tracebacks")) else 100.0)
if len(misses) >= int(gate["runs"]) and sum(misses) / len(misses) > float(gate["threshold"]):
    path = f"{results}/UNSOLVED"
    old = int(open(path).read()) if os.path.isfile(path) else None
    if old is None or level < old:
        open(path, "w").write(str(level))
PY
  ) 7> "$RESULTS/stop.lock"
}

run_worker() { # <results> <slot> <warm 1|0>: take runs off the queue until it is empty
  load_results_env "$1"
  local slot=$2 warm=$3 job label tag sha sessions speech samples frames run workers kind jobno=0 base runs_above
  kind=$(sed -n 's/^kind=//p' "$RESULTS/gate.txt")
  runs_above=$(sed -n 's/^runs_above=//p' "$RESULTS/gate.txt")
  prep_tree "$BENCH_TAG" "$BENCH_SHA" > /dev/null || { echo "$(stamp) card ${CARD_LIST[$slot]} no bench tree" >> "$RESULTS/chain.log"; return 1; }
  ln -sfn "/work-$BENCH_TAG" /work-bench
  base=$((20000 + CARD_LIST[slot] * 1000))
  mps_start
  if [ "$warm" = 1 ]; then
    # a fresh container's first server is cold (JIT, first decode call about 5 s): one throwaway run, excluded from the reports
    read -r _ tag sha _ <<< "$(sed -n 's/^warm_tree=//p' "$RESULTS/gate.txt")"
    speech=$(sed -n 's/^speech=//p' "$RESULTS/gate.txt")
    frames=$(sed -n 's/^frames=//p' "$RESULTS/gate.txt")
    run_ladder_job "$slot" "$base" "warm-c${CARD_LIST[$slot]}" "$tag" "$sha" 8 "$speech" 16 "${frames:-0}"
  fi
  while true; do
    job=$(
      {
        flock 6
        head -1 "$RESULTS/queue"
        sed -i 1d "$RESULTS/queue"
      } 6> "$RESULTS/queue.lock"
    )
    [ -n "$job" ] || break
    read -r label tag sha sessions speech samples frames run <<< "$job"
    if stop_requested "$sessions"; then
      echo "$(stamp) $label SKIPPED (level $(cat "$RESULTS/STOP") failed)" >> "$RESULTS/chain.log"
      continue
    elif [ -n "$runs_above" ] && [ "$run" -gt "$runs_above" ] && [ -f "$RESULTS/UNSOLVED" ] && [ "$sessions" -gt "$(cat "$RESULTS/UNSOLVED")" ]; then
      echo "$(stamp) $label SKIPPED (above the first unsolved level $(cat "$RESULTS/UNSOLVED"))" >> "$RESULTS/chain.log"
      continue
    fi
    jobno=$((jobno + 1))
    run_ladder_job "$slot" "$((base + jobno % 90 * 10))" "$label" "$tag" "$sha" "$sessions" "$speech" "$samples" "$frames"
    if [ "$kind" = sweep ]; then
      note_stop "$RESULTS/$label" "$sessions"
      note_unsolved "$sessions"
    fi
  done
  mps_stop
  touch "$RESULTS/worker-$slot.done"
  workers=$(sed -n 's/^workers=//p' "$RESULTS/gate.txt")
  (
    flock 5
    if [ "$(find "$RESULTS" -maxdepth 1 -name 'worker-*.done' | wc -l)" = "$workers" ] && [ ! -f "$RESULTS/DONE" ]; then
      python "$HARNESS_DIR/report.py" "$kind" "$RESULTS" > "$RESULTS/report.md" 2>&1
      touch "$RESULTS/DONE"
    fi
  ) 5> "$RESULTS/done.lock"
}

run_unit() { # <results> <slot> <tag> <sha> [--gpu] [--full]
  load_results_env "$1"
  local tag=$3 sha=$4 gpu=0 tests="tests/unit_test/minicpm_o tests/unit_test/scheduling tests/unit_test/test_stage_device_contract.py" rc
  shift 4
  for arg in "$@"; do
    case $arg in
      --gpu) gpu=1 ;;
      --full) tests=tests/unit_test ;;
      *) echo "unknown option $arg" >> "$RESULTS/chain.log" ;;
    esac
  done
  prep_tree "$tag" "$sha" > "$RESULTS/prep.log" 2>&1 || { echo "$(stamp) NO_TREE" >> "$RESULTS/chain.log"; touch "$RESULTS/DONE"; return 1; }
  if [ "$gpu" = 1 ]; then
    # shellcheck disable=SC2086
    (cd "/work-$tag" && env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONPATH="/work-$tag" MINICPMO_CHECKPOINT="$MODEL_DIR" \
      taskset -c "$UNIT_CPUS" timeout 5400 python -m pytest tests/unit_test/minicpm_o -q -p no:cacheprovider -rfEs > "$RESULTS/$tag-gpu.log" 2>&1)
    rc=$?
    echo "$(stamp) $tag gpu rc=$rc $(grep -E '[0-9]+ (passed|failed)' "$RESULTS/$tag-gpu.log" | tail -1)" >> "$RESULTS/chain.log"
  else
    # shellcheck disable=SC2086
    (cd "/work-$tag" && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH="/work-$tag" MINICPMO_CHECKPOINT="$MODEL_DIR" \
      taskset -c "$UNIT_CPUS" timeout 5400 python -m pytest $tests -q -p no:cacheprovider -rfE > "$RESULTS/$tag-cpu.log" 2>&1)
    rc=$?
    echo "$(stamp) $tag cpu rc=$rc $(grep -E '[0-9]+ (passed|failed)' "$RESULTS/$tag-cpu.log" | tail -1)" >> "$RESULTS/chain.log"
  fi
  python "$HARNESS_DIR/report.py" unit "$RESULTS" > "$RESULTS/report.md" 2>&1
  touch "$RESULTS/DONE"
}

run_serving() { # <results> <slot> <tag> <sha> <c-list> <reps> [--graphs]
  load_results_env "$1"
  local slot=$2 tag=$3 sha=$4 conc=$5 reps=$6 graphs=0 rep port pacing max yaml startup
  [ "${7:-}" = --graphs ] && graphs=1
  if ! { prep_tree "$tag" "$sha" && prep_tree "$BENCH_TAG" "$BENCH_SHA"; } > "$RESULTS/prep.log" 2>&1; then
    echo "$(stamp) NO_TREE $(cat "$RESULTS/prep.log")" >> "$RESULTS/chain.log"
    touch "$RESULTS/DONE"
    return 1
  fi
  ln -sfn "/work-$BENCH_TAG" /work-bench
  max=$(tr ',' '\n' <<< "$conc" | sort -n | tail -1)
  yaml=$RESULTS/server.yaml
  write_yaml "/work-$tag" "$yaml" "$max" default "$graphs"
  printf '%s\n' "tag=$tag" "sha=$sha" "concurrencies=$conc" "reps=$reps" "graphs=$graphs" > "$RESULTS/meta.txt"
  for rep in $(seq 1 "$reps"); do
    port=$((28000 + CARD_LIST[slot] * 200 + rep * 10))
    if ! startup=$(start_server "/work-$tag" "$yaml" "$port" "$RESULTS/rep$rep-server.log" ""); then
      echo "$(stamp) rep $rep NOT_READY $(grep -m1 Error "$RESULTS/rep$rep-server.log" | cut -c1-300)" >> "$RESULTS/chain.log"
      stop_server "$yaml"
      break
    fi
    echo "READY ${startup}s" > "$RESULTS/rep$rep-server.out"
    for pacing in realtime lockstep; do
      (cd /work-bench && env PYTHONPATH=/work-bench OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "${CPUS_LIST[$slot]}" \
        python -m benchmarks.duplex.serving --url "ws://127.0.0.1:$port/v1/realtime" --audio "$FDB_ROOT"/candor_turn_taking/{1,2,3,4}/input.wav \
        --profile "$CLIENT_PROFILE" --concurrencies "$conc" --warmup-runs 1 --pacing "$pacing" --timeout-s 180 \
        --output-dir "$RESULTS/rep$rep/$pacing" > "$RESULTS/rep$rep-$pacing.out" 2>&1)
    done
    grep -c Traceback "$RESULTS/rep$rep-server.log" > "$RESULTS/rep$rep-tracebacks.txt"
    stop_server "$yaml"
    echo "$(stamp) rep $rep done startup ${startup}s" >> "$RESULTS/chain.log"
  done
  python "$HARNESS_DIR/report.py" serving "$RESULTS" > "$RESULTS/report.md" 2>&1
  touch "$RESULTS/DONE"
}

run_agree() { # <results> <slot> <tag> <sha>: all 727 samples, max_sessions 4, four shards at once, no MPS, no hook
  load_results_env "$1"
  local slot=$2 tag=$3 sha=$4 port yaml=$1/server.yaml startup
  port=$((28100 + CARD_LIST[slot] * 200))
  if ! { prep_tree "$tag" "$sha" && prep_tree "$BENCH_TAG" "$BENCH_SHA"; } > "$RESULTS/prep.log" 2>&1; then
    echo "$(stamp) NO_TREE $(cat "$RESULTS/prep.log")" >> "$RESULTS/chain.log"
    touch "$RESULTS/DONE"
    return 1
  fi
  ln -sfn "/work-$BENCH_TAG" /work-bench
  write_yaml "/work-$tag" "$yaml" 4 default 1
  if ! startup=$(start_server "/work-$tag" "$yaml" "$port" "$RESULTS/server.log" ""); then
    echo "$(stamp) NOT_READY" >> "$RESULTS/chain.log"
    stop_server "$yaml"
    touch "$RESULTS/DONE"
    return 1
  fi
  echo "$(stamp) READY ${startup}s" >> "$RESULTS/chain.log"
  shard() { # <name> <subset>...
    local name=$1 subset n
    shift
    local -a ids=()
    for subset in "$@"; do
      for n in $(find "$FDB_ROOT/$subset" -mindepth 1 -maxdepth 1 -regex '.*/[0-9]+' -printf '%f\n' | sort -n); do ids+=("$subset/$n"); done
    done
    record "$RESULTS/$name" "$port" "$sha" "${CPUS_LIST[$slot]}" 0 "$FDB_ROOT" "${ids[@]}"
    echo "$(stamp) shard $name rc=$? samples=${#ids[@]}" >> "$RESULTS/chain.log"
  }
  shard s1 candor_pause_handling &
  shard s2 candor_turn_taking icc_backchannel &
  shard s3 synthetic_pause_handling &
  shard s4 synthetic_user_interruption &
  wait
  grep -c Traceback "$RESULTS/server.log" > "$RESULTS/tracebacks.txt"
  stop_server "$yaml"
  python "$HARNESS_DIR/report.py" agree "$RESULTS" > "$RESULTS/report.md" 2>&1
  touch "$RESULTS/DONE"
}

run_stop() { # stop gate workers, servers (with their stage processes), clients and MPS in this container
  local pattern
  for pattern in "gate[.]sh run-(worker|unit|serving|agree)" "benchmark_duplex[_]v10" "benchmarks[.]duplex[.]serving" "[p]ytest" "sglang_omni[.]cli serve" "multiprocessing[.]spawn" "multiprocessing[.]resource_tracker" "hostload[.]sh"; do
    pkill -9 -f "$pattern"
  done
  echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe nvidia-cuda-mps-control 2> /dev/null
  true
}

# ---------------------------------------------------------------- dispatch

cmd=${1:-}
[ -n "$cmd" ] && shift
case $cmd in
  push) push "$@" ;;
  pull) pull "$@" ;;
  unit) cmd_single unit "$@" ;;
  serving) cmd_single serving "$@" ;;
  agree) cmd_single agree "$@" ;;
  ladder) cmd_ladder "$@" ;;
  perception) cmd_perception "$@" ;;
  sweep) cmd_sweep "$@" ;;
  status) cmd_status "$@" ;;
  report) python3 "$HERE/report.py" "$(sed -n 's/^kind=//p' "$OMNI_ROOT/logs/gates/${1:?name}/gate.txt")" "$OMNI_ROOT/logs/gates/$1" ;;
  stop) cmd_stop ;;
  run-worker) run_worker "$@" ;;
  run-unit) run_unit "$@" ;;
  run-serving) run_serving "$@" ;;
  run-agree) run_agree "$@" ;;
  run-stop) run_stop ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
