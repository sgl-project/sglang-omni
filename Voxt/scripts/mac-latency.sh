#!/usr/bin/env bash
# Times the first transcription on a Mac with the real speech models
# (TestMacE2EFirstLatency, mac_e2e_latency_test.go): the app just opened
# and a speech model just changed, against a warm baseline. No window,
# key, microphone or speaker: a fake helper plays the speech sample.
#
#   scripts/mac-latency.sh ACCURATE_DIR FAST_DIR [ROUNDS [SETTLE]]
#
# ROUNDS must be 3 because the group gate requires three attempts.
# SETTLE (3s) is the wait between the app's start, or the change of model,
# and the shortcut.
#
# ACCURATE_DIR and FAST_DIR are the folders of two speech models (asr.accurate
# and asr.fast: a snapshot of the model library, or a download of the
# catalogue's files). The Omni runtime is VOXTNEW_APP's
# (/Applications/Voxt.app), unless VOXTNEW_OMNI_RUNTIME names another;
# Silero is the model library's.
# The MEASURE lines are the result; the whole log is in build/latency.log.
set -euo pipefail

[[ "$(uname -s)" == "Darwin" ]] || { echo "run it on a Mac" >&2; exit 1; }
[[ $# -ge 2 ]] || { sed -n '2,17p' "$0" >&2; exit 1; }
rounds="${3:-3}"
[[ "$rounds" == 3 ]] || { echo "ROUNDS must be 3: a short-open median requires exactly three attempts" >&2; exit 2; }
cd "$(dirname "$0")/.."
APP="${VOXTNEW_APP:-/Applications/Voxt.app}"
HUB="${VOXTNEW_HUB:-$HOME/Library/Application Support/SGLang Omni/models}"
silero="$(ls -d "$HUB"/models--mlx-community--silero-vad-v6/snapshots/* 2>/dev/null | head -1 || true)"
models="asr.accurate=$1;asr.fast=$2"
[[ -n "$silero" ]] && models="$models;vad.silero=$silero"
others="$(pgrep -fl 'qwen3_asr_server|silero_vad_server' || true)"
[[ -z "$others" ]] || { echo "note: other model processes are running; the times are skewed:"; echo "$others" | cut -c1-160; }
mkdir -p build
reasons=()
[[ -z "$others" ]] || reasons+=("contenders present before run")
power="$(pmset -g ps 2>/dev/null || true)"
[[ "$power" == *"AC Power"* ]] || reasons+=("not on AC power before run")
pressure="$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)"
preflight="build/latency-preflight.json"
rm -f build/latency.jsonl
export VOXTNEW_LATENCY=1 VOXTNEW_LATENCY_ROUNDS="$rounds" VOXTNEW_LATENCY_SETTLE="${4:-3s}" VOXTNEW_OMNI_MODELS="$models"
export VOXTNEW_MAIN_SHA="${VOXTNEW_MAIN_SHA:?VOXTNEW_MAIN_SHA must name the source commit}"
export VOXTNEW_OMNI_RUNTIME="${VOXTNEW_OMNI_RUNTIME:-$APP/Contents/Resources/omni}"
export VOXTNEW_LATENCY_OUT=build/latency.jsonl
rc=0
go test -count=1 -run TestMacE2EFirstLatency -v -timeout 30m . >build/latency.log 2>&1 || rc=$?
others_after="$(pgrep -fl 'qwen3_asr_server|silero_vad_server' || true)"
[[ -z "$others_after" ]] || reasons+=("contenders present after run")
power_after="$(pmset -g ps 2>/dev/null || true)"
[[ "$power_after" == *"AC Power"* ]] || reasons+=("not on AC power after run")
pressure_after="$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)"
peak=normal
[[ "$pressure" == 1 && "$pressure_after" == 1 ]] || peak=warn
python3 - "$preflight" "$rc" "${reasons[@]+"${reasons[@]}"}" "$peak" <<'PY'
import json, sys
json.dump({"returncode": int(sys.argv[2]),
           "environment_reasons": sys.argv[3:-1] if len(sys.argv) > 3 else [],
           "peak_pressure": sys.argv[-1]}, open(sys.argv[1], "w"))
PY
python3 - build/latency.jsonl build/text-consistency.json <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import profile_loop as loop

rows_path = Path(sys.argv[1])
rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()] if rows_path.exists() else []
loop.write_text_consistency([{"rows": rows}], Path(sys.argv[2]))
PY
if python3 scripts/short_open_gate.py build/latency.jsonl "$preflight" "$rounds"; then
  grep 'MEASURE latency' build/latency.log | sed 's/^ *[a-z_0-9]*\.go:[0-9]*: //' || true
else
  tail -30 build/latency.log | grep -v 'MEASURE latency' || true
  echo "short-open group incomplete: no median (see build/latency-preflight.json and build/latency.jsonl)" >&2
  exit 1
fi
