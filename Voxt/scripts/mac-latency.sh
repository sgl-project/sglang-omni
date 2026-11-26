#!/usr/bin/env bash
# Times the first transcription on a Mac with the real speech models
# (TestMacE2EFirstLatency, mac_e2e_latency_test.go): the app just opened
# and a speech model just changed, against a warm baseline. No window,
# key, microphone or speaker: a fake helper plays the speech sample.
#
#   scripts/mac-latency.sh ACCURATE_DIR FAST_DIR [ROUNDS [SETTLE]]
#
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
cd "$(dirname "$0")/.."
APP="${VOXTNEW_APP:-/Applications/Voxt.app}"
HUB="${VOXTNEW_HUB:-$HOME/Library/Application Support/SGLang Omni/models}"
silero="$(ls -d "$HUB"/models--mlx-community--silero-vad-v6/snapshots/* 2>/dev/null | head -1 || true)"
models="asr.accurate=$1;asr.fast=$2"
[[ -n "$silero" ]] && models="$models;vad.silero=$silero"
mkdir -p build
rm -f build/latency-attempt-*.json build/latency-round-*.jsonl build/latency.log build/text-consistency.json
rounds="${3:-3}"
rc=0
for round in $(seq 1 "$rounds"); do
  attempt="build/latency-attempt-$round.json"
  before_others="$(pgrep -fl 'qwen3_asr_server|silero_vad_server' || true)"
  before_power="$(pmset -g ps 2>/dev/null || true)"
  before_pressure="$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)"
  export VOXTNEW_LATENCY=1 VOXTNEW_LATENCY_ROUNDS=1 VOXTNEW_LATENCY_SETTLE="${4:-3s}" VOXTNEW_OMNI_MODELS="$models"
  export VOXTNEW_MAIN_SHA="${VOXTNEW_MAIN_SHA:?VOXTNEW_MAIN_SHA must name the source commit}"
  export VOXTNEW_OMNI_RUNTIME="${VOXTNEW_OMNI_RUNTIME:-$APP/Contents/Resources/omni}"
  export VOXTNEW_LATENCY_OUT="build/latency-round-$round.jsonl"
  go test -count=1 -run TestMacE2EFirstLatency -v -timeout 30m . >>build/latency.log 2>&1 || rc=$((rc+1))
  after_others="$(pgrep -fl 'qwen3_asr_server|silero_vad_server' || true)"
  after_pressure="$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)"
  python3 - "$attempt" "$before_others" "$before_power" "$before_pressure" "$after_others" "$after_pressure" <<'PY'
import json, sys
def reasons(others, power, pressure):
    out = []
    if others: out.append("contender: " + others.splitlines()[0][:120])
    if "AC Power" not in power: out.append("not on AC power")
    return out
before_o, before_p, before_pr, after_o, after_pr = sys.argv[1:6]
env = {
    "before": {"others": before_o, "pressure": before_pr},
    "after": {"others": after_o, "pressure": after_pr},
    "environment_reasons": reasons(before_o, before_p, before_pr) + reasons(after_o, "", after_pr),
    "peak_pressure": "normal" if before_pr == "1" and after_pr == "1" else "warn",
}
json.dump(env, open(sys.argv[1], "w"))
PY
done
echo "$rc" > build/latency-returncode.txt
python3 - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import profile_loop as loop

observations = []
for round_file in sorted(Path("build").glob("latency-round-*.jsonl")):
    rows = [json.loads(line) for line in round_file.read_text().splitlines() if line.strip()]
    observations.append({"rows": rows})
loop.write_text_consistency(observations, Path("build/text-consistency.json"))
PY
if python3 scripts/short_open_gate.py build "$rounds"; then
  grep 'MEASURE latency' build/latency.log | sed 's/^ *[a-z_0-9]*\.go:[0-9]*: //' || true
else
  echo "short-open group incomplete: no median (see build/latency-attempt-*.json and build/latency-round-*.jsonl)" >&2
  exit 1
fi
