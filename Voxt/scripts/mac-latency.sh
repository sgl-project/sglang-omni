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
others="$(pgrep -fl 'qwen3_asr_server|silero_vad_server' || true)"
[[ -z "$others" ]] || { echo "note: other model processes are running; the times are skewed:"; echo "$others" | cut -c1-160; }
mkdir -p build
export VOXTNEW_LATENCY=1 VOXTNEW_LATENCY_ROUNDS="${3:-3}" VOXTNEW_LATENCY_SETTLE="${4:-3s}" VOXTNEW_OMNI_MODELS="$models"
export VOXTNEW_OMNI_RUNTIME="${VOXTNEW_OMNI_RUNTIME:-$APP/Contents/Resources/omni}"
go test -count=1 -run TestMacE2EFirstLatency -v -timeout 30m . >build/latency.log 2>&1 || { tail -30 build/latency.log; exit 1; }
grep 'MEASURE latency' build/latency.log | sed 's/^ *[a-z_0-9]*\.go:[0-9]*: //'
