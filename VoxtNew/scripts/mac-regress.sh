#!/usr/bin/env bash
# The regression on a Mac with the real models (docs/manual-test.md,
# docs/eval.md): the helper's Swift tests, the speech models with the real
# helper (TestMacE2EOmni), the language models (TestLLM*), the eval in
# quick mode (TestEval*: the speech and language models' scores against
# floors, app enhancement and the dictionary end to end), then the app
# built with the runtimes inside, started (--smoke: its own data, no
# window), asked for an answer by each runtime and quit, with nothing it
# started left running. A summary table at the end; the exit status is
# the number of failed steps.
#
#   MAC_DIR=code/VoxtNew-m3 scripts/mac.sh scripts/mac-regress.sh [--full] [--keep]
#
# It reads the persistent models and runtimes, never downloads or builds
# them (their READMEs say how to refresh them):
#   $HUB (~/.cache/huggingface/hub, or $HF_HUB_CACHE): the six catalogue
#     models' snapshots at their pinned commits (the ActNow family's model
#     library; internal/models' TestRealSnapshots lists them)
#   ~/voxtnew-runtime/omni/     VOXTNEW_RUNTIME_DIR: scripts/build-omni.sh --out
#   ~/voxtnew-runtime/llm/      scripts/build-llm.sh --out
# --full runs the whole eval (about 20 minutes more) and writes its tables
# to build/regress/eval; --keep keeps the app (build/darwin-arm64). No
# microphone, speaker, window, key or clipboard: the helper plays files.
# Another voxtnew-llm or qwen3_asr_server running (another thread's work)
# skews the times: the script says so.
set -uo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run it on a Mac: MAC_DIR=code/VoxtNew-m3 scripts/mac.sh scripts/mac-regress.sh" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FULL=0 KEEP=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    --keep) KEEP=1 ;;
    *) echo "unknown argument $a" >&2; exit 1 ;;
  esac
done
R="${VOXTNEW_RUNTIME_DIR:-$HOME/voxtnew-runtime}"
OUT="$ROOT/build/regress"
rm -rf "$OUT"
mkdir -p "$OUT"

missing=0
# The models' snapshot folders in $HUB, by catalogue id (the catalogue says
# which commit; nothing is downloaded).
VOXTNEW_SNAPSHOTS_OUT="$OUT/snapshots.txt" go test -count=1 -run TestRealSnapshots ./internal/models >/dev/null || exit 2
# snap ID: model ID's snapshot folder ("" when it is not installed).
snap() { sed -n "s|^$1=||p" "$OUT/snapshots.txt"; }
for id in asr.accurate asr.fast vad.silero llm.balanced llm.fast llm.translate; do
  [[ -n "$(snap "$id")" ]] || { echo "missing model $id in the model library (download it in VoxtNew, or internal/models' TestRealMoveToHub)" >&2; missing=1; }
done
[[ -x "$R/omni/bin/qwen3_asr_server" ]] || { echo "missing $R/omni (see $R/README.md)" >&2; missing=1; }
[[ -x "$R/llm/voxtnew-llm" ]] || { echo "missing $R/llm (see $R/README.md)" >&2; missing=1; }
((missing)) && exit 2

# The runtimes run from APFS clones of their own (no copy on disk): other
# threads run ~/voxtnew-runtime's at the same time, and the leftovers this
# script looks for (and kills) must be its own.
mkdir -p "$OUT/runtime"
cp -Rc "$R/omni" "$OUT/runtime/omni" && cp -Rc "$R/llm" "$OUT/runtime/llm" || exit 2
ours="$OUT|$ROOT/build/darwin-arm64"
others="$(pgrep -fl 'voxtnew-llm|qwen3_asr_server' | grep -vE "$ours" || true)"
if [[ -n "$others" ]]; then
  echo "note: other model processes are running; the times are skewed:"
  echo "$others" | cut -c1-160
fi
free="$(memory_pressure -Q | sed -n 's/.*percentage: \([0-9]*\)%.*/\1/p')"
echo "== free memory ${free}%"
if (( ${free:-0} < ${VOXTNEW_MIN_FREE:-30} )); then
  echo "less than ${VOXTNEW_MIN_FREE:-30}% of memory is free: not starting" >&2
  exit 3
fi

export VOXTNEW_HELPER="$OUT/helper/voxtnew-helper" VOXTNEW_OMNI_RUNTIME="$OUT/runtime/omni" VOXTNEW_LLM="$OUT/runtime/llm/voxtnew-llm"
export GOFLAGS=-p=4

names=() results=() times=() notes=()
# step NAME NOTE-GREP CMD...: runs CMD, its output in $OUT/NAME.log; the
# lines matching NOTE-GREP (the last) go to the summary.
step() {
  local name="$1" grep="$2"
  shift 2
  echo "== $name"
  local t0=$SECONDS status
  "$@" >"$OUT/$name.log" 2>&1
  status=$?
  local note
  note="$(grep -E "$grep" "$OUT/$name.log" | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-90)"
  local left
  left="$(pgrep -fl 'qwen3_asr_server|voxtnew-llm|voxtnew-helper' | grep -E "$ours" || true)"
  if [[ -n "$left" ]]; then
    status=1
    note="left running: $(echo "$left" | awk '{print $1}' | tr '\n' ' ')"
    echo "$left" | awk '{print $1}' | xargs kill 2>/dev/null
  fi
  if ((status == 0)); then
    results+=("pass")
  else
    results+=("FAIL")
    grep -E -- "--- FAIL|FAIL|error|Error" "$OUT/$name.log" | head -15
  fi
  names+=("$name") times+=("$((SECONDS - t0)) s") notes+=("$note")
  echo "   ${results[${#results[@]}-1]} in ${times[${#times[@]}-1]}: $note"
}

omni_models="asr.accurate=$(snap asr.accurate);asr.fast=$(snap asr.fast);vad.silero=$(snap vad.silero)"
dev_llm="llm.balanced=$(snap llm.balanced),llm.fast=$(snap llm.fast),llm.translate=$(snap llm.translate)"

step helper-build 'built|helper' scripts/build-helper.sh --out "$OUT/helper"
step helper-swift-test 'Executed .* tests' bash -c 'cd helper && swift test -j 2'
step omni-e2e 'MEASURE asr.accurate live|^(ok|FAIL)' env VOXTNEW_MAC_E2E=1 VOXTNEW_OMNI_MODELS="$omni_models" \
  go test -count=1 -run TestMacE2EOmni -v .
step llm-real '^(ok|FAIL)' env VOXTNEW_LLM_REAL=1 VOXTNEW_DEV_LLM="$dev_llm" \
  go test -count=1 -timeout 30m -run 'TestLLM' -v .
if ((FULL)); then
  step eval-full '^(ok|FAIL)' env VOXTNEW_EVAL=1 VOXTNEW_EVAL_OUT="$OUT/eval" go test -count=1 -timeout 2h -run 'TestEval' -v .
else
  step eval-quick '^(ok|FAIL)' env VOXTNEW_EVAL=1 VOXTNEW_EVAL_QUICK=1 VOXTNEW_EVAL_OUT="$OUT/eval" \
    go test -count=1 -timeout 1h -run 'TestEval' -v .
fi

# The app, with the runtimes inside (APFS clones: no copy on disk).
app_build() {
  local res="$ROOT/resources/darwin-arm64"
  rm -rf "$res/omni" "$res/llm" "$res/helper" &&
    mkdir -p "$res" &&
    cp -Rc "$R/omni" "$res/omni" && cp -Rc "$R/llm" "$res/llm" && cp -Rc "$OUT/helper" "$res/helper" &&
    go tool mygo build &&
    codesign --verify --deep --strict build/darwin-arm64/VoxtNew.app &&
    du -sh build/darwin-arm64/VoxtNew.app | sed 's/^/built /'
}
smoke() {
  local dir="$OUT/smoke"
  mkdir -p "$dir/data"
  # The app's own processes, not the environment's; the models from $HUB
  # (the fast ones), a family directory of its own (the real one is not
  # written).
  env -u VOXTNEW_HELPER -u VOXTNEW_OMNI_RUNTIME -u VOXTNEW_LLM \
    ACTNOW_FAMILY_DIR="$dir/ActNow" VOXTNEW_SMOKE_MODELS=asr.fast,llm.fast \
    "$ROOT/build/darwin-arm64/VoxtNew.app/Contents/MacOS/VoxtNew" --smoke "$dir" &
  local pid=$! t=0 rt="$dir/ActNow/runtime" most=0
  # The family's runtime file (spec §11.1) while it runs: the models it
  # holds; it must be gone after quit.
  while kill -0 $pid 2>/dev/null && ((t < 600)); do
    for f in "$rt"/*.json; do
      [[ -f "$f" ]] || continue
      local n
      n="$(grep -c '"repo"' "$f" || true)"
      if ((n > most)); then most=$n; cp "$f" "$OUT/runtime-file.json"; fi
    done
    sleep 0.5
    t=$((t + 1))
  done
  if kill -0 $pid 2>/dev/null; then
    echo "smoke: FAIL still running after 300 s"
    kill $pid
    return 1
  fi
  wait $pid || return 1
  sleep 1
  local left
  left="$(pgrep -fl 'qwen3_asr_server|voxtnew-llm|voxtnew-helper' | grep -F "$ROOT/build/darwin-arm64/VoxtNew.app" || true)"
  if [[ -n "$left" ]]; then
    echo "smoke: FAIL left running after quit: $left"
    return 1
  fi
  if ((most == 0)); then
    echo "smoke: FAIL no runtime file listed a loaded model"
    return 1
  fi
  if compgen -G "$rt/*.json" >/dev/null; then
    echo "smoke: FAIL the runtime file is left after quit: $(ls "$rt")"
    return 1
  fi
  echo "smoke: runtime file listed $most models ($OUT/runtime-file.json), removed at quit"
  echo "smoke: nothing left running"
}
step app-build '^built' app_build
step app-smoke '^smoke: (FAIL|speech|nothing)' smoke

rm -rf "$ROOT/resources/darwin-arm64/omni" "$ROOT/resources/darwin-arm64/llm" "$ROOT/resources/darwin-arm64/helper"
((KEEP)) || rm -rf "$ROOT/build/darwin-arm64"

echo
echo "| step | result | time | note |"
echo "|---|---|---|---|"
failed=0
for i in "${!names[@]}"; do
  echo "| ${names[$i]} | ${results[$i]} | ${times[$i]} | ${notes[$i]} |"
  [[ "${results[$i]}" == pass ]] || failed=$((failed + 1))
done
[[ -n "$others" ]] && echo "(other model processes were running: times skewed)"
echo "logs: $OUT/*.log"
exit "$failed"
