#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH=""
OUTPUT_DIR="$ROOT/tmp/regression/vad-preferences-smoke"
DURATION_SECONDS=10
SAMPLE_INTERVAL_SECONDS=5
KEEP_WORKDIR=false

usage() {
  cat <<'EOF'
Usage: tools/run_vad_preferences_smoke.sh --app PATH [options]

Runs isolated launch smokes for clean preferences and each global localVADMode
value. Verifies that startup does not crash, does not emit VAD fatal/error logs,
and does not persist removed experimental backend values.

Options:
  --app PATH          Required .app bundle path.
  --output-dir PATH   Output directory. Defaults under tmp/regression.
  --duration SECONDS  Monitoring duration per scenario. Defaults to 10.
  --interval SECONDS  CPU/RSS sample interval. Defaults to 5.
  --keep-workdir      Keep the temporary HOME directories.
  -h, --help          Show this help.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --app)
      APP_PATH="${2:-}"
      [[ -n "$APP_PATH" ]] || die "--app requires a value"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      [[ -n "$OUTPUT_DIR" ]] || die "--output-dir requires a value"
      shift 2
      ;;
    --duration)
      DURATION_SECONDS="${2:-}"
      [[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]] || die "--duration requires a positive integer"
      shift 2
      ;;
    --interval)
      SAMPLE_INTERVAL_SECONDS="${2:-}"
      [[ "$SAMPLE_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || die "--interval requires a positive integer"
      shift 2
      ;;
    --keep-workdir)
      KEEP_WORKDIR=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$APP_PATH" ]] || die "--app is required"
[[ -d "$APP_PATH" ]] || die "app bundle not found: $APP_PATH"
[[ "$DURATION_SECONDS" -gt 0 ]] || die "--duration must be greater than 0"
[[ "$SAMPLE_INTERVAL_SECONDS" -gt 0 ]] || die "--interval must be greater than 0"

APP_PATH="$(cd "$(dirname "$APP_PATH")" && pwd)/$(basename "$APP_PATH")"
INFO_PLIST="$APP_PATH/Contents/Info.plist"
[[ -f "$INFO_PLIST" ]] || die "Info.plist not found: $INFO_PLIST"

BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$INFO_PLIST" 2>/dev/null || true)"
[[ -n "$BUNDLE_ID" ]] || die "unable to read CFBundleIdentifier from $INFO_PLIST"

mkdir -p "$OUTPUT_DIR"
WORKDIR="$(mktemp -d "$OUTPUT_DIR/workdir.XXXXXX")"
if [[ "$KEEP_WORKDIR" != "true" ]]; then
  trap 'rm -rf "$WORKDIR"' EXIT
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY_PATH="$OUTPUT_DIR/preferences-smoke-$TIMESTAMP.summary"

extract_feature_settings() {
  local prefs_plist="$1"
  /usr/bin/plutil -extract featureSettings raw -o - "$prefs_plist" 2>/dev/null || true
}

assert_preferences_safe_after_launch() {
  local scenario="$1"
  local prefs_plist="$2"
  local mode_expected="${3:-}"

  [[ -f "$prefs_plist" ]] || die "$scenario did not write preferences plist: $prefs_plist"
  local feature_settings
  feature_settings="$(extract_feature_settings "$prefs_plist")"
  if [[ -n "$feature_settings" && "$feature_settings" == *"FireRed"* ]]; then
    echo "$feature_settings" > "$OUTPUT_DIR/$scenario-featureSettings.json"
    die "$scenario unexpectedly persisted a removed experimental VAD backend in featureSettings"
  fi
  if [[ -n "$mode_expected" ]]; then
    local mode_value
    mode_value="$(/usr/bin/plutil -extract localVADMode raw -o - "$prefs_plist" 2>/dev/null || true)"
    [[ "$mode_value" == "$mode_expected" ]] || die "$scenario expected localVADMode to remain $mode_expected, got ${mode_value:-missing}"
  fi
}

run_scenario() {
  local scenario="$1"
  local local_vad_mode="${2:-}"
  local scenario_dir="$WORKDIR/$scenario"
  local home_dir="$scenario_dir/home"
  local model_root="$scenario_dir/model-storage"
  local prefs_dir="$home_dir/Library/Preferences"
  local prefs_plist="$prefs_dir/$BUNDLE_ID.plist"
  local smoke_log="$OUTPUT_DIR/preferences-$scenario-$TIMESTAMP.log"

  mkdir -p "$prefs_dir" "$model_root"
  /usr/bin/plutil -create xml1 "$prefs_plist"
  /usr/bin/plutil -insert modelStorageRootPath -string "$model_root" "$prefs_plist"
  if [[ -n "$local_vad_mode" ]]; then
    /usr/bin/plutil -insert localVADMode -string "$local_vad_mode" "$prefs_plist"
  fi

  "$ROOT/tools/run_app_launch_smoke.sh" \
    --app "$APP_PATH" \
    --launch-mode direct \
    --env "HOME=$home_dir" \
    --env "CFFIXED_USER_HOME=$home_dir" \
    --arg "-ApplePersistenceIgnoreState" \
    --arg "YES" \
    --duration "$DURATION_SECONDS" \
    --interval "$SAMPLE_INTERVAL_SECONDS" \
    --log "$smoke_log" \
    --fail-on-log '(VAD|Silero).*(fatal|Fatal|error|Error)' \
    --fail-on-log '(fatal|Fatal|error|Error).*(VAD|Silero)' \
    --fail-on-log '(rawAudio|raw audio|audioSamples|audio samples|sampleBuffer|fullTranscript|full transcript|transcriptText|transcript text|promptText|prompt text|\[VOXT_SMOKE\]\[prompt\])'

  assert_preferences_safe_after_launch "$scenario" "$prefs_plist" "$local_vad_mode"

  {
    echo "scenario=$scenario"
    echo "prefs=$prefs_plist"
    echo "smokeLog=$smoke_log"
    echo "featureSettingsWritten=$([[ -n "$(extract_feature_settings "$prefs_plist")" ]] && echo true || echo false)"
    if [[ -n "$local_vad_mode" ]]; then
      echo "localVADMode=$local_vad_mode"
    fi
  } >> "$SUMMARY_PATH"
}

run_scenario "clean"
run_scenario "local-vad-automatic" "automatic"
run_scenario "local-vad-silero" "silero"
run_scenario "local-vad-energy" "energy"
run_scenario "local-vad-off" "off"

{
  echo "bundleID=$BUNDLE_ID"
  echo "workdir=$WORKDIR"
  echo "keptWorkdir=$KEEP_WORKDIR"
  echo "summary=$SUMMARY_PATH"
} | tee -a "$SUMMARY_PATH"
