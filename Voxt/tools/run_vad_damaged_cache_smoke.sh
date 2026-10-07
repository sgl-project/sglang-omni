#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH=""
OUTPUT_DIR="$ROOT/tmp/regression/vad-damaged-cache-smoke"
DURATION_SECONDS=30
SAMPLE_INTERVAL_SECONDS=5
KEEP_WORKDIR=false

usage() {
  cat <<'EOF'
Usage: tools/run_vad_damaged_cache_smoke.sh --app PATH [options]

Runs a launch smoke with isolated preferences and deliberately damaged
local MLX Silero VAD caches. This exercises the real app startup path without
touching the user's normal Voxt preferences or model cache.

Options:
  --app PATH          Required .app bundle path.
  --output-dir PATH   Output directory. Defaults under tmp/regression.
  --duration SECONDS  Monitoring duration. Defaults to 30.
  --interval SECONDS  CPU/RSS sample interval. Defaults to 5.
  --keep-workdir      Keep the temporary HOME/model cache directory.
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

HOME_DIR="$WORKDIR/home"
MODEL_ROOT="$WORKDIR/model-storage"
PREFS_DIR="$HOME_DIR/Library/Preferences"
PREFS_PLIST="$PREFS_DIR/$BUNDLE_ID.plist"
mkdir -p "$PREFS_DIR" "$MODEL_ROOT"

MLX_CACHE="$MODEL_ROOT/mlx-audio/mlx-community_silero-vad"
mkdir -p "$MLX_CACHE"
: > "$MLX_CACHE/config.json"
: > "$MLX_CACHE/model.safetensors"

FEATURE_SETTINGS_JSON="$(cat <<'JSON'
{
  "transcription": {
    "asrSelectionID": "mlx:mlx-community/SenseVoiceSmall",
    "llmEnabled": false,
    "llmSelectionID": "local-llm:mlx-community/Qwen3.5-2B-4bit",
    "prompt": "",
    "appContext": {
      "enabled": false,
      "textEnabled": false,
      "screenshotEnabled": false
    },
    "notes": {
      "enabled": false,
      "triggerShortcut": {
        "keyCode": 49,
        "modifiersRawValue": 0,
        "sidedModifiersRawValue": 0
      },
      "titleModelSelectionID": "local-llm:mlx-community/Qwen3.5-2B-4bit",
      "soundEnabled": false,
      "soundPreset": "soft",
      "obsidianSync": {
        "enabled": false,
        "vaultPath": "",
        "relativeFolder": "Voxt",
        "groupingMode": "file"
      },
      "remindersSync": {
        "enabled": false,
        "selectedListIdentifier": "",
        "selectedListTitle": ""
      }
    }
  },
  "translation": {
    "asrSelectionID": "mlx:mlx-community/SenseVoiceSmall",
    "modelSelectionID": "local-llm:mlx-community/Qwen3.5-2B-4bit",
    "targetLanguageRawValue": "english",
    "prompt": "",
    "showResultWindow": true
  },
  "rewrite": {
    "asrSelectionID": "mlx:mlx-community/SenseVoiceSmall",
    "llmSelectionID": "local-llm:mlx-community/Qwen3.5-2B-4bit",
    "prompt": "",
    "appContext": {
      "enabled": false,
      "textEnabled": false,
      "screenshotEnabled": false
    },
    "appEnhancementEnabled": true,
    "continueShortcut": {
      "keyCode": 49,
      "modifiersRawValue": 0,
      "sidedModifiersRawValue": 0
    }
  },
  "meeting": {
    "asrSelectionID": "mlx:mlx-community/SenseVoiceSmall",
    "summaryModelSelectionID": "local-llm:mlx-community/Qwen3.5-2B-4bit",
    "summaryPrompt": "",
    "summaryAutoGenerate": true,
    "realtimeTranslateEnabled": false,
    "realtimeTargetLanguageRawValue": "",
    "hideOverlayFromScreenSharing": false,
    "chunkingModeRawValue": "quality",
    "speakerDiarizationModelRawValue": "sortformerV2",
    "finalTranscriptOptimizationEnabled": true
  }
}
JSON
)"

/usr/bin/plutil -create xml1 "$PREFS_PLIST"
/usr/bin/plutil -insert onboardingCompleted -bool YES "$PREFS_PLIST"
/usr/bin/plutil -insert modelStorageRootPath -string "$MODEL_ROOT" "$PREFS_PLIST"
/usr/bin/plutil -insert localVADMode -string silero "$PREFS_PLIST"
/usr/bin/plutil -insert featureSettings -string "$FEATURE_SETTINGS_JSON" "$PREFS_PLIST"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SMOKE_LOG="$OUTPUT_DIR/damaged-cache-smoke-$TIMESTAMP.log"

"$ROOT/tools/run_app_launch_smoke.sh" \
  --app "$APP_PATH" \
  --launch-mode direct \
  --env "HOME=$HOME_DIR" \
  --env "CFFIXED_USER_HOME=$HOME_DIR" \
  --arg "-ApplePersistenceIgnoreState" \
  --arg "YES" \
  --duration "$DURATION_SECONDS" \
  --interval "$SAMPLE_INTERVAL_SECONDS" \
  --log "$SMOKE_LOG" \
  --fail-on-log '(VAD|Silero).*(fatal|Fatal|error|Error)' \
  --fail-on-log '(fatal|Fatal|error|Error).*(VAD|Silero)' \
  --fail-on-log '(rawAudio|raw audio|audioSamples|audio samples|sampleBuffer|fullTranscript|full transcript|transcriptText|transcript text|promptText|prompt text|\[VOXT_SMOKE\]\[prompt\])'

{
  echo "bundleID=$BUNDLE_ID"
  echo "isolatedHome=$HOME_DIR"
  echo "modelRoot=$MODEL_ROOT"
  echo "mlxCache=$MLX_CACHE"
  echo "smokeLog=$SMOKE_LOG"
  echo "keptWorkdir=$KEEP_WORKDIR"
} | tee "$OUTPUT_DIR/damaged-cache-smoke-$TIMESTAMP.summary"
