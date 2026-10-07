#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT/Voxt.xcodeproj"
SCHEME="Voxt"
CONFIGURATION="Release"
DESTINATION="platform=macOS"
DERIVED_DATA="$ROOT/tmp/regression/vad-phase6-smoke-derived"
OUTPUT_DIR="$ROOT/tmp/regression/vad-phase6-smoke"
SPM_CACHE_PATH="${VOXT_SPM_CACHE_PATH:-$ROOT/tmp/regression/spm-cache}"
SPM_CLONE_PATH="${VOXT_SPM_CLONE_PATH:-$ROOT/tmp/regression/spm-source-packages}"
DURATION_SECONDS=60
SAMPLE_INTERVAL_SECONDS=10
DAMAGED_CACHE_DURATION_SECONDS=""
DAMAGED_CACHE_INTERVAL_SECONDS=""
PREFERENCES_SMOKE_DURATION_SECONDS=""
PREFERENCES_SMOKE_INTERVAL_SECONDS=""
CODE_SIGNING_ALLOWED="NO"
APP_PATH=""
EXTERNAL_BUILD_LOG=""
REGRESSION_LOG=""
MICROPHONE="TODO"
HEADPHONES="TODO"
NETWORK="TODO"
MODEL_DEBUG_NOTE="TODO"
SKIP_BUILD=false
DAMAGED_CACHE_SMOKE=true
PREFERENCES_SMOKE=true
VAD_LOG_DENYLIST_REGEXES=(
  '(VAD|Silero).*(fatal|Fatal|error|Error)'
  '(fatal|Fatal|error|Error).*(VAD|Silero)'
  '(rawAudio|raw audio|audioSamples|audio samples|sampleBuffer|fullTranscript|full transcript|transcriptText|transcript text|promptText|prompt text|\[VOXT_SMOKE\]\[prompt\])'
)

usage() {
  cat <<'EOF'
Usage: tools/run_vad_phase6_smoke.sh [options]

Builds the Voxt Release app, runs a launch smoke with local VAD / Silero
fatal/error denylist checks, and generates a phase-6 acceptance report.

Options:
  --app PATH                      Use an existing .app and skip xcodebuild.
  --build-log PATH                Existing xcodebuild log for --app/--skip-build.
  --regression-log PATH           Existing local regression matrix log.
  --derived-data PATH             DerivedData path for xcodebuild.
  --output-dir PATH               Directory for build log, smoke log, report.
  --spm-cache PATH                SwiftPM package cache path.
  --spm-clone PATH                SwiftPM cloned source packages path.
  --duration SECONDS              Launch smoke duration. Defaults to 60.
  --interval SECONDS              CPU/RSS sample interval. Defaults to 10.
  --damaged-cache-duration SECONDS
                                  Damaged-cache launch smoke duration.
                                  Defaults to min(duration, 30).
  --damaged-cache-interval SECONDS
                                  Damaged-cache sample interval. Defaults to
                                  the main smoke interval.
  --preferences-duration SECONDS  Preferences launch smoke duration. Defaults
                                  to min(duration, 10).
  --preferences-interval SECONDS  Preferences sample interval. Defaults to the
                                  main smoke interval.
  --code-signing-allowed YES|NO   xcodebuild CODE_SIGNING_ALLOWED. Defaults NO.
  --microphone TEXT               Microphone evidence for the generated report.
  --headphones TEXT               Output device evidence for the generated report.
  --network TEXT                  Network evidence for the generated report.
  --model-debug-note TEXT         Model Debug VAD snapshot evidence.
  --skip-build                    Reuse app from derived data.
  --skip-damaged-cache-smoke      Do not run isolated damaged cache smoke.
  --skip-preferences-smoke        Do not run clean/legacy preferences smoke.
  -h, --help                      Show this help.
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
      SKIP_BUILD=true
      shift 2
      ;;
    --build-log)
      EXTERNAL_BUILD_LOG="${2:-}"
      [[ -n "$EXTERNAL_BUILD_LOG" ]] || die "--build-log requires a value"
      shift 2
      ;;
    --regression-log)
      REGRESSION_LOG="${2:-}"
      [[ -n "$REGRESSION_LOG" ]] || die "--regression-log requires a value"
      shift 2
      ;;
    --derived-data)
      DERIVED_DATA="${2:-}"
      [[ -n "$DERIVED_DATA" ]] || die "--derived-data requires a value"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      [[ -n "$OUTPUT_DIR" ]] || die "--output-dir requires a value"
      shift 2
      ;;
    --spm-cache)
      SPM_CACHE_PATH="${2:-}"
      [[ -n "$SPM_CACHE_PATH" ]] || die "--spm-cache requires a value"
      shift 2
      ;;
    --spm-clone)
      SPM_CLONE_PATH="${2:-}"
      [[ -n "$SPM_CLONE_PATH" ]] || die "--spm-clone requires a value"
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
    --damaged-cache-duration)
      DAMAGED_CACHE_DURATION_SECONDS="${2:-}"
      [[ "$DAMAGED_CACHE_DURATION_SECONDS" =~ ^[0-9]+$ ]] || die "--damaged-cache-duration requires a positive integer"
      shift 2
      ;;
    --damaged-cache-interval)
      DAMAGED_CACHE_INTERVAL_SECONDS="${2:-}"
      [[ "$DAMAGED_CACHE_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || die "--damaged-cache-interval requires a positive integer"
      shift 2
      ;;
    --preferences-duration)
      PREFERENCES_SMOKE_DURATION_SECONDS="${2:-}"
      [[ "$PREFERENCES_SMOKE_DURATION_SECONDS" =~ ^[0-9]+$ ]] || die "--preferences-duration requires a positive integer"
      shift 2
      ;;
    --preferences-interval)
      PREFERENCES_SMOKE_INTERVAL_SECONDS="${2:-}"
      [[ "$PREFERENCES_SMOKE_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || die "--preferences-interval requires a positive integer"
      shift 2
      ;;
    --code-signing-allowed)
      CODE_SIGNING_ALLOWED="${2:-}"
      [[ "$CODE_SIGNING_ALLOWED" == "YES" || "$CODE_SIGNING_ALLOWED" == "NO" ]] || die "--code-signing-allowed must be YES or NO"
      shift 2
      ;;
    --microphone)
      MICROPHONE="${2:-}"
      [[ -n "$MICROPHONE" ]] || die "--microphone requires a value"
      shift 2
      ;;
    --headphones)
      HEADPHONES="${2:-}"
      [[ -n "$HEADPHONES" ]] || die "--headphones requires a value"
      shift 2
      ;;
    --network)
      NETWORK="${2:-}"
      [[ -n "$NETWORK" ]] || die "--network requires a value"
      shift 2
      ;;
    --model-debug-note)
      MODEL_DEBUG_NOTE="${2:-}"
      [[ -n "$MODEL_DEBUG_NOTE" ]] || die "--model-debug-note requires a value"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --skip-damaged-cache-smoke)
      DAMAGED_CACHE_SMOKE=false
      shift
      ;;
    --skip-preferences-smoke)
      PREFERENCES_SMOKE=false
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

[[ "$DURATION_SECONDS" -gt 0 ]] || die "--duration must be greater than 0"
[[ "$SAMPLE_INTERVAL_SECONDS" -gt 0 ]] || die "--interval must be greater than 0"
if [[ -z "$DAMAGED_CACHE_DURATION_SECONDS" ]]; then
  if [[ "$DURATION_SECONDS" -lt 30 ]]; then
    DAMAGED_CACHE_DURATION_SECONDS="$DURATION_SECONDS"
  else
    DAMAGED_CACHE_DURATION_SECONDS=30
  fi
fi
if [[ -z "$DAMAGED_CACHE_INTERVAL_SECONDS" ]]; then
  DAMAGED_CACHE_INTERVAL_SECONDS="$SAMPLE_INTERVAL_SECONDS"
fi
if [[ -z "$PREFERENCES_SMOKE_DURATION_SECONDS" ]]; then
  if [[ "$DURATION_SECONDS" -lt 10 ]]; then
    PREFERENCES_SMOKE_DURATION_SECONDS="$DURATION_SECONDS"
  else
    PREFERENCES_SMOKE_DURATION_SECONDS=10
  fi
fi
if [[ -z "$PREFERENCES_SMOKE_INTERVAL_SECONDS" ]]; then
  PREFERENCES_SMOKE_INTERVAL_SECONDS="$SAMPLE_INTERVAL_SECONDS"
fi
[[ "$DAMAGED_CACHE_DURATION_SECONDS" -gt 0 ]] || die "--damaged-cache-duration must be greater than 0"
[[ "$DAMAGED_CACHE_INTERVAL_SECONDS" -gt 0 ]] || die "--damaged-cache-interval must be greater than 0"
[[ "$PREFERENCES_SMOKE_DURATION_SECONDS" -gt 0 ]] || die "--preferences-duration must be greater than 0"
[[ "$PREFERENCES_SMOKE_INTERVAL_SECONDS" -gt 0 ]] || die "--preferences-interval must be greater than 0"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SPM_CACHE_PATH" "$SPM_CLONE_PATH"

VAD_LOG_DENYLIST_ARGS=()
for regex in "${VAD_LOG_DENYLIST_REGEXES[@]}"; do
  VAD_LOG_DENYLIST_ARGS+=(--fail-on-log "$regex")
done

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BUILD_LOG="$OUTPUT_DIR/release-build-$TIMESTAMP.log"
SMOKE_LOG="$OUTPUT_DIR/release-vad-denylist-smoke-$TIMESTAMP.log"
DAMAGED_CACHE_OUTPUT_DIR="$OUTPUT_DIR/damaged-cache-$TIMESTAMP"
PREFERENCES_OUTPUT_DIR="$OUTPUT_DIR/preferences-$TIMESTAMP"
REPORT_PATH="$OUTPUT_DIR/vad-acceptance-$TIMESTAMP.md"
BUILD_LOG_ARTIFACT=""
DAMAGED_CACHE_SMOKE_LOG=""
PREFERENCES_SMOKE_SUMMARY=""

if [[ -n "$EXTERNAL_BUILD_LOG" ]]; then
  [[ -f "$EXTERNAL_BUILD_LOG" ]] || die "build log not found: $EXTERNAL_BUILD_LOG"
  BUILD_LOG_ARTIFACT="$EXTERNAL_BUILD_LOG"
fi
if [[ -n "$REGRESSION_LOG" ]]; then
  [[ -f "$REGRESSION_LOG" ]] || die "regression log not found: $REGRESSION_LOG"
fi

if [[ "$SKIP_BUILD" != "true" ]]; then
  echo "buildLog=$BUILD_LOG"
  xcodebuild build \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -destination "$DESTINATION" \
    -derivedDataPath "$DERIVED_DATA" \
    -clonedSourcePackagesDirPath "$SPM_CLONE_PATH" \
    -packageCachePath "$SPM_CACHE_PATH" \
    CODE_SIGNING_ALLOWED="$CODE_SIGNING_ALLOWED" \
    | tee "$BUILD_LOG"
  BUILD_LOG_ARTIFACT="$BUILD_LOG"
fi

if [[ -z "$APP_PATH" ]]; then
  APP_PATH="$DERIVED_DATA/Build/Products/$CONFIGURATION/Voxt.app"
fi

[[ -d "$APP_PATH" ]] || die "app bundle not found: $APP_PATH"

"$ROOT/tools/run_app_launch_smoke.sh" \
  --app "$APP_PATH" \
  --duration "$DURATION_SECONDS" \
  --interval "$SAMPLE_INTERVAL_SECONDS" \
  --log "$SMOKE_LOG" \
  "${VAD_LOG_DENYLIST_ARGS[@]}"

if [[ "$DAMAGED_CACHE_SMOKE" == "true" ]]; then
  "$ROOT/tools/run_vad_damaged_cache_smoke.sh" \
    --app "$APP_PATH" \
    --duration "$DAMAGED_CACHE_DURATION_SECONDS" \
    --interval "$DAMAGED_CACHE_INTERVAL_SECONDS" \
    --output-dir "$DAMAGED_CACHE_OUTPUT_DIR"
  DAMAGED_CACHE_SMOKE_LOG="$(find "$DAMAGED_CACHE_OUTPUT_DIR" -type f -name 'damaged-cache-smoke-*.log' | sort | tail -n 1)"
  [[ -n "$DAMAGED_CACHE_SMOKE_LOG" ]] || die "damaged cache smoke log not found under $DAMAGED_CACHE_OUTPUT_DIR"
fi

if [[ "$PREFERENCES_SMOKE" == "true" ]]; then
  "$ROOT/tools/run_vad_preferences_smoke.sh" \
    --app "$APP_PATH" \
    --duration "$PREFERENCES_SMOKE_DURATION_SECONDS" \
    --interval "$PREFERENCES_SMOKE_INTERVAL_SECONDS" \
    --output-dir "$PREFERENCES_OUTPUT_DIR"
  PREFERENCES_SMOKE_SUMMARY="$(find "$PREFERENCES_OUTPUT_DIR" -type f -name 'preferences-smoke-*.summary' | sort | tail -n 1)"
  [[ -n "$PREFERENCES_SMOKE_SUMMARY" ]] || die "preferences smoke summary not found under $PREFERENCES_OUTPUT_DIR"
fi

REPORT_ARGS=(
  --app "$APP_PATH"
  --smoke-log "$SMOKE_LOG"
  --output "$REPORT_PATH"
)
if [[ -n "$BUILD_LOG_ARTIFACT" ]]; then
  REPORT_ARGS+=(--build-log "$BUILD_LOG_ARTIFACT")
fi
if [[ -n "$REGRESSION_LOG" ]]; then
  REPORT_ARGS+=(--regression-log "$REGRESSION_LOG")
fi
if [[ -n "$DAMAGED_CACHE_SMOKE_LOG" ]]; then
  REPORT_ARGS+=(--damaged-cache-smoke-log "$DAMAGED_CACHE_SMOKE_LOG")
fi
if [[ -n "$PREFERENCES_SMOKE_SUMMARY" ]]; then
  REPORT_ARGS+=(--preferences-smoke-summary "$PREFERENCES_SMOKE_SUMMARY")
fi
REPORT_ARGS+=(
  --microphone "$MICROPHONE"
  --headphones "$HEADPHONES"
  --network "$NETWORK"
  --model-debug-note "$MODEL_DEBUG_NOTE"
)

"$ROOT/tools/create_vad_acceptance_report.sh" "${REPORT_ARGS[@]}"

echo "app=$APP_PATH"
echo "smokeLog=$SMOKE_LOG"
if [[ "$DAMAGED_CACHE_SMOKE" == "true" ]]; then
  echo "damagedCacheOutputDir=$DAMAGED_CACHE_OUTPUT_DIR"
  echo "damagedCacheSmokeLog=$DAMAGED_CACHE_SMOKE_LOG"
fi
if [[ "$PREFERENCES_SMOKE" == "true" ]]; then
  echo "preferencesOutputDir=$PREFERENCES_OUTPUT_DIR"
  echo "preferencesSmokeSummary=$PREFERENCES_SMOKE_SUMMARY"
fi
echo "report=$REPORT_PATH"
