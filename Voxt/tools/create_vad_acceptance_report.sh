#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH=""
BUILD_LOG=""
REGRESSION_LOG=""
SMOKE_LOG=""
DAMAGED_CACHE_SMOKE_LOG=""
PREFERENCES_SMOKE_SUMMARY=""
MICROPHONE="TODO"
HEADPHONES="TODO"
NETWORK="TODO"
MODEL_DEBUG_NOTE="TODO"
OUTPUT_DIR="$ROOT/tmp/regression/vad-acceptance"
REPORT_PATH=""

usage() {
  cat <<'EOF'
Usage: tools/create_vad_acceptance_report.sh [options]

Creates a timestamped Markdown report for VAD / ASR phase-6 manual acceptance.
The report captures build identity, local system details, optional app signing
status, optional launch-smoke evidence, and a complete scenario checklist.

Options:
  --app PATH          Optional .app bundle under test.
  --build-log PATH    Optional xcodebuild log for the app under test.
  --regression-log PATH
                      Optional local regression matrix log.
  --smoke-log PATH    Optional log from tools/run_app_launch_smoke.sh.
  --damaged-cache-smoke-log PATH
                      Optional log from tools/run_vad_damaged_cache_smoke.sh.
  --preferences-smoke-summary PATH
                      Optional summary from tools/run_vad_preferences_smoke.sh.
  --microphone TEXT   Microphone used during manual acceptance.
  --headphones TEXT   Output device used during manual acceptance.
  --network TEXT      Network state used during manual acceptance.
  --model-debug-note TEXT
                      Model Debug VAD snapshot evidence or artifact path.
  --output-dir PATH   Output directory. Defaults to tmp/regression/vad-acceptance.
  --output PATH       Exact output file path. Overrides --output-dir.
  -h, --help          Show this help.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

value_or_unknown() {
  local value="${1:-}"
  if [[ -n "$value" ]]; then
    printf '%s\n' "$value"
  else
    printf 'unknown\n'
  fi
}

command_status() {
  local label="$1"
  shift
  local output
  local status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    printf '%s: pass\n' "$label"
  else
    printf '%s: fail (%s)\n' "$label" "$status"
  fi
  if [[ -n "$output" ]]; then
    printf '%s\n' "$output" | sed 's/^/  /'
  fi
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --app)
      APP_PATH="${2:-}"
      [[ -n "$APP_PATH" ]] || die "--app requires a value"
      shift 2
      ;;
    --build-log)
      BUILD_LOG="${2:-}"
      [[ -n "$BUILD_LOG" ]] || die "--build-log requires a value"
      shift 2
      ;;
    --regression-log)
      REGRESSION_LOG="${2:-}"
      [[ -n "$REGRESSION_LOG" ]] || die "--regression-log requires a value"
      shift 2
      ;;
    --smoke-log)
      SMOKE_LOG="${2:-}"
      [[ -n "$SMOKE_LOG" ]] || die "--smoke-log requires a value"
      shift 2
      ;;
    --damaged-cache-smoke-log)
      DAMAGED_CACHE_SMOKE_LOG="${2:-}"
      [[ -n "$DAMAGED_CACHE_SMOKE_LOG" ]] || die "--damaged-cache-smoke-log requires a value"
      shift 2
      ;;
    --preferences-smoke-summary)
      PREFERENCES_SMOKE_SUMMARY="${2:-}"
      [[ -n "$PREFERENCES_SMOKE_SUMMARY" ]] || die "--preferences-smoke-summary requires a value"
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
    --output-dir)
      OUTPUT_DIR="${2:-}"
      [[ -n "$OUTPUT_DIR" ]] || die "--output-dir requires a value"
      shift 2
      ;;
    --output)
      REPORT_PATH="${2:-}"
      [[ -n "$REPORT_PATH" ]] || die "--output requires a value"
      shift 2
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

if [[ -n "$APP_PATH" ]]; then
  [[ -d "$APP_PATH" ]] || die "app bundle not found: $APP_PATH"
  APP_PATH="$(cd "$(dirname "$APP_PATH")" && pwd)/$(basename "$APP_PATH")"
fi

if [[ -n "$SMOKE_LOG" ]]; then
  [[ -f "$SMOKE_LOG" ]] || die "smoke log not found: $SMOKE_LOG"
  SMOKE_LOG="$(cd "$(dirname "$SMOKE_LOG")" && pwd)/$(basename "$SMOKE_LOG")"
fi

if [[ -n "$BUILD_LOG" ]]; then
  [[ -f "$BUILD_LOG" ]] || die "build log not found: $BUILD_LOG"
  BUILD_LOG="$(cd "$(dirname "$BUILD_LOG")" && pwd)/$(basename "$BUILD_LOG")"
fi

if [[ -n "$REGRESSION_LOG" ]]; then
  [[ -f "$REGRESSION_LOG" ]] || die "regression log not found: $REGRESSION_LOG"
  REGRESSION_LOG="$(cd "$(dirname "$REGRESSION_LOG")" && pwd)/$(basename "$REGRESSION_LOG")"
fi

if [[ -n "$DAMAGED_CACHE_SMOKE_LOG" ]]; then
  [[ -f "$DAMAGED_CACHE_SMOKE_LOG" ]] || die "damaged cache smoke log not found: $DAMAGED_CACHE_SMOKE_LOG"
  DAMAGED_CACHE_SMOKE_LOG="$(cd "$(dirname "$DAMAGED_CACHE_SMOKE_LOG")" && pwd)/$(basename "$DAMAGED_CACHE_SMOKE_LOG")"
fi

if [[ -n "$PREFERENCES_SMOKE_SUMMARY" ]]; then
  [[ -f "$PREFERENCES_SMOKE_SUMMARY" ]] || die "preferences smoke summary not found: $PREFERENCES_SMOKE_SUMMARY"
  PREFERENCES_SMOKE_SUMMARY="$(cd "$(dirname "$PREFERENCES_SMOKE_SUMMARY")" && pwd)/$(basename "$PREFERENCES_SMOKE_SUMMARY")"
fi

if [[ -z "$REPORT_PATH" ]]; then
  mkdir -p "$OUTPUT_DIR"
  REPORT_PATH="$OUTPUT_DIR/vad-acceptance-$(date +%Y%m%d-%H%M%S).md"
else
  mkdir -p "$(dirname "$REPORT_PATH")"
fi

BRANCH="$(git -C "$ROOT" branch --show-current 2>/dev/null || true)"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || true)"
DIRTY_COUNT="$(git -C "$ROOT" status --short 2>/dev/null | wc -l | tr -d ' ')"
STATUS_SUMMARY="$(git -C "$ROOT" status --short 2>/dev/null | sed -n '1,40p' || true)"
MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || true)"
MACOS_BUILD="$(sw_vers -buildVersion 2>/dev/null || true)"
HOST_ARCH="$(uname -m)"
HARDWARE_MODEL="$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Model Name|Model Identifier|Chip|Processor Name|Memory/ {print $1 ": " $2}' | sed 's/^ *//' | paste -sd ';' - | sed 's/;/; /g' || true)"
TIMESTAMP_LOCAL="$(date '+%Y-%m-%d %H:%M:%S %Z')"
TIMESTAMP_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

APP_VERSION="not provided"
APP_BUILD="not provided"
APP_EXECUTABLE="not provided"
APP_EXECUTABLE_SHA256="not provided"
APP_QUARANTINE="not provided"
CODESIGN_STATUS="not provided"
SPCTL_STATUS="not provided"

if [[ -n "$APP_PATH" ]]; then
  INFO_PLIST="$APP_PATH/Contents/Info.plist"
  if [[ -f "$INFO_PLIST" ]]; then
    APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$INFO_PLIST" 2>/dev/null || true)"
    APP_BUILD="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$INFO_PLIST" 2>/dev/null || true)"
    EXECUTABLE_NAME="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$INFO_PLIST" 2>/dev/null || true)"
    if [[ -n "$EXECUTABLE_NAME" && -x "$APP_PATH/Contents/MacOS/$EXECUTABLE_NAME" ]]; then
      APP_EXECUTABLE="$APP_PATH/Contents/MacOS/$EXECUTABLE_NAME"
      APP_EXECUTABLE_SHA256="$(shasum -a 256 "$APP_EXECUTABLE" | awk '{print $1}')"
    fi
  fi
  APP_VERSION="$(value_or_unknown "$APP_VERSION")"
  APP_BUILD="$(value_or_unknown "$APP_BUILD")"
  APP_QUARANTINE="$(xattr -p com.apple.quarantine "$APP_PATH" 2>/dev/null || true)"
  APP_QUARANTINE="$(value_or_unknown "$APP_QUARANTINE")"
  CODESIGN_STATUS="$(command_status "codesign --verify" codesign --verify --deep --strict --verbose=2 "$APP_PATH")"
  SPCTL_STATUS="$(command_status "spctl --assess" spctl --assess --type execute -vv "$APP_PATH")"
fi

summarize_smoke_log() {
  local log_path="$1"
  local denylist_hits
  local termination
  local duration
  local pid
  local cpu_rss

  denylist_hits="$(grep -E 'logFailureRegex|processExitedBeforeEnd=true' "$log_path" || true)"
  termination="$(grep -E '^terminated=' "$log_path" | tail -n 1 || true)"
  duration="$(grep -E '^durationSeconds=' "$log_path" | tail -n 1 || true)"
  pid="$(grep -E '^pid=' "$log_path" | tail -n 1 || true)"
  cpu_rss="$(
    awk '
      /^[[:space:]]*[0-9]+[[:space:]]/ {
        sampleCount += 1
        cpu=$3 + 0
        rss=$4 + 0
        if (!seen || cpu < minCpu) minCpu=cpu
        if (!seen || cpu > maxCpu) maxCpu=cpu
        if (!seen || rss < minRss) minRss=rss
        if (!seen || rss > maxRss) maxRss=rss
        if (sampleCount > 1 && rss >= 1024) {
          if (!stableSeen || rss < stableMinRss) stableMinRss=rss
          if (!stableSeen || rss > stableMaxRss) stableMaxRss=rss
          stableSeen=1
        }
        seen=1
      }
      END {
        if (seen) {
          if (stableSeen) {
            minRss=stableMinRss
            maxRss=stableMaxRss
          }
          printf "cpu=%.1f-%.1f%% rss=%d-%dKB", minCpu, maxCpu, minRss, maxRss
        } else {
          printf "no samples"
        }
      }
    ' "$log_path"
  )"
  if [[ -z "$denylist_hits" ]]; then
    denylist_hits="none"
  fi
  printf '%s; %s; %s; %s; denylist=%s\n' "$duration" "$pid" "$cpu_rss" "$termination" "$denylist_hits"
}

SMOKE_SUMMARY="not provided"
if [[ -n "$SMOKE_LOG" ]]; then
  SMOKE_SUMMARY="$(summarize_smoke_log "$SMOKE_LOG")"
fi

DAMAGED_CACHE_SMOKE_SUMMARY="not provided"
if [[ -n "$DAMAGED_CACHE_SMOKE_LOG" ]]; then
  DAMAGED_CACHE_SMOKE_SUMMARY="$(summarize_smoke_log "$DAMAGED_CACHE_SMOKE_LOG")"
fi

PREFERENCES_SMOKE_RESULT="not provided"
if [[ -n "$PREFERENCES_SMOKE_SUMMARY" ]]; then
  PREFERENCES_SMOKE_RESULT="$(tr '\n' ';' < "$PREFERENCES_SMOKE_SUMMARY" | sed 's/; */; /g; s/; $//')"
fi

BUILD_SUMMARY="not provided"
if [[ -n "$BUILD_LOG" ]]; then
  if grep -F '** BUILD SUCCEEDED **' "$BUILD_LOG" >/dev/null 2>&1; then
    BUILD_SUMMARY="pass; log=$BUILD_LOG"
  else
    BUILD_SUMMARY="fail; log=$BUILD_LOG"
  fi
fi

REGRESSION_SUMMARY="not provided"
if [[ -n "$REGRESSION_LOG" ]]; then
  if grep -F '** TEST SUCCEEDED **' "$REGRESSION_LOG" >/dev/null 2>&1 \
    && ! grep -E '\*\* TEST FAILED \*\*|Failing tests:|Completed with failures|!! Group failed' "$REGRESSION_LOG" >/dev/null 2>&1; then
    REGRESSION_SUMMARY="pass; log=$REGRESSION_LOG"
  else
    REGRESSION_SUMMARY="fail; log=$REGRESSION_LOG"
  fi
fi

cat > "$REPORT_PATH" <<EOF
# VAD / ASR Phase 6 Acceptance Report

Generated: $TIMESTAMP_LOCAL ($TIMESTAMP_UTC)

## Build Identity

- Branch: $(value_or_unknown "$BRANCH")
- Commit: $(value_or_unknown "$COMMIT")
- Dirty files: $DIRTY_COUNT
- App: $(value_or_unknown "$APP_PATH")
- App version: $APP_VERSION
- App build: $APP_BUILD
- Executable: $APP_EXECUTABLE
- Executable sha256: $APP_EXECUTABLE_SHA256

## Local Environment

- macOS: $(value_or_unknown "$MACOS_VERSION") ($(value_or_unknown "$MACOS_BUILD"))
- Host arch: $HOST_ARCH
- Hardware: $(value_or_unknown "$HARDWARE_MODEL")
- Microphone: $MICROPHONE
- Headphones / output device: $HEADPHONES
- Network: $NETWORK

## App Trust And Packaging

- Quarantine: $APP_QUARANTINE

\`\`\`text
$CODESIGN_STATUS
\`\`\`

\`\`\`text
$SPCTL_STATUS
\`\`\`

## Automated Evidence

- Regression matrix log: $(value_or_unknown "$REGRESSION_LOG")
- Regression matrix summary: $REGRESSION_SUMMARY
- Release build log: $(value_or_unknown "$BUILD_LOG")
- Release build summary: $BUILD_SUMMARY
- Launch smoke log: $(value_or_unknown "$SMOKE_LOG")
- Launch smoke summary: $SMOKE_SUMMARY
- Damaged cache smoke log: $(value_or_unknown "$DAMAGED_CACHE_SMOKE_LOG")
- Damaged cache smoke summary: $DAMAGED_CACHE_SMOKE_SUMMARY
- Preferences smoke summary: $(value_or_unknown "$PREFERENCES_SMOKE_SUMMARY")
- Preferences smoke result: $PREFERENCES_SMOKE_RESULT
- Model Debug VAD snapshot screenshot or notes: $MODEL_DEBUG_NOTE

## Manual Scenario Checklist

| Area | Scenario | Backend / Mode | Status | Evidence / Notes |
| --- | --- | --- | --- | --- |
| Meeting | Single clear speaker, 5 min with silence before/after speech | VAD Mode: Automatic | TODO |  |
| Meeting | Single clear speaker, 5 min with silence before/after speech | VAD Mode: Silero | TODO |  |
| Meeting | Silent room, 3 min | VAD Mode: Automatic / Silero | TODO |  |
| Meeting | Keyboard and mouse noise, 2 min | VAD Mode: Automatic / Silero | TODO |  |
| Meeting | Music or video background, then speech | VAD Mode: Automatic / Silero | TODO |  |
| Meeting | Far-field low-volume speech, 3 min | VAD Mode: Automatic / Silero | TODO |  |
| Meeting | Two speakers with overlap, 5 min | VAD Mode: Automatic / Silero | TODO |  |
| Meeting | Energy-only comparison | VAD Mode: Energy | TODO |  |
| Meeting | VAD disabled behavior | VAD Mode: Off | TODO |  |
| Recording | Transcription with local VAD mode | Automatic / Silero / Off | TODO |  |
| Recording | Translation with local VAD mode | Automatic / Silero / Off | TODO |  |
| Recording | Rewrite with local VAD mode | Automatic / Silero / Off | TODO |  |
| Recording | Long-tail recording, at least 30 min | Automatic / Silero | TODO |  |
| Operations | 20 rapid start/stop cycles | Hotkey | TODO |  |
| Operations | ESC cancel during recording | Capture + VAD + ASR + LLM | TODO |  |
| Operations | Voice end command | Hotkey settings | TODO |  |
| Operations | Device hotplug during meeting and recording | USB mic / headset | TODO |  |
| Operations | Sleep/wake while idle and while recording | Capture + VAD actors | TODO |  |
| Operations | Microphone permission revoke/restore | macOS privacy | TODO |  |
| Stability | Release idle, 300 sec | Launch smoke | TODO |  |
| Stability | 30 min real recording and meeting | CPU/RSS/energy | TODO |  |
| Stability | 2 hour long recording | Memory / final transcript | TODO |  |
| Stability | 8 hour silence soak | Segment volume / memory | TODO |  |
| Stability | Damaged MLX Silero model cache | VAD Mode: Silero | TODO |  |
| Packaging | Unsigned local package launch | --no-sign | TODO |  |
| Packaging | Developer ID signed launch | Signed app | TODO |  |
| Packaging | Notarized dmg/pkg first launch | Gatekeeper | TODO |  |
| Packaging | First install with clean preferences | Defaults | TODO |  |
| Packaging | Global VAD mode preferences | Automatic / Silero / Energy / Off | TODO |  |

## Blocking Criteria

- Any crash, capture deadlock, inability to stop/cancel, repeated final commit, or bulk empty ASR is blocking.
- Any log that contains raw audio content, full transcript text, or unredacted private user content is blocking.
- Rows marked PASS/PASSED/通过/已通过/N/A/NA must include evidence, an artifact path, or an explicit reason in Evidence / Notes.
- Automatic and Silero modes must not increase empty ASR commits, startup crashes, or stop/cancel latency.
- Off mode must preserve capture, final ASR, ESC cancel, and voice end command behavior even when local VAD gating is disabled.

## Issues

| ID | Severity | Scenario | Repro Steps | Expected | Actual | Logs / Artifacts | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| None | N/A | No open issues at report generation | N/A | N/A | N/A | N/A | N/A |
EOF

echo "report=$REPORT_PATH"
