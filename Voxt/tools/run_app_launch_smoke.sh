#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH=""
DURATION_SECONDS=30
SAMPLE_INTERVAL_SECONDS=5
LOG_PATH="$ROOT/tmp/regression/app-launch-smoke-$(date +%Y%m%d-%H%M%S).log"
TERMINATE_APP=true
FAIL_ON_LOG_REGEXES=()
LAUNCH_MODE="open"
LAUNCH_ENV=()
LAUNCH_ARGS=()

usage() {
  cat <<'EOF'
Usage: tools/run_app_launch_smoke.sh --app PATH [options]

Launches a macOS .app bundle, verifies the exact launched executable remains
alive, samples CPU/RSS for a short idle window, captures recent unified logs,
and terminates only the launched process by default.

Options:
  --app PATH              Required .app bundle path.
  --duration SECONDS      Monitoring duration. Defaults to 30.
  --interval SECONDS      CPU/RSS sample interval. Defaults to 5.
  --log PATH              Output log path. Defaults under tmp/regression.
  --fail-on-log REGEX     Fail if the captured unified log matches REGEX.
                          Can be passed multiple times.
  --launch-mode MODE      Launch with "open" or by running the app executable
                          directly. Defaults to open.
  --env KEY=VALUE         Environment variable for direct launch mode. Can be
                          passed multiple times.
  --arg VALUE             Argument passed to the launched app. Can be passed
                          multiple times.
  --leave-running         Do not terminate the launched app.
  -h, --help              Show this help.
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
    --log)
      LOG_PATH="${2:-}"
      [[ -n "$LOG_PATH" ]] || die "--log requires a value"
      shift 2
      ;;
    --fail-on-log)
      [[ -n "${2:-}" ]] || die "--fail-on-log requires a value"
      FAIL_ON_LOG_REGEXES+=("$2")
      shift 2
      ;;
    --launch-mode)
      LAUNCH_MODE="${2:-}"
      [[ "$LAUNCH_MODE" == "open" || "$LAUNCH_MODE" == "direct" ]] || die "--launch-mode must be open or direct"
      shift 2
      ;;
    --env)
      [[ -n "${2:-}" ]] || die "--env requires KEY=VALUE"
      [[ "$2" == *=* ]] || die "--env requires KEY=VALUE"
      LAUNCH_ENV+=("$2")
      shift 2
      ;;
    --arg)
      [[ -n "${2:-}" ]] || die "--arg requires a value"
      LAUNCH_ARGS+=("$2")
      shift 2
      ;;
    --leave-running)
      TERMINATE_APP=false
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
[[ "$DURATION_SECONDS" -gt 0 ]] || die "--duration must be greater than 0"
[[ "$SAMPLE_INTERVAL_SECONDS" -gt 0 ]] || die "--interval must be greater than 0"
[[ -d "$APP_PATH" ]] || die "app bundle not found: $APP_PATH"
if [[ "$LAUNCH_MODE" == "open" && "${#LAUNCH_ENV[@]}" -gt 0 ]]; then
  die "--env is only supported with --launch-mode direct"
fi

APP_PATH="$(cd "$(dirname "$APP_PATH")" && pwd)/$(basename "$APP_PATH")"
INFO_PLIST="$APP_PATH/Contents/Info.plist"
[[ -f "$INFO_PLIST" ]] || die "Info.plist not found: $INFO_PLIST"

EXECUTABLE_NAME="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$INFO_PLIST" 2>/dev/null || true)"
[[ -n "$EXECUTABLE_NAME" ]] || die "unable to read CFBundleExecutable from $INFO_PLIST"
EXECUTABLE_PATH="$APP_PATH/Contents/MacOS/$EXECUTABLE_NAME"
[[ -x "$EXECUTABLE_PATH" ]] || die "executable not found: $EXECUTABLE_PATH"

mkdir -p "$(dirname "$LOG_PATH")"
: > "$LOG_PATH"

find_launched_pid() {
  ps -axo pid=,args= | awk -v executable="$EXECUTABLE_PATH" 'index($0, executable) { print $1; exit }' || true
}

{
  echo "app=$APP_PATH"
  echo "executable=$EXECUTABLE_PATH"
  echo "durationSeconds=$DURATION_SECONDS"
  echo "sampleIntervalSeconds=$SAMPLE_INTERVAL_SECONDS"
  echo "launchMode=$LAUNCH_MODE"
  if [[ "${#FAIL_ON_LOG_REGEXES[@]}" -gt 0 ]]; then
    for regex in "${FAIL_ON_LOG_REGEXES[@]}"; do
      echo "failOnLogRegex=$regex"
    done
  else
    echo "failOnLogRegex=none"
  fi
  if [[ "${#LAUNCH_ENV[@]}" -gt 0 ]]; then
    printf 'launchEnv=%q ' "${LAUNCH_ENV[@]}"
    printf '\n'
  fi
  if [[ "${#LAUNCH_ARGS[@]}" -gt 0 ]]; then
    printf 'launchArgs=%q ' "${LAUNCH_ARGS[@]}"
    printf '\n'
  fi
  echo "startedAt=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOG_PATH"

PID=""
if [[ "$LAUNCH_MODE" == "direct" ]]; then
  env "${LAUNCH_ENV[@]}" "$EXECUTABLE_PATH" "${LAUNCH_ARGS[@]}" >> "$LOG_PATH" 2>&1 &
  PID="$!"
else
  if [[ "${#LAUNCH_ARGS[@]}" -gt 0 ]]; then
    open -n "$APP_PATH" --args "${LAUNCH_ARGS[@]}"
  else
    open -n "$APP_PATH"
  fi

  for _ in {1..30}; do
    PID="$(find_launched_pid)"
    if [[ -n "$PID" ]]; then
      break
    fi
    sleep 1
  done
fi

[[ -n "$PID" ]] || {
  echo "launchResult=no_pid" >> "$LOG_PATH"
  exit 1
}

echo "pid=$PID" >> "$LOG_PATH"
echo "samples:" >> "$LOG_PATH"

END_TIME=$((SECONDS + DURATION_SECONDS))
while [[ "$SECONDS" -lt "$END_TIME" ]]; do
  if ! ps -p "$PID" >/dev/null 2>&1; then
    echo "processExitedBeforeEnd=true" >> "$LOG_PATH"
    exit 1
  fi
  ps -p "$PID" -o pid=,stat=,%cpu=,rss=,etime=,command= >> "$LOG_PATH"
  sleep "$SAMPLE_INTERVAL_SECONDS"
done

if ! ps -p "$PID" >/dev/null 2>&1; then
  echo "processExitedBeforeEnd=true" >> "$LOG_PATH"
  exit 1
fi

LOG_WINDOW_SECONDS=$((DURATION_SECONDS + 30))
PREDICATE="process == \"$EXECUTABLE_NAME\" && processID == $PID"
UNIFIED_LOG_CAPTURE="$(mktemp "${TMPDIR:-/tmp}/voxt-launch-smoke-log.XXXXXX")"
{
  echo "processAfterMonitor=$(ps -p "$PID" -o pid=,stat=,%cpu=,rss=,etime=,command= || true)"
  echo "unifiedLogPredicate=$PREDICATE"
  /usr/bin/log show --style compact --last "${LOG_WINDOW_SECONDS}s" --predicate "$PREDICATE" 2>&1 || true
} > "$UNIFIED_LOG_CAPTURE"
cat "$UNIFIED_LOG_CAPTURE" >> "$LOG_PATH"

LOG_SCAN_FAILED=false
for regex in "${FAIL_ON_LOG_REGEXES[@]}"; do
  if grep -E "$regex" "$UNIFIED_LOG_CAPTURE" >/dev/null 2>&1; then
    echo "logFailureRegex=$regex" >> "$LOG_PATH"
    LOG_SCAN_FAILED=true
  fi
done
rm -f "$UNIFIED_LOG_CAPTURE"

if [[ "$TERMINATE_APP" == "true" ]]; then
  kill -TERM "$PID" 2>/dev/null || true
  if [[ "$LAUNCH_MODE" == "direct" ]]; then
    wait "$PID" 2>/dev/null || true
    if ps -p "$PID" >/dev/null 2>&1; then
      kill -KILL "$PID" 2>/dev/null || true
      wait "$PID" 2>/dev/null || true
      echo "terminated=kill" >> "$LOG_PATH"
    else
      echo "terminated=term" >> "$LOG_PATH"
    fi
  else
    sleep 2
    if ps -p "$PID" >/dev/null 2>&1; then
      kill -KILL "$PID" 2>/dev/null || true
      echo "terminated=kill" >> "$LOG_PATH"
    else
      echo "terminated=term" >> "$LOG_PATH"
    fi
  fi
else
  echo "terminated=left-running" >> "$LOG_PATH"
fi

echo "finishedAt=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_PATH"
echo "log=$LOG_PATH"

if [[ "$LOG_SCAN_FAILED" == "true" ]]; then
  exit 1
fi
