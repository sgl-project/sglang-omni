#!/bin/bash
# Complete offline suite. Optional arguments select individual Swift fixtures.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf 'This suite requires macOS on Apple Silicon.\n' >&2
  exit 1
fi
tests=(
  ClientRegression Insertion Feedback HoverFeedback FallbackFeedback
  Personalization PolishFidelity ServiceConfiguration Settings
  TargetObservation GeneralTarget ClipboardPaste CurrentCursor CurrentCursorFeedback
  Timing RecordingLimit CompletionFeedback ClientComposition HTTPTransport
)
if [[ $# -gt 0 ]]; then tests=("$@"); fi
test_dir="$project_dir/.build/offline-tests"
mkdir -p "$test_dir/module-cache"
sdk_path="$(xcrun --show-sdk-path)"
for test_name in "${tests[@]}"; do
  app_sources=()
  support_sources=()
  case "$test_name" in
    ClientRegression|RecordingLimit) app_sources=(AudioCapture) ;;
    Timing) app_sources=(TimingPresentation) ;;
    ServiceConfiguration) app_sources=(ClientPreferences ServicePreferences RecordingShortcut) ;;
    Settings) app_sources=(ClientPreferences RecordingShortcut) ;;
    TargetObservation|GeneralTarget) app_sources=(ClipboardPaste TextInsertion) ;;
    ClipboardPaste) app_sources=(ClipboardPaste) ;;
    CurrentCursor|CurrentCursorFeedback) app_sources=(ClipboardPaste CurrentCursorTarget) ;;
    ClientComposition)
      app_sources=(ClientState ClientPreferences ServicePreferences RecordingShortcut
                   AudioCapture ClipboardPaste CurrentCursorTarget TextInsertion)
      support_sources=("$project_dir/Tests/Support/HTTPStub.swift") ;;
    HTTPTransport) support_sources=("$project_dir/Tests/Support/HTTPStub.swift") ;;
    Insertion|Feedback|HoverFeedback|FallbackFeedback|Personalization|PolishFidelity|CompletionFeedback) ;;
    *) printf 'Unknown offline fixture: %s\n' "$test_name" >&2; exit 1 ;;
  esac
  sources=("$project_dir"/Sources/DictationCore/*.swift)
  # macOS ships Bash 3.2: empty arrays need this expansion under set -u.
  for source in ${app_sources[@]+"${app_sources[@]}"}; do
    sources+=("$project_dir/Sources/DictationApp/$source.swift")
  done
  printf '\nRunning %s\n' "$test_name"
  xcrun swiftc -swift-version 5 -parse-as-library -target arm64-apple-macosx14.0 \
    -module-cache-path "$test_dir/module-cache" -sdk "$sdk_path" \
    "${sources[@]}" ${support_sources[@]+"${support_sources[@]}"} "$project_dir/Tests/${test_name}_test.swift" \
    -o "$test_dir/${test_name}_test"
  "$test_dir/${test_name}_test"
done
printf '\nPASS: all selected offline fixtures. Package, build and live checks are separate.\n'
