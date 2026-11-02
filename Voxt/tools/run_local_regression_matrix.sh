#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$ROOT/Voxt.xcodeproj"
SCHEME="Voxt"
CONFIGURATION="TestDebug"
DESTINATION="platform=macOS"
SPM_CACHE_PATH="${VOXT_SPM_CACHE_PATH:-$ROOT/tmp/regression/spm-cache}"
SPM_CLONE_PATH="${VOXT_SPM_CLONE_PATH:-$ROOT/tmp/regression/spm-source-packages}"

if [[ "${CI:-}" == "true" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "Local regression matrix is intended for local machines only." >&2
  exit 1
fi

GROUP="${1:-all}"

run_tests() {
  local label="$1"
  shift
  echo
  echo "==> Running $label"
  if model_tests_enabled; then
    run_tests_with_model_gate "$label" "$@"
  else
    mkdir -p "$SPM_CACHE_PATH" "$SPM_CLONE_PATH"
    xcodebuild test \
      -project "$PROJECT" \
      -scheme "$SCHEME" \
      -configuration "$CONFIGURATION" \
      -destination "$DESTINATION" \
      -clonedSourcePackagesDirPath "$SPM_CLONE_PATH" \
      -packageCachePath "$SPM_CACHE_PATH" \
      -onlyUsePackageVersionsFromResolvedFile \
      CODE_SIGNING_ALLOWED=NO \
      "$@"
  fi
}

model_tests_enabled() {
  case "${VOXT_RUN_MODEL_TESTS:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

plist_set_or_add() {
  local plist="$1"
  local key_path="$2"
  local value="$3"
  /usr/libexec/PlistBuddy -c "Set $key_path $value" "$plist" 2>/dev/null \
    || /usr/libexec/PlistBuddy -c "Add $key_path string $value" "$plist"
}

run_tests_with_model_gate() {
  local label="$1"
  shift
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local derived_data="$ROOT/tmp/regression/model-derived-$stamp"
  rm -rf "$derived_data"
  mkdir -p "$SPM_CACHE_PATH" "$SPM_CLONE_PATH"

  echo "==> Building test products for $label with VOXT_RUN_MODEL_TESTS=1"
  xcodebuild build-for-testing \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -configuration "$CONFIGURATION" \
    -destination "$DESTINATION" \
    -derivedDataPath "$derived_data" \
    -clonedSourcePackagesDirPath "$SPM_CLONE_PATH" \
    -packageCachePath "$SPM_CACHE_PATH" \
    -onlyUsePackageVersionsFromResolvedFile \
    CODE_SIGNING_ALLOWED=NO \
    "$@"

  local xctestrun
  xctestrun="$(find "$derived_data/Build/Products" -name '*.xctestrun' -print -quit)"
  if [[ -z "$xctestrun" ]]; then
    echo "Could not find .xctestrun under $derived_data/Build/Products" >&2
    return 1
  fi

  plist_set_or_add "$xctestrun" ":VoxtTests:EnvironmentVariables:VOXT_RUN_MODEL_TESTS" "1"
  plist_set_or_add "$xctestrun" ":VoxtTests:TestingEnvironmentVariables:VOXT_RUN_MODEL_TESTS" "1"

  echo "==> Running $label from $xctestrun"
  xcodebuild test-without-building \
    -xctestrun "$xctestrun" \
    -destination "$DESTINATION" \
    "$@"
}

run_group_collecting_failures() {
  local overall=0
  local labels=()
  while [[ "$#" -gt 0 ]]; do
    local group_name="$1"
    shift
    labels+=("$group_name")
    set +e
    "$group_name"
    local status=$?
    set -e
    if [[ $status -ne 0 ]]; then
      overall=$status
      echo
      echo "!! Group failed: $group_name (exit $status)"
    fi
  done

  if [[ $overall -ne 0 ]]; then
    echo
    echo "Completed with failures across groups: ${labels[*]}"
    return "$overall"
  fi
}

run_core() {
  run_tests "core pipeline/runtime regression" \
    -only-testing:VoxtTests/TranscriptionCapturePipelineTests \
    -only-testing:VoxtTests/SessionTimingSummarySupportTests \
    -only-testing:VoxtTests/SessionTextIOTests \
    -only-testing:VoxtTests/SessionEndFlowTests \
    -only-testing:VoxtTests/LLMExecutionPlanCompilerTests \
    -only-testing:VoxtTests/EnhancementPromptResolverTests \
    -only-testing:VoxtTests/PromptBuildersTests \
    -only-testing:VoxtTests/AppPromptDefaultsTests \
    -only-testing:VoxtTests/ASRVoiceActivityPlanningTests \
    -only-testing:VoxtTests/FeatureSettingsStoreTests \
    -only-testing:VoxtTests/MLXTranscriptionPlanningTests \
    -only-testing:VoxtTests/ModelDebugSupportTests
}

# Complements run_core; globbed suite families keep split tests in the focused gate.
run_refactor() {
  local selectors=(
    -only-testing:VoxtTests/OnboardingGuideTests
    -only-testing:VoxtTests/SettingsTypesTests
    -only-testing:VoxtTests/SettingsPermissionSupportTests
    -only-testing:VoxtTests/ASRHintSettingsTests
    -only-testing:VoxtTests/MeetingStartPlannerTests
    -only-testing:VoxtTests/MeetingASRSupportTests
    -only-testing:VoxtTests/RemoteASRSupportTests
    -only-testing:VoxtTests/MeetingLiveSessionSupportTests
    -only-testing:VoxtTests/DoubaoPacketCodecTests
    -only-testing:VoxtTests/RemoteASRResponseStateTests
    -only-testing:VoxtTests/RemoteASRCompletionTests
    -only-testing:VoxtTests/MeetingRemoteSessionLifecycleTests
    -only-testing:VoxtTests/TrackedTaskStoreTests
    -only-testing:VoxtTests/LLMRequestLifecycleTests
    -only-testing:VoxtTests/MeetingLiveSessionRegistryTests
    -only-testing:VoxtTests/MLXCorrectionPassCoordinatorTests
    -only-testing:VoxtTests/MLXNativeLiveRuntimeTests
    -only-testing:VoxtTests/HotkeyEventTapRunLoopTests
    -only-testing:VoxtTests/SharedModelLoadCoordinatorTests
    -only-testing:VoxtTests/MeetingImportedFileAnalyzerTests
    -only-testing:VoxtTests/MeetingFileTaskQueueTests
    -only-testing:VoxtTests/MeetingFinalizationContextTests
    -only-testing:VoxtTests/MeetingFinalizationCheckpointStoreTests
    -only-testing:VoxtTests/RecordingSessionLifecycleTests
    -only-testing:VoxtTests/TextInjectionTransactionTests
    -only-testing:VoxtTests/PasteboardTextWriterTests
    -only-testing:VoxtTests/RemoteProviderConnectivityTesterTests
    -only-testing:VoxtTests/RemoteProviderConfigurationPolicyTests
    -only-testing:VoxtTests/DictionarySuggestionStoreTests
    -only-testing:VoxtTests/DictionarySuggestionReloadTests
    -only-testing:VoxtTests/RemoteProviderSheetOperationsTests
    -only-testing:VoxtTests/ConnectivityWebSocketSessionTests
    -only-testing:VoxtTests/RemoteASRPreviewControllerTests
    -only-testing:VoxtTests/ModelDownloadProgressTests
    -only-testing:VoxtTests/HotkeyPreferenceCorruptStorageTests
    -only-testing:VoxtTests/MLXInferenceConfigurationTests
    -only-testing:VoxtTests/MeetingLiveTranscriptPresentationTests
    -only-testing:VoxtTests/MeetingCaptureTimelineTests
    -only-testing:VoxtTests/ModelDownloadStatusSnapshotTests
    -only-testing:VoxtTests/RemoteEndpointSecurityPolicyTests
    -only-testing:VoxtTests/RewriteAnswerContentNormalizerTests
    -only-testing:VoxtTests/RewriteAnswerPayloadParserTests
    -only-testing:VoxtTests/CustomLLMModelConfigurationTests
    -only-testing:VoxtTests/CustomLLMRequestRuntimeTests
    -only-testing:VoxtTests/CustomLLMModelSupportTests
    -only-testing:VoxtTests/CustomLLMModelDownloadSupportTests
    -only-testing:VoxtTests/ModelDownloadSourceSupportTests
    -only-testing:VoxtTests/ModelInstallationCacheTests
    -only-testing:VoxtTests/ModelConfigurationIssueResolverTests
    -only-testing:VoxtTests/VoxtLogRedactorTests
    -only-testing:VoxtTests/MLXModelSupportTests
    -only-testing:VoxtTests/MLXModelPerRepoStateSupportTests
    -only-testing:VoxtTests/GGUFUTF8OutputAccumulatorTests/testWaitsForCompleteMultibyteSequenceBeforeDecoding
    -only-testing:VoxtTests/GGUFUTF8OutputAccumulatorTests/testFinalizesInvalidUTF8WithReplacementFlag
    -only-testing:VoxtTests/GGUFUTF8OutputAccumulatorTests/testApplicationTerminationShutdownRejectsNewGGUFInference
    -only-testing:VoxtTests/SQLiteStorageRepositoryTests
    -only-testing:VoxtTests/AutomaticDictionaryLearningMonitorTests
    -only-testing:VoxtTests/DictionaryEntryCollectionTests
    -only-testing:VoxtTests/DictionaryMatcherTests
    -only-testing:VoxtTests/DictionaryMatcherAliasTests
    -only-testing:VoxtTests/DictionaryStoreAsyncTests
    -only-testing:VoxtTests/TranscriptionHistoryStoreAsyncTests
    -only-testing:VoxtTests/TranscriptionHistoryEntryAudioTests
    -only-testing:VoxtTests/TranscriptionHistoryConversationSupportTests
    -only-testing:VoxtTests/HistoryValueResolverTests
    -only-testing:VoxtTests/HistoryCorrectionPresentationTests
    -only-testing:VoxtTests/MeetingDetailFormattingTests
    -only-testing:VoxtTests/MeetingTranscriptVirtualListTests
    -only-testing:VoxtTests/MeetingDetailTranscriptListCacheTests
    -only-testing:VoxtTests/VoxtNoteStoreTests
    -only-testing:VoxtTests/VoxtObsidianSyncCoordinatorTests
    -only-testing:VoxtTests/VoxtRemindersSyncCoordinatorTests
  )
  local path suite
  for path in \
    "$ROOT"/VoxtTests/RemoteLLMRuntimeClient*Tests.swift \
    "$ROOT"/VoxtTests/RemoteModelConfiguration*Tests.swift \
    "$ROOT"/VoxtTests/HotkeyManager*Tests.swift \
    "$ROOT"/VoxtTests/MLXModelManager*Tests.swift \
    "$ROOT"/VoxtTests/MeetingDetailViewModel*Tests.swift; do
    suite="$(basename "$path" .swift)"
    selectors+=("-only-testing:VoxtTests/$suite")
  done
  run_tests "refactoring behavior contracts" "${selectors[@]}"
}

run_mlx() {
  run_tests "MLX public fixture regression" \
    -only-testing:VoxtTests/QwenOfficialFixtureASRIntegrationTests \
    -only-testing:VoxtTests/MLXLongFormReplayIntegrationTests \
    -only-testing:VoxtTests/MLXFinalOnlyReplayIntegrationTests \
    -only-testing:VoxtTests/MLXRealtimeReplayIntegrationTests \
    -only-testing:VoxtTests/MLXPipelineMetricsIntegrationTests
}

run_gguf() {
  run_tests_with_model_gate "GGUF native termination regression" \
    -only-testing:VoxtTests/GGUFUTF8OutputAccumulatorTests/testInstalledGGUFModelIsExplicitlyReleasedDuringApplicationTermination
}

run_vad() {
  run_tests "local VAD planning regression" \
    -only-testing:VoxtTests/ASRVoiceActivityPlanningTests \
    -only-testing:VoxtTests/FeatureSettingsStoreTests \
    -only-testing:VoxtTests/ModelDebugSupportTests
}

run_installed_matrix() {
  run_tests "installed-model long-form matrix" \
    -only-testing:VoxtTests/InstalledASRLongFormMatrixIntegrationTests
}

case "$GROUP" in
  core)
    run_core
    ;;
  refactor)
    run_group_collecting_failures run_core run_refactor
    ;;
  mlx)
    run_mlx
    ;;
  gguf)
    run_gguf
    ;;
  vad)
    run_vad
    ;;
  installed)
    run_installed_matrix
    ;;
  all)
    # VAD's three suites are already included in core; do not run them twice.
    run_group_collecting_failures run_core run_refactor run_mlx
    ;;
  full)
    run_group_collecting_failures run_core run_refactor run_mlx run_gguf run_installed_matrix
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    echo "Usage: $0 [core|refactor|mlx|gguf|vad|installed|all|full]" >&2
    exit 2
    ;;
esac
