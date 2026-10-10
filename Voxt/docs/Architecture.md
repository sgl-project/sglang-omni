# Source layout and runtime boundaries

Voxt is a macOS 15+ menu bar application, built with the shared `Voxt` Xcode scheme. Build instructions live in [CONTRIBUTING](../CONTRIBUTING.md); dependency pins and model validation requirements live in [MLXAudioDependency](MLXAudioDependency.md).

## Source map

| Directory | Responsibility |
| --- | --- |
| `Voxt/App/` | Application lifecycle, dependency assembly, feature routing, text delivery and window coordination |
| `Voxt/App/Recording/` | Recording start/stop, capture and output routing |
| `Voxt/Core/` | Shared values, persistence, preferences, prompts, security, logging and runtime support |
| `Voxt/Core/LLM/` | Execution plans, provider payloads, remote requests, parsing and answer models |
| `Voxt/Core/History/`, `Voxt/Core/Dictionary/` | Domain stores, repositories and query/value support |
| `Voxt/Core/Notes/` | Note storage, external export state and Obsidian/Reminders sync |
| `Voxt/Core/Models/` | Local LLM/GGUF loading, installation, downloads and model storage support |
| `Voxt/Core/Transcription/` | Capture metrics, VAD planning, ASR hints and transcript support |
| `Voxt/Transcription/` | Speech and MLX transcription, ASR model management |
| `Voxt/Transcription/RemoteASR/` | Remote ASR capture, upload and streaming protocols |
| `Voxt/Meeting/` | Meeting coordination and live sessions; capture, processing and speaker-analysis subdomains |
| `Voxt/Hotkey/` | Event handling, trigger rules, preferences and shortcut recording |
| `Voxt/Settings/` | SwiftUI settings and onboarding; `Shell/` owns navigation and shared presentation |
| `Voxt/Windows/` | AppKit windows, overlays, detail views and their view models |
| `Voxt/Resources/`, `*.lproj`, assets | Bundled prompts, runtimes, localization and visual resources |
| `VoxtTests/` | Tests; `TestSupport/` contains shared fixtures, doubles and model-test gates |
| `tools/`, `Config/`, `packaging/` | Validation/build scripts, signing configuration and packaging inputs |

`Voxt/Voxt/Info.plist` is referenced by the build settings; the nested directory is not an orphan. There is no `Voxt/UI/` directory. Both targets use file-system-synchronized Xcode groups. Preserve membership exceptions and resource references when moving files.

## Main paths

- `AppDelegate` assembles model managers, stores, transcribers and UI coordinators. Its extensions route hotkeys, recording, translation, rewrite, notes and meetings. `LLMRequestLifecycle` owns request validity and outstanding work; `TrackedTaskStore` retains cancelled capture/LLM/audio submission tasks until exit rather than treating cancellation as completion.
- Recording selects a `TranscriberProtocol` implementation. MLX planning, merging, preview text, capture buffers, detached inference and structured segment conversion are in separate files under `Transcription/`. `MLXTranscriber` still owns task cancellation, session revision and model leases; the split does not change that lifetime boundary.
- LLM tasks compile into requests before local or remote execution. `RemoteLLMRuntimeClient` routes compiled requests; sibling extensions own Responses execution, chat completion execution, request construction, provider settings and runtime policy. Existing endpoint/message/parser helpers remain shared with connectivity checks.
- `RecordingSessionLifecycle` controls identity, cancellation and exactly-once output/end admission; capture/UI/timing remain separate. Final output is prepared once as an immutable `SessionFinalizeContext`, then delivered to the input target or answer UI. The delivery callback checks that same session before recording history and dictionary evidence. `TextInputIO` retains input inspection, `TextOutputDelivery` owns integration with paste/key posting, and `TextInjectionTransaction` validates queued work and completes once. Separate output generations permit follow-up keys after normal teardown but reject begin/cancel/dismiss; temporary clipboard restoration checks write ownership. There is no editor acknowledgement or cross-process clipboard CAS. Session-end orchestration remains under `App/Recording/`; there is no generic finalize-stage runner or active preview-replacement pipeline.
- Remote ASR retains recording/generation ownership in `RemoteASRTranscriber`, with file requests and provider streams in sibling files. Provider response actors freeze terminal text and respect cancellation. Dictation and meetings share bounded Doubao framing/gzip and Aliyun endpoint helpers, not their different transcript projection policies.
- Meeting coordination combines microphone/system audio, live transcription and final processing. Remote provider classes are separate; their base session owns ordered buffering, stop deadlines and exactly-once cleanup. `MeetingLiveSessionRegistry` pairs sessions with tokens (including draining sessions); the coordinator has a cleanup barrier before reuse of archive, VAD and model resources. File imports have a separate task owner and per-import transcriber/audio/model-use pipeline. Live finalization snapshots metadata and keeps its stop barrier until checkpoint cleanup; real device/model integration still requires acceptance.
- History and dictionary persistence use repositories backed by `VoxtDatabase` (GRDB/SQLite), with legacy-data migration. Their domain/Codable values are in `TranscriptionHistoryModels` / `DictionaryModels`; stores still own mutable snapshots, caches and writes. Notes and external sync have separate stores/coordinators. Suggestion models/scan policy are separate from mutable scan state and legacy suggestion-file compatibility. New history scans add directly to the dictionary; the retired empty discovery chain no longer runs, while old history suggestion snapshots still decode.

Resumable transfer is shared under `Core/Models` in separate types/delegate/transport files. Custom LLM request policy/tokenizer adaptation and GGUF runtime have their own files; managers retain mutable state. Remote-provider option/default resolution is separated without moving protected credential presence or runtime construction outside their `fileprivate` boundary.

Shared model loading uses `Core/Models/SharedModelLoadCoordinator<Value>`: current waiter entries and outstanding native-loading tasks have different lifetimes. Cancelled generations remain waitable by shutdown and block idle reclamation until they actually return. Manager shutdown callers share one complete cleanup task.

## Permission and input boundaries

LLM requests are text-only. Automatic foreground-window text/structure capture, screenshots, image attachment models and debug capture have been removed. App identity snapshots still support App Branch, dictionary scope and delivery targeting; selected-text operations are retained. Some text-generation checkpoints still require MLXVLM's loading factory.

Global hotkeys and shortcut recording use Accessibility-backed modifying event taps (with local recording fallback), not Input Monitoring/HID listeners. Recording mute changes the current output device's writable mute property and restores only owned changes; it also mutes Voxt sounds. It does not create an audio tap. Meeting system audio still uses a public Core Audio process tap and requests access when capture starts, without private TCC preflight. Screen Recording is no longer requested. See the [implementation and pending macOS validation](ContextEnhancementRemovalImplementation.zh-CN.md).

## Current limitations and maintenance rules

This is one app target, not a set of independently isolated Swift modules. `AppDelegate` still owns substantial cross-feature state, and some shared preference types reside under `Settings`. Directory names do not enforce a dependency boundary.

Keep framework/UI work at presentation and integration boundaries; prefer pure value/planning logic for testable decisions. Reuse existing managers and stores before adding abstractions. Split by ownership or protocol responsibility, not arbitrary line counts. Do not broaden every `private` member merely to enable file splitting.

The project enables default MainActor isolation and approachable concurrency. Preserve explicit `nonisolated`, `Sendable`, task cancellation and ownership assumptions when extracting code. Existing `@unchecked Sendable` types require case-by-case review, not mechanical annotation removal.

Hotkey tap installations own their source, callback context and dedicated thread independently of routing state. Queued work is generation-scoped; business actions run outside the routing lock. Native MLX streams similarly own their event/feed tasks and transferred model use in `MLXNativeLiveRuntime`, but the library's synchronous cancellation is not an awaitable native-worker barrier.

MeetingDetail separates window/controller setup, window layout, playback controller/pane and read-only presentation policy. Its view model keeps async work and mutable published state; the parent view still owns the player and transcript/scroll synchronization.

Settings sidebar/header/footer and notification/feedback dialogs have dedicated files. History note rows retain their local editing state and private footer controls separately from list/filter presentation. Browser permission probes and connectivity protocol helpers are split without changing their existing authorization, payload or timeout policies. Remote provider sheet validation and configuration snapshots have separate extensions; the parent still owns form state, while `RemoteProviderSheetOperations` owns scoped requests, result identity and retirement. Connectivity probes pair socket/session cleanup and close transport before waiting on timed-out receives. Onboarding step implementations and presentation components are separated, but SwiftUI state remains owned by `OnboardingGuideView`; cross-file extensions are not independent state owners.

The concentrated closeout also tracks retiring MLX/remote-preview tasks, serializes meeting VAD preparation with cleanup, and prevents stale dictionary reads from replacing newer snapshots. Five stateful coordination files remain above the review threshold by explicit ownership decision, not by exposing their private state to file slices. See the [closeout and real-environment acceptance checklist (中文)](RefactoringCloseout.zh-CN.md).

Installation presence is asynchronous: callers must not turn an unknown cache result into a missing-model warning. Settings shell/catalog listen to a shared debounced installation-revision publisher instead of polling or observing every progress change. Structured log metadata is redacted by sensitive key as well as value, including nested dictionaries and arrays.

The subsequent [performance/security audit (中文)](PerformanceSecurityReview.zh-CN.md) records new findings and still-open boundaries. Earlier closeout is not a claim that future audits will find no issues.

See the [refactoring assessment (中文)](RefactoringAssessment.zh-CN.md) and [phased implementation record](RefactoringProgress.zh-CN.md) for measured hotspots, completed cleanup, pending work and verification limits.
