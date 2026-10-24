# Local Regression Matrix

Use this document to choose focused local checks before a full release gate. CI remains the source of truth for broad regression coverage.

## Groups

| Group | Scope | Command |
| --- | --- | --- |
| core | Capture pipeline, session flow, prompt building, VAD planning, Feature Settings, MLX planning, model debug | `tools/run_local_regression_matrix.sh core` |
| refactor | core + split LLM/configuration/hotkey/model/meeting-detail suites, onboarding and storage contracts | `tools/run_local_regression_matrix.sh refactor` |
| mlx | MLX public fixture and replay tests | `VOXT_RUN_MODEL_TESTS=1 tools/run_local_regression_matrix.sh mlx` |
| gguf | Installed GGUF inference and native termination cleanup | `tools/run_local_regression_matrix.sh gguf` |
| vad | Local VAD mode, runtime policy, storage, debug snapshot | `tools/run_local_regression_matrix.sh vad` |
| installed | Installed-model long-form matrix | `VOXT_RUN_MODEL_TESTS=1 tools/run_local_regression_matrix.sh installed` |
| all | refactor + mlx (core already includes vad) | `VOXT_RUN_MODEL_TESTS=1 tools/run_local_regression_matrix.sh all` |
| full | all + gguf + installed | `VOXT_RUN_MODEL_TESTS=1 tools/run_local_regression_matrix.sh full` |

The script resolves the repository from its own location, disables signing, and uses the committed lockfile strictly. `VOXT_SPM_CACHE_PATH` and `VOXT_SPM_CLONE_PATH` override the local cache paths. Model groups need installed checkpoints; review skips separately from passes.

The `refactor` group also includes the stage-4 ASR framing/response/completion and meeting-session lifecycle tests, alongside existing ASR/meeting support cases. It also includes stage-5A tracked-task, request validity, meeting-token, correction-pass and capture-epoch contracts. Its fake sessions, task barriers and controlled deadlines do not exercise real provider accounts or devices.

Stage 6A adds existing dictionary learning/matcher/store, history serialization/store/correction, and meeting-detail formatting/virtual-list/cache coverage to `refactor`. Playback controls, scrolling and accessibility still need UI acceptance on macOS.

Stage 6B includes current local-LLM request policy, source selection, resumable URLProtocol, installation/cache and catalog contracts. The three non-model GGUF tests are selected by method; installed GGUF inference is still isolated in `gguf`. Tooling validates both suite and method names to detect stale selectors.

Stage 6C adds queued-injection and pasteboard-ownership tests, lifecycle output-generation cases, and the existing connectivity tester suite. Pasteboard tests use unique named boards, never the general clipboard. They do not post real keys or validate editor/focus behavior; manual acceptance must cover cancel/restart during paste, overlay dismissal, follow-up keys, window switching and concurrent user copies.

Stage 6D adds `DictionarySuggestionStoreTests` and the existing `RemoteProviderConfigurationPolicyTests` to `refactor`. The store uses isolated defaults and temporary legacy files to check compatibility, merge/write-back, direct dictionary insertion and scan checkpoints; provider cases target current generation/schema validation rather than a retired OpenAI wrapper. These do not replace asynchronous reload/network or UI acceptance.

The concentrated closeout adds scoped sheet tasks, WebSocket close/deadline, stale reload, ASR preview, corrupt hotkey preference, bounded progress, inference planning and meeting presentation suites. `RemoteLLMRuntimeClientFailureTests` uses isolated URLProtocol sessions for actual streaming execution/fallback/cancellation without provider accounts. CI now builds Release after tests and archives xcresult/discovery plus command timing/resource logs. See [closeout](RefactoringCloseout.zh-CN.md) for remaining manual/native/model and performance acceptance; a build RSS observation is not application/Metal memory validation.

The startup-model fix adds `ModelConfigurationIssueResolverTests` and `VoxtLogRedactorTests` to `refactor`. Installation checks still use background cached scans; tests verify that checking is not reported as missing and that the actual installation-revision publisher refreshes issues without user interaction. The source inventory is archived by CI as review evidence, not an assertion that all listed sites are safe. Follow the [performance/security review](PerformanceSecurityReview.zh-CN.md) for real-device profiling and outstanding risks.

The old `whisper` / `diagnostic` groups were removed because their dedicated Whisper test classes no longer exist. This does not remove MLX Whisper model support or migration coverage. Unknown groups fail rather than selecting nonexistent suites. The `all` / `full` groups avoid rerunning VAD suites already selected by core.

See [test-suite organization](../VoxtTests/README.md) and the [phased refactoring record](RefactoringProgress.zh-CN.md).

## VAD / ASR Gate Safety

Current runtime VAD choices are global and local-only:

- `Automatic`: Voxt selects the local policy for the workflow.
- `Silero`: local `mlx-community/silero-vad`.
- `Energy`: lightweight local level detector.
- `Off`: disables local VAD gating while preserving capture and final ASR.

The VAD group must cover:

- `LocalVADMode` parsing, persistence, and default behavior.
- Runtime backend resolution for transcription, translation, rewrite, and meeting.
- Meeting frame VAD behavior for Silero, Energy, and Off.
- Local gate disabled for non-local ASR.
- Debug snapshot fields: VAD Mode, Frame VAD, Local Gate.
- Segmenter hysteresis, trailing silence, minimum speech duration, and telemetry reasons.
- Non-finite audio sample, level, rate, and timestamp normalization before model input or segment state updates.

## Phase 6 Smoke Gate

Run before manual acceptance or packaging validation:

```bash
tools/run_vad_phase6_smoke.sh --duration 300 --interval 30
```

The phase-6 smoke gate builds Release by default and then runs:

- Launch smoke with VAD/Silero fatal/error denylist.
- Privacy denylist for raw audio, full transcript, transcript text, and prompt text.
- Damaged local MLX Silero cache smoke.
- Clean preferences smoke.
- `localVADMode` smoke for `automatic`, `silero`, `energy`, and `off`.
- Acceptance report generation.

Validate a filled report with:

```bash
tools/validate_vad_acceptance_report.sh --report <report.md>
```

Use `--allow-unsigned` only for local unsigned package preflight.

## Manual Gate

Manual acceptance lives in `docs/VADManualAcceptance.zh-CN.md`. Do not promote a VAD parameter change to default until meeting, transcription, translation, rewrite, long-recording, stop/cancel, device-change, sleep/wake, and packaging scenarios pass with evidence.
