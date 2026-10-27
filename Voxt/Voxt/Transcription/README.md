# Transcription

Native HTTP engine adapters, local model managers, model artifacts and shared transcriber contracts. Historical MLX-prefixed app types manage local ASR; they no longer imply a Swift Audio numerical backend.

## Responsibilities

- Routes local ASR through the existing Native HTTP/WebSocket servers, alongside Speech and Remote ASR.
- Manages ASR discovery, downloads, repository state and bounded artifact validation.
- Preserves family-specific requests, language/hints, structured segments, preview/final projection and longform behavior.

`OmniASRBackend` constructs the Native runtime. Catalog/state values and resumable transfer live under `Core/Models`; `MLXModelDownloadSupport` retains metadata/validation and tokenizer-asset policy. Managers own install/download state, storage revisions and load lifetimes. `MLXModelDownloadAttempt` receives one attempt's endpoint, root, token and progress sink; `MLXModelArtifactPreparation` shares asset transfer and repair operations without accessing manager state. Lifecycle tests use identity-only fixtures that reject inference.

## Native HTTP ownership

`OmniASRRuntime` owns the child process, active uses and a cooperating canonical model-directory reader. It acquires the reader before spawning an ASR, Silero or Sortformer server and retains it until actual child exit. A running child retains its runtime after callers drop it. Shutdown escalates graceful exit to termination signals; deadlines cannot release readers while a process remains alive. Uninterruptible children can delay retirement.

`OmniASRRuntimeSupport` contains endpoint/configuration values, diagnostics and line buffering; `OmniASRRuntimeFinalRequests` retains family-specific requests. Endpoint uses close their exact scope. Network cancellation and task settlement are not per-stream GPU drain proof. Whole-model release is observable through process exit, without introducing new APIs into the shared Native package.

Directory leases protect cooperating operations, not external blob writes, uncoordinated namespace replacement or an App crash releasing locks before an orphaned child exits. Cache/blob/repair and parent-crash/orphan recovery are not all coordinated.

## App task ownership

`MLXTranscriber` owns recording tasks, revision checks, model leases and capture buffers. `MLXAudioCaptureGraph` borrows its engine for device/format configuration and blocking-start coordination, without owning session state. Pure policy lives in `MLXTranscriptionPlanning`, `MLXInferenceConfiguration`, `MLXTranscriptMerging` and `MLXLiveTextPreview`; `MLXDetachedInference` shares final request orchestration and Native VAD options between correction and buffered transcription. The retired Swift SenseVoice probability-range reconstruction is not a second segmentation path.

`MLXCorrectionPassCoordinator` owns the in-flight correction slot until its task exits, including cancelled work. Final passes wait for retiring intermediate work; the transcriber rechecks revision/model use after suspension.

`MLXNativeLiveRuntime` owns the installed stream, feed/event tasks and transferred model use. Setup, finalization/preload/watchdogs and early prewarm retain cancelled tasks until settlement so shutdown can observe them. Audio-engine startup cancellation requests stop but still waits for the underlying start call. Neither that wait nor synchronous stream cancellation is a hard Native GPU-exit deadline.

## Dependencies and validation

Audio STT/VAD/Core loaders, numerical compatibility adapters, products and package pin are retired. XPC helper/client/contracts, validation configurations and the C ABI route are removed. Default production remains the unmodified Native HTTP implementation, not a new 30-second or bounded XPC route.

Keep model catalogs, saved choices and installed directories, MLX Swift LLM/VLM/MLX, Tokenizers/HuggingFace, llama.swift and Remote. Captured AudioCore DSP and independent model/sample-bound numerical references replace linked-library oracles; missing references skip parity explicitly.

Preserve actor isolation, nonisolated inference, cancellation propagation and existing buffer locks. See the [Native HTTP plan](../../plan/NativeHTTPMigrationPlan.zh-CN.md) and [earlier phase record](../../docs/RefactoringProgress.zh-CN.md) before changing ownership.

Recording replaces its Silero frame decider on each activation. Dropping the streaming detector closes its native VAD sockets and returns its shared-runtime lease; otherwise each discarded session consumes a server worker until later activations cannot open a VAD stream. A VAD exchange has a two-second response deadline and propagates caller cancellation to its socket. Recording falls back to energy VAD for the remainder of a failed session and tries Silero again on the next activation, so an unresponsive VAD cannot indefinitely starve live ASR or block stopping.
