# Transcription

Transcription engine adapters, local model managers, model artifacts, and shared transcriber contracts.

## Responsibilities

- Implements MLX, Speech, remote ASR, and legacy Whisper migration integration points.
- Manages local ASR model discovery, downloads, repository state, and artifact validation.
- Provides common transcriber protocols, support types, and post-processing for transcript text.

ASR model construction is in `MLXSTTModelLoader`. Catalog/state values and reusable resumable transfer live under `Core/Models`; `MLXModelDownloadSupport` retains ASR metadata/validation and tokenizer-asset policy. Managers continue to own install/download state, storage revision and load lifetimes.

## MLX boundaries

`MLXTranscriber` owns recording tasks, revision checks, model leases and capture-buffer instances. Pure policy is in `MLXTranscriptionPlanning`, `MLXInferenceConfiguration`, `MLXTranscriptMerging`, and `MLXLiveTextPreview`. Shared values, buffers, detached inference and structured segment conversion live in their correspondingly named files.

`MLXCorrectionPassCoordinator` owns the one in-flight correction pass. Cancelling a pass does not free its slot until the task has exited; final passes wait for retiring intermediate work. The transcriber still owns session revision and model leases, and verifies them after asynchronous inference/archive work.

`MLXNativeLiveRuntime` owns an installed native stream, event/feed tasks and its transferred model use. Replacement/retirement keeps old tasks separate from the new stream, and releases the use once Voxt's tasks exit; abandoned owners also cancel and retire their stream. Setup, session loops/finalization/preload/watchdogs and early prewarm use tracked-task stores so cancelled work remains visible during shutdown and blocks idle reclamation until exit. Audio-engine startup cancellation requests stop, but still waits for the native start call; this is not a hard native-exit deadline.

This is not a native decode/Metal completion barrier: the dependency's synchronous `cancel()` has no awaitable internal-worker completion API. Verify actual model memory/cancellation behavior separately.

Keep `nonisolated` inference, cancellation propagation and the existing buffer locks intact when moving code. The source split does not introduce another model/session owner. See the [phase record](../../docs/RefactoringProgress.zh-CN.md) before changing runtime ownership.
