# Remote ASR

Remote ASR transcriber implementations and streaming support for provider-backed transcription.

## Responsibilities

- Implements remote ASR client behavior for providers such as Aliyun, Doubao, and StepFun.
- Maintains streaming contexts, provider payload support, realtime debug helpers, and shared remote ASR contracts.
- Keeps provider-specific streaming code separate from local transcription engines.

## Runtime boundaries

- `RemoteASRTranscriber`: recording orchestration, generation IDs, capture and final delivery; completion tasks stay tracked through cancellation and exit.
- `RemoteASRPreviewController`: per-invocation polling/deduplication with generation and recording checks before publication; retired requests remain waitable.
- `RemoteASRCaptureSupport`, `RemoteASRRequestContext`, `RemoteASRPreviewAudio`, `RemoteASRErrorPresentation`: capture conversion, provider hint context, temporary WAV handling and user-facing errors.
- `RemoteASRFileRequests`, `RemoteASRAliyunStreaming`, `RemoteASRDoubaoStreaming`: request and provider implementations on the existing transcriber, not additional session owners.
- `RemoteASRStreamingContexts`: socket/capture contexts and a bounded, cancellable handshake gate.
- `RemoteASRResponseStates`: provider-specific text accumulators; terminal results are frozen and waits propagate cancellation.
- `DoubaoPacketCodec`: framing and bounded gzip shared with meetings. Invalid gzip and unknown compression fail explicitly instead of being displayed as transcript text.
- `RemoteASRDoubaoResponse`: dictation's whole-text projection; meetings keep their timestamped utterance projection separately.
- Payload, text and endpoint helpers have dedicated provider/responsibility files; the former monolithic `RemoteASRSupport.swift` has been removed.

Run `tools/run_local_regression_matrix.sh refactor` on macOS for completion, cancellation, framing and existing provider contracts. Deterministic fake-session tests do not replace real provider/manual verification; see the [phase record](../../../docs/RefactoringProgress.zh-CN.md).
