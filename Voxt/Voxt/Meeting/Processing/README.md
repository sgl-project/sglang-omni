# Meeting Processing

Post-capture meeting processing for ASR, translation, transcript assembly, and final summaries.

## Responsibilities

- Transcribes meeting segments and combines them into coherent final transcript output.
- Smooths speaker turns, assembles speaker-aware transcript text, and prepares final meeting records.
- Applies translation and summary support after audio capture is complete.

## File import and finalization

`MeetingImportedFileAnalyzer` registers import work before awaiting the previous live-session cleanup. Cancellation captures that operation rather than looking up a potentially newer task later. It remains busy through cleanup and cancels abandoned operations.

`MeetingFileTask` and `MeetingFileAnalysisProgress` own task state and progress mapping. `MeetingFileTaskStore` owns persistence and staging-directory accounting. `MeetingFileTaskQueue` admits bounded pending imports and serializes preprocessing with file analysis. It streams directly from the authorized source into a versioned canonical WAV cache instead of first copying an entire video. The cache is published only after validation and atomic rename; retries reuse it and v1 raw staged tasks are upgraded lazily. The original 4 GiB source limit remains until large-container stress tests justify raising it.

`MeetingFilePreparationLimits` bounds decoded duration/output, conversion buffers and disk usage. `MeetingFilePreparationResources` waits at checkpoints under memory/thermal pressure. AVFoundation's internal allocations are not covered by the application buffer ceiling.

File inference memory policy is centralized in `MeetingLocalInferenceCoordinator`: `MeetingFileInferenceCache` scopes the unused MLX cache limit to at most 128 MiB for each admitted file work unit, restores the previous setting on all exits and reclaims oversized unused buffers at safe boundaries. It never changes the active-memory limit or unloads live weights/speaker state. Normal and warning levels permit bounded file work; critical pressure still blocks. An unavailable pressure probe conservatively falls back to notifications without altering unrelated callers' warning policy. There are no estimated device budgets, timeout-based force-resume or repeated model reloads. Limits concern unused cache, not total process/GPU peak memory; actual CPU/memory/quality validation remains pending.

ASR chunk collections are lazy: planning holds only ranges, and iteration copies just the currently consumed PCM chunk. File speaker feeds reuse trusted standardized 16 kHz samples without another full-window sanitizing copy.

`MeetingFileAnalysisCheckpointStore` is the intentionally small P2 recovery slice: after each imported-audio descriptor, it atomically stores the committed ASR segments, completed descriptor count, prepared-audio sample count and model fingerprint. A retry/restart resumes at the next descriptor only when input and model identity match; otherwise the checkpoint is discarded. It does not persist model tensors, token streams or speaker-analysis state.

File speaker analysis now uses an operation-owned Sortformer engine with bounded, frame-aligned feeds (at most five seconds and no more than the configured AOSC retirement budget). It preserves streaming identity, validates FIFO/cache lengths after each feed, and obtains a low-priority file permit per feed rather than holding one for the whole recording. Errors propagate so the completed ASR checkpoint is retained instead of silently saving a transcript-only success. The file page opens the existing meeting detail window in read-only file-draft mode using the completed checkpoint's timestamped segments; it does not create a history record or start summary generation. Successful completion upgrades an open draft to the final history result in the same window. See [long-file repair and pending validation](../../../docs/SortformerLongFileRepair.zh-CN.md).

`MeetingImportedFilePipeline` owns one import's transcriber, temporary audio and model use; it does not mutate live coordinator engine/transcriber fields. Prepared queue inputs are borrowed read-only for analysis. Only after successful analysis, and when history audio storage was enabled at analysis start, is an independent archive copy created for history's move-based import. Failed/cancelled analysis no longer copies the entire WAV unnecessarily; disabled audio storage skips the copy. Cleanup removes only operation-owned temporary files, never the queue cache.

File-task logs stay at lifecycle boundaries: preparation, analysis stages, resource waits, failures and completion. Per-window, per-feed and allocator snapshots are opt-in through `VOXT_FILE_TASK_TRACE=1`; task identity lives independently in `MeetingFileTaskContext`.

Implementation details, remaining limitations and pending macOS acceptance: [File preprocessing resource safety](../../../docs/FilePreprocessingResourceSafety.zh-CN.md).

`MeetingFinalizationContext` snapshots stop-time identity, engine/model metadata, duration and visible segments for all recovery checkpoints. The live coordinator remains occupied until final checkpoint cleanup finishes, and repeated stop calls return the same task.
