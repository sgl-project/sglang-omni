# Speaker Analysis

Speaker diarization models, labels, settings and display helpers for meeting transcripts.

## Responsibilities

- Configures the Native HTTP diarization engine and model manager.
- Maps speaker turns to stable labels/readable display names.
- Stores speaker-analysis settings and shared models for meeting processing.

## Transport ownership

`SortformerMeetingSpeakerDiarizationEngine` delegates to the existing Native HTTP server. Unavailable Native throws explicitly; the Swift Audio fallback is retired. `MeetingSortformerConfiguration` reads a bounded Foundation projection for the ordinary scheduler, not numerical model compatibility or shared-container authorization.

The HTTP engine owns an exact runtime/model-directory lease and serializes sessions across suspension. File completion/failure retires that exact process before releasing admission; shared endpoint reference counts or socket cancellation are not process-exit proof. Cached/live ownership remains separate from file-scoped cleanup. Directory selection preserves readable-root order and rejects busy/unsafe acquisition rather than silently selecting another cache.

The original approximately five-second feed scheduler, FIFO/cache geometry and PCM-relative offsets remain. The uncalled legacy realtime annotation actor was removed after caller/selector/resource audit. No replacement 24-second whole-window transport, arbitrary segmentation or Native-cap change was introduced.

XPC factory/consumer/owner code and its bounded timing contract are removed. The shared Native package remains the official implementation; configuration and transport adaptation belong to Voxt.

## Cleanup and acceptance

File failures, cancellation and slow-feed checks await owned session/runtime cleanup before releasing inference admission. A noncooperative asset/progress callback cannot be forced to return. Stream/socket close is not an added Native GPU barrier; the model directory reader remains until actual process exit.

Weights/catalog choices and existing storage paths remain, without Audio package linkage. Numerical acceptance requires an independently captured legacy reference matching model, samples and configuration. Missing real Sortformer weights/reference is an explicit skip. Cache/blob/orphan recovery remains separate work.
