import Foundation

/// Immutable identity/metadata shared by every recovery checkpoint and final
/// result. Preferences may change while inference or disk I/O is suspended.
struct MeetingFinalizationContext {
    let sessionID: UUID
    let captureMode: MeetingCaptureMode
    let engine: TranscriptionEngine
    let modelDescription: String
    let mlxModelRepo: String
    let duration: TimeInterval
    let visibleSegments: [MeetingTranscriptSegment]

    func checkpoint(
        stage: MeetingFinalizationStage,
        segments: [MeetingTranscriptSegment],
        archivedAudioURL: URL?,
        updatedAt: Date = Date()
    ) -> MeetingFinalizationCheckpoint {
        MeetingFinalizationCheckpoint(
            sessionID: sessionID,
            updatedAt: updatedAt,
            stage: stage,
            captureMode: captureMode,
            transcriptionEngineRawValue: engine.rawValue,
            transcriptionModelDescription: modelDescription,
            segments: segments,
            visibleSnapshotSegments: visibleSegments,
            audioDurationSeconds: duration,
            archivedAudioPath: archivedAudioURL?.path
        )
    }
}
