import XCTest
@testable import Voxt

@MainActor
final class MeetingFinalizationContextTests: XCTestCase {
    func testAllCheckpointsShareCapturedIdentityAndMetadata() {
        let id = UUID()
        let visible = segment("visible")
        let context = MeetingFinalizationContext(
            sessionID: id, captureMode: .recording, engine: .mlxAudio,
            modelDescription: "captured model", mlxModelRepo: "captured/repo",
            duration: 12, visibleSegments: [visible]
        )
        let url = URL(fileURLWithPath: "/test/archive.wav")
        let date = Date(timeIntervalSince1970: 100)
        for stage in [MeetingFinalizationStage.captured, .finalTranscript, .speakerAnalysis] {
            let checkpoint = context.checkpoint(stage: stage, segments: [segment(stage.rawValue)], archivedAudioURL: url, updatedAt: date)
            XCTAssertEqual(checkpoint.sessionID, id)
            XCTAssertEqual(checkpoint.captureMode, .recording)
            XCTAssertEqual(checkpoint.transcriptionEngineRawValue, TranscriptionEngine.mlxAudio.rawValue)
            XCTAssertEqual(checkpoint.transcriptionModelDescription, "captured model")
            XCTAssertEqual(checkpoint.visibleSnapshotSegments, [visible])
            XCTAssertEqual(checkpoint.audioDurationSeconds, 12)
            XCTAssertEqual(checkpoint.archivedAudioPath, url.path)
            XCTAssertEqual(checkpoint.updatedAt, date)
            XCTAssertEqual(checkpoint.stage, stage)
        }
    }

    func testCheckpointUsesStageSegmentsWithoutReplacingVisibleSnapshot() {
        let visible = segment("visible")
        let context = MeetingFinalizationContext(
            sessionID: UUID(), captureMode: .meeting, engine: .remote,
            modelDescription: "remote", mlxModelRepo: "unused", duration: 1,
            visibleSegments: [visible]
        )
        let corrected = segment("corrected")
        let checkpoint = context.checkpoint(stage: .finalTranscript, segments: [corrected], archivedAudioURL: nil)
        XCTAssertEqual(checkpoint.segments, [corrected])
        XCTAssertEqual(checkpoint.visibleSnapshotSegments, [visible])
        XCTAssertNil(checkpoint.archivedAudioPath)
    }

    func testMetadataIsAValueSnapshotNotMutableEngineSelection() {
        var modelDescription = "original"
        let context = MeetingFinalizationContext(
            sessionID: UUID(), captureMode: .meeting, engine: .remote,
            modelDescription: modelDescription, mlxModelRepo: "unused", duration: 2, visibleSegments: []
        )
        modelDescription = "replacement"
        let checkpoint = context.checkpoint(stage: .captured, segments: [], archivedAudioURL: nil)
        XCTAssertEqual(checkpoint.transcriptionModelDescription, "original")
        XCTAssertNotEqual(checkpoint.transcriptionModelDescription, modelDescription)
    }

    private func segment(_ text: String) -> MeetingTranscriptSegment {
        .init(speaker: .me, startSeconds: 0, endSeconds: 1, text: text)
    }
}
