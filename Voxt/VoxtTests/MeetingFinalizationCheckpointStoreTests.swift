// MeetingFinalizationCheckpointStoreTests.swift
// Verifies atomic meeting finalization recovery metadata.

import XCTest
@testable import Voxt

final class MeetingFinalizationCheckpointStoreTests: XCTestCase {
    func testCheckpointRoundTripAndConditionalClear() async throws {
        let directory = try TemporaryDirectory()
        let store = MeetingFinalizationCheckpointStore(directoryURL: directory.url)
        let sessionID = UUID()
        let segment = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 1,
            endSeconds: 2,
            text: "recoverable text"
        )
        let checkpoint = MeetingFinalizationCheckpoint(
            sessionID: sessionID,
            updatedAt: Date(timeIntervalSince1970: 1_700_000_000),
            stage: .finalTranscript,
            captureMode: .meeting,
            transcriptionEngineRawValue: TranscriptionEngine.mlxAudio.rawValue,
            transcriptionModelDescription: "Visible local model",
            segments: [segment],
            visibleSnapshotSegments: [segment],
            audioDurationSeconds: 42,
            archivedAudioPath: "/tmp/meeting.wav"
        )

        await store.save(checkpoint)
        let loaded = await store.load()
        XCTAssertEqual(loaded?.sessionID, sessionID)
        XCTAssertEqual(loaded?.stage, .finalTranscript)
        XCTAssertEqual(loaded?.segments, [segment])

        await store.clear(sessionID: UUID())
        let retained = await store.load()
        XCTAssertNotNil(retained)

        await store.clear(sessionID: sessionID)
        let cleared = await store.load()
        XCTAssertNil(cleared)
    }
}
