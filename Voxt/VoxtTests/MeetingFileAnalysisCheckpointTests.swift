import Foundation
import XCTest
@testable import Voxt

@MainActor
final class MeetingFileAnalysisCheckpointTests: XCTestCase {
    func testCheckpointStoreRoundTripsAndClearsDescriptorProgress() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-file-checkpoint-\(UUID().uuidString)", isDirectory: true)
        let store = MeetingFileAnalysisCheckpointStore(directoryURL: directory)
        let taskID = UUID()
        let segment = TranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 2,
            text: "checkpoint"
        )
        let checkpoint = MeetingFileASRCheckpoint(
            schemaVersion: MeetingFileASRCheckpoint.currentSchemaVersion,
            taskID: taskID,
            preparedAudioSampleCount: 960_000,
            descriptorCount: 4,
            modelFingerprint: "mlx|qwen|test",
            completedDescriptorCount: 2,
            segments: [segment],
            updatedAt: Date()
        )
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }

        await store.save(checkpoint)
        let loaded = await store.load(taskID: taskID)
        XCTAssertEqual(loaded?.completedDescriptorCount, 2)
        XCTAssertEqual(loaded?.segments, [segment])
        XCTAssertTrue(loaded?.isUsable == true)

        await store.clear(taskID: taskID)
        let cleared = await store.load(taskID: taskID)
        XCTAssertNil(cleared)
    }

    func testInvalidCheckpointIsRejected() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-file-checkpoint-invalid-\(UUID().uuidString)", isDirectory: true)
        let store = MeetingFileAnalysisCheckpointStore(directoryURL: directory)
        let taskID = UUID()
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }

        await store.save(MeetingFileASRCheckpoint(
            schemaVersion: MeetingFileASRCheckpoint.currentSchemaVersion,
            taskID: taskID,
            preparedAudioSampleCount: 10,
            descriptorCount: 2,
            modelFingerprint: "test",
            completedDescriptorCount: 3,
            segments: [],
            updatedAt: Date()
        ))
        let loaded = await store.load(taskID: taskID)
        XCTAssertNil(loaded)
    }
}
