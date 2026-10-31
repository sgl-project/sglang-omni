import Foundation
import Testing
@testable import VoxtMobileKit

/// Recordings live on disk from the first seconds until the Mac's text arrives, so a crash,
/// a kill or a dead network never costs the user what they said.
@Suite struct RecordingStoreTests {
    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("recordings-\(UUID().uuidString)", isDirectory: true)
    }

    private func seconds(_ value: Double) -> [Float] { Array(repeating: 0.1, count: Int(value * 16_000)) }

    @Test func audioWrittenWhileRecordingReadsBackAsWAVAfterARelaunch() throws {
        let dir = directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = RecordingStore(directory: dir)
        let id = try store.begin(options: .init(mode: .edit, selectedText: "draft"), requestID: "kb-1",
                                 now: Date(timeIntervalSince1970: 1_000))
        try store.append(seconds(0.5), to: id)
        try store.append(seconds(0.5), to: id)

        let relaunched = RecordingStore(directory: dir)
        let saved = try #require(relaunched.recording(id))
        #expect(saved.status == .recording)
        #expect(saved.duration == 1)
        #expect(saved.requestID == "kb-1")
        #expect(saved.options.mode == .edit)
        #expect(saved.options.selectedText == "draft")
        #expect(saved.startedAt == Date(timeIntervalSince1970: 1_000))
        let wav = try #require(relaunched.wav(id))
        #expect(wav == WAVEncoder.encode(samples: seconds(1), sampleRate: 16_000))
    }

    @Test func recordingsCutOffByAKillComeBackAsInterrupted() throws {
        let dir = directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = RecordingStore(directory: dir)
        let recording = try store.begin(options: .init(), requestID: nil, now: Date())
        try store.append(seconds(2), to: recording)
        let uploading = try store.begin(options: .init(), requestID: nil, now: Date())
        try store.append(seconds(3), to: uploading)
        store.mark(uploading, .processing)
        let empty = try store.begin(options: .init(), requestID: nil, now: Date())

        let relaunched = RecordingStore(directory: dir)
        relaunched.recoverInterrupted()
        #expect(relaunched.recording(recording)?.status == .interrupted)
        #expect(relaunched.recording(uploading)?.status == .interrupted)
        #expect(relaunched.recording(empty) == nil)
        #expect(Set(relaunched.unprocessed.map(\.id)) == [recording, uploading])
    }

    @Test func unprocessedListsWhatNeverBecameTextNewestFirst() throws {
        let dir = directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = RecordingStore(directory: dir)
        func make(_ status: SavedRecording.Status, at time: TimeInterval) throws -> UUID {
            let id = try store.begin(options: .init(), requestID: nil, now: Date(timeIntervalSince1970: time))
            try store.append(seconds(1), to: id)
            store.mark(id, status, errorMessage: status == .failed ? "Couldn't reach your Mac" : nil)
            return id
        }
        let failed = try make(.failed, at: 100)
        let cancelled = try make(.cancelled, at: 300)
        let interrupted = try make(.interrupted, at: 200)
        _ = try make(.recording, at: 400)
        _ = try make(.processing, at: 500)

        #expect(store.unprocessed.map(\.id) == [cancelled, interrupted, failed])
        #expect(store.recording(failed)?.errorMessage == "Couldn't reach your Mac")
    }

    @Test func removingARecordingDeletesItsAudio() throws {
        let dir = directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = RecordingStore(directory: dir)
        let id = try store.begin(options: .init(), requestID: nil, now: Date())
        try store.append(seconds(1), to: id)
        store.remove(id)
        #expect(store.recording(id) == nil)
        #expect(store.wav(id) == nil)
        #expect(try FileManager.default.contentsOfDirectory(atPath: dir.path).isEmpty)
    }
}
