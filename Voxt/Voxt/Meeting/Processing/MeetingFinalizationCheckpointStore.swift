// MeetingFinalizationCheckpointStore.swift
// Persists the last complete post-processing stage so an interrupted meeting can be recovered.

import Foundation

nonisolated enum MeetingFinalizationStage: String, Codable, Sendable {
    case captured
    case finalTranscript
    case speakerAnalysis
}

nonisolated struct MeetingFinalizationCheckpoint: Codable, Sendable {
    let sessionID: UUID
    let updatedAt: Date
    let stage: MeetingFinalizationStage
    let captureMode: MeetingCaptureMode
    let transcriptionEngineRawValue: String
    let transcriptionModelDescription: String
    let segments: [MeetingTranscriptSegment]
    let visibleSnapshotSegments: [MeetingTranscriptSegment]
    let audioDurationSeconds: TimeInterval
    let archivedAudioPath: String?
}

actor MeetingFinalizationCheckpointStore {
    static let shared = MeetingFinalizationCheckpointStore()

    private let fileManager: FileManager
    private let checkpointURL: URL

    init(fileManager: FileManager = .default, directoryURL: URL? = nil) {
        self.fileManager = fileManager
        let directory = directoryURL ?? Self.defaultDirectory(fileManager: fileManager)
        checkpointURL = directory.appendingPathComponent("meeting-finalization.json")
    }

    func save(_ checkpoint: MeetingFinalizationCheckpoint) {
        do {
            try fileManager.createDirectory(
                at: checkpointURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(checkpoint).write(to: checkpointURL, options: .atomic)
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: checkpointURL.path)
        } catch {
            VoxtLog.meetingWarning("Meeting finalization checkpoint write failed: \(error.localizedDescription)")
        }
    }

    func load() -> MeetingFinalizationCheckpoint? {
        guard let data = try? Data(contentsOf: checkpointURL) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(MeetingFinalizationCheckpoint.self, from: data)
        } catch {
            VoxtLog.meetingWarning("Meeting finalization checkpoint decode failed: \(error.localizedDescription)")
            try? fileManager.removeItem(at: checkpointURL)
            return nil
        }
    }

    func clear(sessionID: UUID? = nil) {
        if let sessionID, let checkpoint = load(), checkpoint.sessionID != sessionID {
            return
        }
        try? fileManager.removeItem(at: checkpointURL)
    }

    private nonisolated static func defaultDirectory(fileManager: FileManager) -> URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support", isDirectory: true)
        return appSupport
            .appendingPathComponent("Voxt", isDirectory: true)
            .appendingPathComponent("Recovery", isDirectory: true)
    }
}
