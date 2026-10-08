import Foundation

/// Small, descriptor-level recovery state for imported-file ASR.
/// It intentionally does not persist model tensors, token streams or speaker state.
nonisolated struct MeetingFileASRCheckpoint: Codable, Sendable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let taskID: UUID
    let preparedAudioSampleCount: Int
    let descriptorCount: Int
    let modelFingerprint: String
    let completedDescriptorCount: Int
    let segments: [MeetingTranscriptSegment]
    let updatedAt: Date

    var isUsable: Bool {
        schemaVersion == Self.currentSchemaVersion
            && completedDescriptorCount >= 0
            && completedDescriptorCount <= descriptorCount
    }
}

actor MeetingFileAnalysisCheckpointStore {
    static let shared = MeetingFileAnalysisCheckpointStore()

    private let fileManager: FileManager
    private let directoryURL: URL

    init(fileManager: FileManager = .default, directoryURL: URL? = nil) {
        self.fileManager = fileManager
        if let directoryURL {
            self.directoryURL = directoryURL
        } else {
            let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? fileManager.temporaryDirectory
            self.directoryURL = appSupport
                .appendingPathComponent("Voxt", isDirectory: true)
                .appendingPathComponent("meeting-file-checkpoints", isDirectory: true)
        }
    }

    func load(taskID: UUID) -> MeetingFileASRCheckpoint? {
        let url = url(for: taskID)
        guard let data = try? Data(contentsOf: url) else { return nil }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let checkpoint = try decoder.decode(MeetingFileASRCheckpoint.self, from: data)
            guard checkpoint.isUsable else {
                try? fileManager.removeItem(at: url)
                return nil
            }
            return checkpoint
        } catch {
            VoxtLog.meetingWarning("File ASR checkpoint decode failed. taskID=\(taskID)")
            try? fileManager.removeItem(at: url)
            return nil
        }
    }

    func save(_ checkpoint: MeetingFileASRCheckpoint) {
        do {
            try fileManager.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.sortedKeys]
            let url = url(for: checkpoint.taskID)
            try encoder.encode(checkpoint).write(to: url, options: .atomic)
            try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch {
            VoxtLog.meetingWarning("File ASR checkpoint write failed. taskID=\(checkpoint.taskID)")
        }
    }

    func clear(taskID: UUID) {
        try? fileManager.removeItem(at: url(for: taskID))
    }

    private func url(for taskID: UUID) -> URL {
        directoryURL.appendingPathComponent("\(taskID.uuidString).json")
    }
}
