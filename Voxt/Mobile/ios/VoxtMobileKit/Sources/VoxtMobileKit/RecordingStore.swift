import Foundation

/// A recording kept on the phone until the Mac's text for it arrives.
public struct SavedRecording: Codable, Identifiable, Equatable, Sendable {
    public enum Status: String, Codable, Sendable {
        /// Still being recorded, or sent to the Mac; a relaunch finds these as interrupted.
        case recording, processing
        /// The Mac never returned text: the request failed, the app stopped, or the user cancelled.
        case failed, interrupted, cancelled
    }

    public let id: UUID
    public let startedAt: Date
    public var status: Status
    public var options: CompositionOptions
    /// The keyboard request it answers; its text is inserted only into that request's field.
    public var requestID: String?
    public var errorMessage: String?
    /// Seconds of audio on disk; read from the audio file, not stored.
    public var duration: TimeInterval = 0

    private enum CodingKeys: String, CodingKey { case id, startedAt, status, options, requestID, errorMessage }
}

/// Recordings as 16 kHz mono 16-bit PCM in the app's own container, one audio file and one
/// small JSON file each. Audio is appended while the user speaks, so a crash or a kill
/// loses at most the seconds since the last save.
public final class RecordingStore {
    public static let sampleRate = 16_000
    private let directory: URL
    private let fileManager = FileManager.default

    public init(directory: URL) {
        self.directory = directory
    }

    /// Application Support/Recordings, outside the App Group: the keyboard never needs the audio.
    public static func appDefault() -> RecordingStore {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return RecordingStore(directory: base.appendingPathComponent("Recordings", isDirectory: true))
    }

    @discardableResult
    public func begin(id: UUID = UUID(), options: CompositionOptions, requestID: String?, now: Date) throws -> UUID {
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        guard fileManager.createFile(atPath: audioURL(id).path, contents: nil, attributes: Self.fileAttributes) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try write(SavedRecording(id: id, startedAt: now, status: .recording, options: options, requestID: requestID))
        return id
    }

    public func append(_ samples: [Float], to id: UUID) throws {
        guard !samples.isEmpty else { return }
        let handle = try FileHandle(forWritingTo: audioURL(id))
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: WAVEncoder.pcm16(samples))
    }

    public func mark(_ id: UUID, _ status: SavedRecording.Status, errorMessage: String? = nil) {
        guard var recording = recording(id) else { return }
        recording.status = status
        recording.errorMessage = errorMessage
        try? write(recording)
    }

    public func remove(_ id: UUID) {
        try? fileManager.removeItem(at: audioURL(id))
        try? fileManager.removeItem(at: metadataURL(id))
    }

    public func recording(_ id: UUID) -> SavedRecording? {
        guard let data = try? Data(contentsOf: metadataURL(id)),
              var recording = try? JSONDecoder().decode(SavedRecording.self, from: data) else { return nil }
        let bytes = (try? fileManager.attributesOfItem(atPath: audioURL(id).path)[.size] as? Int) ?? 0
        recording.duration = WAVEncoder.duration(sampleCount: bytes / 2, sampleRate: Self.sampleRate)
        return recording
    }

    public func wav(_ id: UUID) -> Data? {
        guard recording(id) != nil, let pcm = try? Data(contentsOf: audioURL(id)) else { return nil }
        return WAVEncoder.wav(pcm16: pcm, sampleRate: Self.sampleRate)
    }

    /// Every recording on disk, newest first.
    public var all: [SavedRecording] {
        let names = (try? fileManager.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter { $0.hasSuffix(".json") }
            .compactMap { UUID(uuidString: String($0.dropLast(5))).flatMap(recording) }
            .sorted { $0.startedAt > $1.startedAt }
    }

    /// Recordings whose text never arrived, newest first: what History offers to send again.
    public var unprocessed: [SavedRecording] {
        all.filter { [.failed, .interrupted, .cancelled].contains($0.status) }
    }

    /// At launch nothing is recording or uploading yet, so those were cut off by a kill or a crash.
    public func recoverInterrupted() {
        for recording in all where recording.status == .recording || recording.status == .processing {
            if recording.duration == 0 { remove(recording.id) } else { mark(recording.id, .interrupted) }
        }
    }

    private func write(_ recording: SavedRecording) throws {
        try JSONEncoder().encode(recording).write(to: metadataURL(recording.id), options: Self.writeOptions)
    }

    private func audioURL(_ id: UUID) -> URL { directory.appendingPathComponent("\(id.uuidString).pcm") }
    private func metadataURL(_ id: UUID) -> URL { directory.appendingPathComponent("\(id.uuidString).json") }

    // Readable after the first unlock, so a dictation from the keyboard on a locked-then-unlocked
    // phone can still save while the app runs in the background.
    #if os(iOS)
    private static var fileAttributes: [FileAttributeKey: Any] { [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication] }
    private static let writeOptions: Data.WritingOptions = [.atomic, .completeFileProtectionUntilFirstUserAuthentication]
    #else
    private static var fileAttributes: [FileAttributeKey: Any]? { nil }
    private static let writeOptions: Data.WritingOptions = [.atomic]
    #endif
}
