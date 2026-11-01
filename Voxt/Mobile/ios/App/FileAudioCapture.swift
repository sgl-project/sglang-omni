#if DEBUG
import AVFoundation
import VoxtMobileKit

/// UI tests set `VOXT_TEST_AUDIO_FILE` so a recording returns the file's audio instead of the
/// microphone's; the simulator has no reliable way to feed speech into the mic.
@MainActor
final class FileAudioCapture: AudioCapturing {
    static let environmentKey = "VOXT_TEST_AUDIO_FILE"

    private let url: URL
    private var samples: [Float] = []
    private var startedAt: Date?

    var level: Float {
        guard let startedAt, !samples.isEmpty else { return 0 }
        let offset = Int(Date().timeIntervalSince(startedAt) * 16000) % samples.count
        return AudioLevel.rms(Array(samples[offset..<min(samples.count, offset + 4800)]))
    }

    init(url: URL) {
        self.url = url
    }

    static func fromEnvironment() -> FileAudioCapture? {
        guard let path = ProcessInfo.processInfo.environment[environmentKey], !path.isEmpty else { return nil }
        return FileAudioCapture(url: fixtureURL(path))
    }

    static func fixtureURL(_ path: String) -> URL {
        // Simulator reinstalls can move the data container. A staged basename
        // stays relative to this process rather than the previous container UUID.
        path.contains("/") ? URL(fileURLWithPath: path)
            : FileManager.default.temporaryDirectory.appendingPathComponent(path)
    }

    func startSession() throws {
        samples = readSamples()
        guard !samples.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
    }
    func endSession() { startedAt = nil }
    func beginCapture() { startedAt = Date() }

    func endCapture() -> [Float] {
        startedAt = nil
        return samples
    }

    private func readSamples() -> [Float] {
        guard let file = try? AVAudioFile(forReading: url),
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              (try? file.read(into: buffer)) != nil,
              let downsampler = try? PCMDownsampler(inputFormat: file.processingFormat),
              let samples = try? downsampler.convert(buffer) + downsampler.flush() else {
            return []
        }
        return samples
    }
}
#endif
