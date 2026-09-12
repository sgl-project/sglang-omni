import AVFoundation
import Foundation

@MainActor
final class LimitRecorder: AudioRecording {
    var stops = 0
    func start() async throws { }
    func stop() throws -> Data { stops += 1; return Data([1]) }
    func cancel() { }
}

@MainActor
final class LimitService: SpeechServing {
    var requests = 0
    func transcribe(wav: Data) async throws -> String { requests += 1; return "测试" }
    func polish(text: String) async throws -> String { text }
}

@main
struct RecordingLimitTest {
    static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        precondition(condition(), message)
    }

    @MainActor
    static func main() async throws {
        let rate = 16000.0
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate,
                                  channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(rate * 20))!
        buffer.frameLength = buffer.frameCapacity
        let capture = CaptureBuffer(rate: rate)
        // Distinct sections prove that audio after the old 30-second boundary survives.
        for amplitude: Float in [0.1, 0.2, 0.3, 0.4] {
            buffer.floatChannelData![0].update(repeating: amplitude, count: Int(buffer.frameLength))
            capture.append(buffer)
        }
        let samples = capture.finish()
        expect(samples.count == Int(rate * 60), "Capture must retain exactly one minute and discard later samples")
        expect(samples[Int(rate * 35)] == 0.2 && samples[Int(rate * 55)] == 0.3,
               "The second half of a minute must not be truncated at the old limit")
        capture.append(buffer)
        expect(capture.finish().isEmpty, "Finished capture must not accept later audio")

        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("omni-duration-test-\(UUID())")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for seconds in [45.0, 60.0] {
            let file = folder.appendingPathComponent("allowed-\(Int(seconds)).wav")
            let wav = try AudioEncoder.wav(samples: Array(samples.prefix(Int(rate * seconds))), sampleRate: rate)
            try wav.write(to: file)
            let loaded = try AudioEncoder.load(file)
            let outputFrames = (loaded.count - 44) / 2
            expect(abs(outputFrames - Int(rate * seconds)) <= 1, "Import must retain the entire allowed recording")
        }
        let oversized = folder.appendingPathComponent("too-long.wav")
        try AudioEncoder.wav(samples: samples + [Float](repeating: 0.4, count: Int(rate)), sampleRate: rate)
            .write(to: oversized)
        do {
            _ = try AudioEncoder.load(oversized)
            preconditionFailure("A recording longer than one minute must be rejected")
        } catch let error as DictationError {
            expect(error.message.contains("60 秒"), "Oversized import must explain the current limit")
        }

        let recorder = LimitRecorder()
        let service = LimitService()
        let session = DictationSession(recorder: recorder, service: service)
        expect(session.recordingLimit == 60, "The default session must use the shared one-minute limit")
        // Exercise the same auto-stop path with a short injected limit, without opening a microphone.
        let short = DictationSession(recorder: recorder, service: service, recordingLimit: 0.02)
        short.toggleRecording()
        let deadline = ProcessInfo.processInfo.systemUptime + 3
        while short.phase != .ready, ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        expect(short.phase == .ready && recorder.stops == 1 && service.requests == 1,
               "Reaching the configured limit must stop and transcribe once")
        print("PASS: one-minute capture/import boundaries, second-half retention, over-limit rejection and automatic stop")
    }
}
