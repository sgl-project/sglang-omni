// Compiled in the same source file as AudioCapture.swift to exercise its private
// recorder state without granting microphone access or adding production test hooks.
@MainActor
private extension MicrophoneRecorder {
    func prepareFixture() {
        cancel()
        rate = 16000
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate,
                                   channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8000)!
        buffer.frameLength = buffer.frameCapacity
        for index in 0..<Int(buffer.frameLength) {
            buffer.floatChannelData![0][index] = sin(Float(index) * 0.1) * 0.3
        }
        captured = CaptureBuffer(rate: rate)
        captured?.append(buffer)
    }

    func failFixture() { captured?.markFailure() }
}

@MainActor
private final class FixtureRecorder: AudioRecording {
    let real = MicrophoneRecorder()
    func start() async throws { real.prepareFixture() }
    func stop() throws -> Data { try real.stop() }
    func cancel() { real.cancel() }
    var recordingError: String? { real.recordingError }
}

@MainActor
private final class Service: SpeechServing {
    var requests = 0
    func transcribe(wav: Data) async throws -> String { requests += 1; return "valid recording" }
    func polish(text: String) async throws -> String { text }
}

@main
private enum CaptureStopTests {
    @MainActor
    static func main() async throws {
        let healthy = MicrophoneRecorder()
        healthy.prepareFixture()
        let wav = try healthy.stop()
        precondition(wav.starts(with: Data("RIFF".utf8)), "Healthy recording must still encode")

        let recorder = FixtureRecorder(), service = Service()
        let session = DictationSession(recorder: recorder, service: service)
        session.toggleRecording()
        while session.phase != .recording { try await Task.sleep(nanoseconds: 1_000_000) }
        try await Task.sleep(nanoseconds: 70_000_000)
        // No suspension between failure and stop: the session timer cannot catch it.
        recorder.real.failFixture()
        session.toggleRecording()
        try await Task.sleep(nanoseconds: 20_000_000)
        precondition(session.phase == .failed && session.notice.contains("设备发生变化"),
                     "Stopping must surface a capture error that arrived after the last timer tick")
        precondition(service.requests == 0, "Invalid buffered audio must never reach ASR")
        precondition(recorder.recordingError == nil, "Failure must still clean up the recorder")
        recorder.real.prepareFixture()
        let recovered = try recorder.real.stop()
        precondition(!recovered.isEmpty, "A fresh recording must not inherit the old failure")
        session.cancel()
        print("PASS: capture failure at stop prevents ASR submission; cleanup and subsequent recording work")
    }
}
