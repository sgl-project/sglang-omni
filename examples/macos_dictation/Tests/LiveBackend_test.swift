import Foundation

@MainActor
private final class UnusedRecorder: AudioRecording {
    func start() async throws { throw DictationError("文件验证不使用麦克风") }
    func stop() throws -> Data { throw DictationError("文件验证不使用麦克风") }
    func cancel() {}
}

@main
struct LiveBackendTest {
    @MainActor
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            throw DictationError("用法：LiveBackend_test /absolute/path/to/audio.m4a")
        }
        let wav = try AudioEncoder.load(URL(fileURLWithPath: CommandLine.arguments[1]))
        let service = LocalSpeechService()
        let status = await service.health()
        print("Omni: \(status.asr)\nOllama: \(status.ollama)")
        let model = DictationSession(recorder: UnusedRecorder(), service: service)
        model.polishEnabled = true
        model.submitAudio(wav)
        let deadline = Date().addingTimeInterval(370)
        while model.isBusy, Date() < deadline { try await Task.sleep(nanoseconds: 50_000_000) }
        guard model.phase == .ready, !model.rawText.isEmpty, model.hasPolishedResult else {
            throw DictationError("真实后端验证未通过：\(model.notice)")
        }
        print("原文：\(model.rawText)\n整理：\(model.resultText)")
        for (stage, seconds) in [("ASR", model.asrSeconds), ("Ollama", model.polishSeconds), ("总请求", model.totalSeconds)] {
            if let seconds { print(String(format: "%@：%.3f 秒；单次观测 n=1，无法计算 std。", stage, seconds)) }
        }
        print("PASS: same Swift audio encoder, session and backend service as the native client; microphone not exercised")
    }
}
