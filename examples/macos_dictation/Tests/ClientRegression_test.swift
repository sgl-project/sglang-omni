import Foundation

struct CheckFailure: LocalizedError, CustomStringConvertible {
    let description: String
    var errorDescription: String? { description }
}

func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw CheckFailure(description: message) }
}

@MainActor
final class TestRecorder: AudioRecording {
    var startHandler: () async throws -> Void = {}
    var stopError: Error?
    func start() async throws { try await startHandler() }
    func stop() throws -> Data {
        if let stopError { throw stopError }
        return Data([1, 2, 3])
    }
    func cancel() {}
}

@MainActor
final class TestService: SpeechServing {
    var asr: (Data) async throws -> String = { _ in "这个 PR 不要合并。" }
    var cleanup: (String) async throws -> String = { _ in "这个 PR 先不要合并。" }
    var asrCalls = 0
    var polishCalls = 0
    func transcribe(wav: Data) async throws -> String {
        asrCalls += 1
        return try await asr(wav)
    }
    func polish(text: String) async throws -> String {
        polishCalls += 1
        return try await cleanup(text)
    }
}

@main
struct RegressionMain {
    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(3)
        while !predicate(), Date() < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        try check(predicate(), "等待状态超时")
    }

    @MainActor
    static func main() async throws {
        let service = TestService()
        let recorder = TestRecorder()
        let session = DictationSession(recorder: recorder, service: service)
        session.toggleRecording()
        try await wait { session.phase == .recording }
        session.toggleRecording()
        try await wait { session.phase == .ready }
        try check(session.resultText == session.rawText && service.polishCalls == 0, "默认原文模式不应调用整理")

        session.polishEnabled = true
        session.submitAudio(Data([1]))
        try await wait { session.phase == .ready }
        try check(session.rawText == "这个 PR 不要合并。" && session.resultText == "这个 PR 先不要合并。", "必须保留两个版本")

        service.cleanup = { _ in throw CheckFailure(description: "Ollama 连接失败") }
        session.submitAudio(Data([1]))
        try await wait { session.phase == .ready }
        try check(session.resultText == session.rawText && session.notice.contains("原文"), "整理失败不能丢失原文")

        service.asr = { _ in " \n" }
        let polishCount = service.polishCalls
        session.submitAudio(Data([1]))
        try await wait { session.phase == .ready }
        try check(service.polishCalls == polishCount && session.notice.contains("没有识别"), "空转写不应继续整理")

        var oldASR: CheckedContinuation<String, Error>?
        service.asr = { _ in try await withCheckedThrowingContinuation { oldASR = $0 } }
        session.submitAudio(Data([1]))
        try await wait { oldASR != nil }
        let calls = service.asrCalls
        session.toggleRecording()
        session.submitAudio(Data([2]))
        try check(service.asrCalls == calls, "忙碌时不能重复提交")
        session.cancel()
        oldASR?.resume(returning: "迟到的 ASR")
        try await Task.sleep(nanoseconds: 10_000_000)
        try check(session.phase == .idle && session.rawText.isEmpty, "取消后应丢弃迟到结果")

        var oldPolish: CheckedContinuation<String, Error>?
        service.asr = { _ in "旧原文" }
        service.cleanup = { _ in try await withCheckedThrowingContinuation { oldPolish = $0 } }
        session.submitAudio(Data([1]))
        try await wait { oldPolish != nil }
        session.cancel()
        session.polishEnabled = false
        service.asr = { _ in "新原文" }
        session.submitAudio(Data([2]))
        try await wait { session.phase == .ready }
        oldPolish?.resume(returning: "迟到的整理")
        try await Task.sleep(nanoseconds: 10_000_000)
        try check(session.resultText == "新原文", "旧整理不能覆盖新一轮")

        var permission: CheckedContinuation<Void, Error>?
        recorder.startHandler = { try await withCheckedThrowingContinuation { permission = $0 } }
        session.toggleRecording()
        try await wait { permission != nil }
        session.cancel()
        permission?.resume()
        try await Task.sleep(nanoseconds: 10_000_000)
        try check(session.phase == .idle && session.recordingStartedAt == nil, "取消授权等待后不能开始录音")

        recorder.startHandler = { throw CheckFailure(description: "麦克风权限拒绝") }
        session.toggleRecording()
        try await wait { session.phase == .failed }
        try check(session.notice.contains("麦克风") && !session.isBusy, "权限失败应给出提示")

        recorder.startHandler = {}
        recorder.stopError = CheckFailure(description: "没有录到音频")
        session.toggleRecording()
        try await wait { session.phase == .recording }
        session.toggleRecording()
        try check(session.phase == .failed, "空录音必须失败")

        let failureService = TestService()
        failureService.asr = { _ in throw DictationError("ASR 测试连接失败") }
        let failureSession = DictationSession(recorder: TestRecorder(), service: failureService)
        failureSession.submitAudio(Data([1]))
        try await wait { failureSession.phase == .failed }
        try check(failureSession.rawText.isEmpty && failureSession.notice.contains("ASR"), "转写失败不应生成结果")

        let limited = DictationSession(recorder: TestRecorder(), service: TestService(), recordingLimit: 0.02)
        limited.toggleRecording()
        try await wait { limited.phase == .ready }
        try check(!limited.rawText.isEmpty, "达到时长上限后应停止并转写")

        try check(LocalSpeechService.polishRequest(text: "原文").httpMethod == "POST", "整理请求必须使用 POST")
        let payload = try JSONSerialization.jsonObject(with: LocalSpeechService.polishRequest(text: "原文").httpBody!) as! [String: Any]
        try check(payload["think"] as? Bool == false && payload["stream"] as? Bool == false, "Ollama 必须关闭 think 和 stream")
        let messages = payload["messages"] as! [[String: String]]
        let quotedInput = try JSONSerialization.jsonObject(with: Data(messages.last!["content"]!.utf8)) as! [String: String]
        try check(quotedInput["transcript"] == "原文", "校对任务中的原文必须完整保留，不得作为裸问句发送")
        let upload = LocalSpeechService.transcriptionRequest(wav: Data("AUDIO_SENTINEL".utf8))
        let multipart = String(data: upload.httpBody!, encoding: .utf8)!
        try check(multipart.contains("name=\"file\"") && multipart.contains("AUDIO_SENTINEL")
                  && multipart.contains("Qwen/Qwen3-ASR-0.6B") && multipart.contains("name=\"response_format\""),
                  "Omni 上传必须包含音频、模型名和返回格式")
        do {
            _ = try LocalSpeechService.decodePolish(Data(#"{"done":true,"done_reason":"length","message":{"content":"截断"}}"#.utf8))
            throw CheckFailure(description: "截断的整理结果不能通过")
        } catch let failure as CheckFailure { throw failure } catch {}
        do {
            _ = try LocalSpeechService.decodePolish(Data(#"{"done":true,"done_reason":"stop","message":{"content":"  "}}"#.utf8))
            throw CheckFailure(description: "空整理结果不能通过")
        } catch let failure as CheckFailure { throw failure } catch {}

        let samples = (0..<48000).map { Float(sin(Double($0) * 2 * .pi * 440 / 48000) * 0.1) }
        let wav = try AudioEncoder.wav(samples: samples, sampleRate: 48000)
        try check(String(data: wav.prefix(4), encoding: .ascii) == "RIFF", "WAV 必须含 RIFF 头")
        try check(String(data: wav[8..<12], encoding: .ascii) == "WAVE", "WAV 标记错误")
        let frames = (wav.count - 44) / 2
        try check(abs(frames - 16000) < 400, "重采样长度异常")
        try check(wav[22] == 1 && wav[34] == 16, "应为单声道 PCM16")
        print("PASS: session cancellation/isolation/errors/limit, request contracts, PCM WAV conversion")
    }
}
