import Combine
import Foundation

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
}

@MainActor
private func wait(_ message: String, until condition: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 3
    while !condition(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    try expect(condition(), message)
}

@MainActor
private final class Recorder: AudioRecording {
    func start() async throws {}
    func stop() throws -> Data { Data([1]) }
    func cancel() {}
}

@MainActor
private final class Service: SpeechServing {
    var reply: CheckedContinuation<String, Error>?
    func transcribe(wav: Data) async throws -> String { "可以换快捷键吗？" }
    func polish(text: String) async throws -> String {
        try await withCheckedThrowingContinuation { reply = $0 }
    }
}

@MainActor
private final class Target: DictationTextTarget {
    let applicationName = "测试输入框"
    let supportsConfirmation: Bool
    var dismissesFeedbackAfterPaste: Bool { !supportsConfirmation }
    var writes: [String] = []
    init(confirmed: Bool) { supportsConfirmation = confirmed }
    func validate() throws {}
    func insert(_ text: String) throws { writes.append(text) }
    func confirms(_ text: String) -> Bool { supportsConfirmation }
    func stopObserving() {}
}

@main
private enum FallbackFeedbackTests {
    @MainActor
    static func main() async {
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        for confirmed in [false, true] {
            for hover in ["none", "beforeCompletion", "duringFade"] {
                let service = Service()
                let target = Target(confirmed: confirmed)
                let insertion = DictationInsertion()
                let session = DictationSession(recorder: Recorder(), service: service)
                let feedback = DictationFeedback(session: session, insertion: insertion,
                                                 holdDuration: 0.06, fadeDuration: 0.08)
                defer { insertion.cancel(); session.cancel() }
                session.polishEnabled = true
                session.onResult = { insertion.complete($0) }
                insertion.prepare(target)
                session.submitAudio(Data([1]))
                try await wait("整理未开始") { service.reply != nil }
                feedback.setHovered(true)
                feedback.setHovered(false)
                try await Task.sleep(nanoseconds: 220_000_000)
                try expect(feedback.opacity == 1, "整理未完成时移开鼠标不能隐藏")
                if hover == "beforeCompletion" { feedback.setHovered(true) }
                service.reply?.resume(throwing: DictationError("Ollama 未正常完成整理，可能返回空文本或达到输出上限。"))
                service.reply = nil
                try await wait("原文未完成投递") {
                    insertion.didTriggerPaste && !insertion.isDelivering
                }
                if hover == "duringFade" {
                    try await wait("回退原文已投递，但未进入渐隐") { feedback.opacity > 0 && feedback.opacity < 1 }
                    feedback.setHovered(true)
                }
                if hover != "none" {
                    try await Task.sleep(nanoseconds: 220_000_000)
                    try expect(feedback.opacity == 1, "回退原文后悬停仍保持显示")
                    feedback.setHovered(false)
                    try await wait("移开后未渐隐") { feedback.opacity > 0 && feedback.opacity < 1 }
                    feedback.setHovered(true)
                    try await Task.sleep(nanoseconds: 220_000_000)
                    try expect(feedback.opacity == 1, "再次移入应取消旧渐隐")
                    feedback.setHovered(false)
                }
                try await wait("回退原文已正常投递，浮条却没有自动收起") { feedback.opacity == 0 }
                try expect(session.rawText == "可以换快捷键吗？" && session.resultText == session.rawText
                           && !session.hasPolishedResult && session.notice.contains("已保留原文"),
                           "渐隐必须保留原文和整理失败原因")
                try expect(target.writes == [session.rawText] && insertion.didInsert == confirmed,
                           "渐隐不能重复投递或伪报已确认填入")
                feedback.setHovered(true)
                try expect(feedback.opacity == 0, "隐藏后的鼠标事件不能重开浮条")
            }
        }
        print("PASS: fallback delivery fades in both modes; hover/re-entry, processing visibility, result retention and accurate delivery status")
    }
}
