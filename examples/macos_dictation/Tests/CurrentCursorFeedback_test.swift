import AppKit
import Combine

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
    var polishFails = false
    func transcribe(wav: Data) async throws -> String {
        try await withCheckedThrowingContinuation { reply = $0 }
    }
    func polish(text: String) async throws -> String {
        if polishFails { throw DictationError("整理失败") }
        return text
    }
}

@MainActor
private final class CompatibilityTarget: DictationTextTarget {
    let applicationName = "锁定模式兼容输入框"
    let supportsConfirmation = false
    func validate() throws {}
    func insert(_ text: String) throws {}
    func confirms(_ text: String) -> Bool { false }
    func stopObserving() {}
}

@MainActor
private final class Fixture {
    let board = NSPasteboard.withUniqueName()
    let service = Service()
    let session: DictationSession
    let insertion = DictationInsertion()
    let feedback: DictationFeedback
    var allowed = true
    var postFails = false
    var posts = 0

    init(reduceMotion: Bool = false) {
        session = DictationSession(recorder: Recorder(), service: service)
        feedback = DictationFeedback(session: session, insertion: insertion,
                                     holdDuration: 0.06, fadeDuration: 0.08,
                                     reduceMotion: { reduceMotion })
        session.onResult = { [weak self] in self?.insertion.complete($0) }
    }

    func begin(compatibility: Bool = false) async throws {
        if compatibility { insertion.prepare(CompatibilityTarget()) }
        else {
            let paste = ClipboardPaste(pasteboard: board) { [weak self] _, _ in
                guard let self else { return }
                if self.postFails { throw DictationError("投递失败") }
                self.posts += 1
            }
            insertion.prepare(CurrentCursorTarget(paste: paste, canPost: { [weak self] in self?.allowed == true }))
        }
        session.submitAudio(Data([1]))
        try await wait("ASR 请求未开始") { self.service.reply != nil }
    }

    func finish(_ text: String = "原文😀") {
        service.reply?.resume(returning: text)
        service.reply = nil
    }

    func cleanup() {
        insertion.cancel()
        session.cancel()
        board.releaseGlobally()
    }
}

@main
private enum CurrentCursorFeedbackTests {
    @MainActor
    static func main() async {
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        // Real current-cursor target and clipboard helper; only a named test pasteboard
        // and injected key transport. Never records audio, inspects apps or posts keys.
        let normal = Fixture()
        defer { normal.cleanup() }
        var opacity: [Double] = []
        let observer = normal.feedback.$opacity.sink { opacity.append($0) }
        try await normal.begin()
        normal.finish()
        try await wait("粘贴未触发") { normal.insertion.didTriggerPaste }
        try expect(normal.feedback.opacity == 1, "粘贴后应先显示提示")
        try await wait("当前光标已触发粘贴，但浮条没有自动渐隐") { normal.feedback.opacity == 0 }
        try expect(opacity.contains { $0 > 0 && $0 < 1 }, "应该经过渐隐")
        try expect(normal.insertion.didTriggerPaste && !normal.insertion.didInsert
                   && normal.insertion.message.contains("无法确认"), "渐隐不能伪报已确认填入")
        try expect(normal.session.rawText == "原文😀" && normal.session.resultText == "原文😀",
                   "渐隐后仍可查看本轮原文和结果")
        try expect(normal.posts == 1 && normal.board.string(forType: .string) == "原文😀",
                   "渐隐不重复粘贴、不恢复或清空未确认的剪贴板")
        observer.cancel()

        for duringFade in [false, true] {
            let next = Fixture()
            defer { next.cleanup() }
            try await next.begin()
            next.finish()
            try await wait("粘贴未触发") { next.insertion.didTriggerPaste }
            if duringFade { try await wait("未进入渐隐") { next.feedback.opacity < 1 } }
            try await next.begin()
            try await Task.sleep(nanoseconds: 200_000_000)
            try expect(next.feedback.opacity == 1, "旧渐隐不能隐藏新一轮")
            next.insertion.cancel()
            next.session.cancel()
            next.finish("取消后的迟到结果")
            try await Task.sleep(nanoseconds: 200_000_000)
            try expect(next.feedback.opacity == 0 && next.session.rawText.isEmpty && next.posts == 1,
                       "取消应隐藏并丢弃迟到结果")
        }

        for scenario in ["permission", "postFailure", "polishFailure", "asrFailure", "empty", "compatibility"] {
            let failure = Fixture()
            defer { failure.cleanup() }
            failure.allowed = scenario != "permission"
            failure.postFails = scenario == "postFailure"
            failure.session.polishEnabled = scenario == "polishFailure"
            failure.service.polishFails = scenario == "polishFailure"
            try await failure.begin(compatibility: scenario == "compatibility")
            if scenario == "asrFailure" {
                failure.service.reply?.resume(throwing: DictationError("ASR 失败"))
                failure.service.reply = nil
            } else { failure.finish(scenario == "empty" ? "" : "原文") }
            try await wait("结果未结束") { !failure.session.isBusy }
            try await Task.sleep(nanoseconds: 200_000_000)
            if scenario == "polishFailure" {
                try await wait("回退原文已粘贴但未淡出") { failure.feedback.opacity == 0 }
                try expect(failure.insertion.didTriggerPaste && !failure.session.notice.isEmpty,
                           "已粘贴原文后应淡出，整理失败原因仍保留在结果中")
            } else {
                try expect(failure.feedback.opacity == 1, "异常或旧兼容模式提示应保留：\(scenario)")
            }
        }

        let copied = Fixture(reduceMotion: true)
        defer { copied.cleanup() }
        var reducedOpacity: [Double] = []
        let reducedObserver = copied.feedback.$opacity.sink { reducedOpacity.append($0) }
        try await copied.begin()
        copied.finish()
        try await wait("粘贴未触发") { copied.insertion.didTriggerPaste }
        copied.board.clearContents()
        copied.board.setString("用户后来复制的文字", forType: .string)
        try await wait("减少动态效果时也应自动收起") { copied.feedback.opacity == 0 }
        try expect(!reducedOpacity.contains { $0 > 0 && $0 < 1 }
                   && copied.board.string(forType: .string) == "用户后来复制的文字",
                   "减少动态效果直接收起，且不覆盖用户新复制的内容")
        reducedObserver.cancel()
        print("PASS: current-cursor fading, accurate status, text/clipboard retention, new rounds, cancellation, failures and reduced motion")
    }
}
