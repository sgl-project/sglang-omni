import Combine
import Foundation

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
}

@MainActor
private func wait(_ condition: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 3
    while !condition(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    try expect(condition(), "等待状态超时")
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
private final class Target: DictationTextTarget {
    let applicationName = "当前光标替身"
    let supportsConfirmation = false
    let dismissesFeedbackAfterPaste = true
    var writes = 0
    func validate() throws {}
    func insert(_ text: String) throws { writes += 1 }
    func confirms(_ text: String) -> Bool { false }
    func stopObserving() {}
}

@MainActor
private final class Fixture {
    let service = Service()
    let target = Target()
    let insertion = DictationInsertion()
    let session: DictationSession
    let feedback: DictationFeedback
    init(reduceMotion: Bool = false) {
        session = DictationSession(recorder: Recorder(), service: service)
        feedback = DictationFeedback(session: session, insertion: insertion,
                                     holdDuration: 0.06, fadeDuration: 0.08,
                                     reduceMotion: { reduceMotion })
        session.onResult = { [weak self] in self?.insertion.complete($0) }
    }
    func begin() async throws {
        insertion.prepare(target)
        session.submitAudio(Data([1]))
        try await wait { self.service.reply != nil }
    }
    func finish() {
        service.reply?.resume(returning: "保留本轮文字")
        service.reply = nil
    }
    func cancel() { insertion.cancel(); session.cancel() }
}

@main
private enum HoverFeedbackTests {
    @MainActor
    static func main() async {
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        for moment in ["beforeCompletion", "hold", "fade"] {
            let f = Fixture()
            defer { f.cancel() }
            try await f.begin()
            if moment == "beforeCompletion" { f.feedback.setHovered(true) }
            f.finish()
            try await wait { f.insertion.didTriggerPaste }
            if moment == "fade" { try await wait { f.feedback.opacity > 0 && f.feedback.opacity < 1 } }
            f.feedback.setHovered(true)
            try expect(f.feedback.opacity == 1, "移入浮条立即恢复完整显示：\(moment)")
            try await Task.sleep(nanoseconds: 220_000_000)
            try expect(f.feedback.opacity == 1, "悬停期间不能隐藏：\(moment)")
            f.feedback.setHovered(false)
            try await wait { f.feedback.opacity > 0 && f.feedback.opacity < 1 }
            // Re-entering during the leave fade must cancel the previous fade task.
            f.feedback.setHovered(true)
            try await Task.sleep(nanoseconds: 220_000_000)
            try expect(f.feedback.opacity == 1, "再次移入应取消旧渐隐")
            f.feedback.setHovered(false)
            try await wait { f.feedback.opacity == 0 }
            f.feedback.setHovered(true)
            f.feedback.setHovered(false)
            try await Task.sleep(nanoseconds: 220_000_000)
            try expect(f.feedback.opacity == 0, "隐藏后的迟到鼠标事件不能重新弹出提示")
            try expect(f.session.rawText == "保留本轮文字" && f.target.writes == 1 && !f.insertion.didInsert,
                       "悬停只影响显示，不清空、不重写、不伪报已确认填入")
        }

        for action in ["dismiss", "cancel", "newRound"] {
            let f = Fixture()
            defer { f.cancel() }
            try await f.begin()
            f.finish()
            try await wait { f.insertion.didTriggerPaste }
            f.feedback.setHovered(true)
            if action == "dismiss" { f.feedback.dismiss() }
            else if action == "cancel" { f.cancel() }
            else { try await f.begin() }
            f.feedback.setHovered(false)
            try await Task.sleep(nanoseconds: 220_000_000)
            try expect(f.feedback.opacity == (action == "newRound" ? 1 : 0),
                       "悬停结束不能撤销关闭/取消或隐藏新一轮：\(action)")
            if action == "newRound" {
                f.finish()
                try await wait { f.feedback.opacity == 0 }
                try expect(f.target.writes == 2, "新一轮仍正常完成")
            } else {
                f.feedback.setHovered(true)
                try expect(f.feedback.opacity == 0, "关闭后不能被悬停重新唤起")
            }
        }

        let failure = Fixture()
        defer { failure.cancel() }
        failure.service.polishFails = true
        failure.session.polishEnabled = true
        try await failure.begin()
        failure.feedback.setHovered(true)
        failure.feedback.setHovered(false)
        try await Task.sleep(nanoseconds: 220_000_000)
        try expect(failure.feedback.opacity == 1, "处理中移开不能隐藏浮条")
        failure.finish()
        try await wait { failure.insertion.didTriggerPaste }
        failure.feedback.setHovered(true)
        try await Task.sleep(nanoseconds: 220_000_000)
        try expect(failure.feedback.opacity == 1, "回退原文后悬停应保持提示")
        failure.feedback.setHovered(false)
        try await wait { failure.feedback.opacity == 0 }
        try expect(!failure.session.notice.isEmpty && failure.target.writes == 1,
                   "回退原文已投递时移开应隐藏浮条，但保留原因且不重复粘贴")

        let reduced = Fixture(reduceMotion: true)
        defer { reduced.cancel() }
        var opacity: [Double] = []
        let observer = reduced.feedback.$opacity.sink { opacity.append($0) }
        try await reduced.begin()
        reduced.feedback.setHovered(true)
        reduced.finish()
        try await wait { reduced.insertion.didTriggerPaste }
        try await Task.sleep(nanoseconds: 220_000_000)
        try expect(reduced.feedback.opacity == 1, "减少动态效果时悬停同样保持显示")
        reduced.feedback.setHovered(false)
        try await wait { reduced.feedback.opacity == 0 }
        try expect(!opacity.contains { $0 > 0 && $0 < 1 }, "减少动态效果时移开直接隐藏")
        observer.cancel()
        print("PASS: hover before completion, hold/fade interruption, leave fade, re-entry, cancellation, new rounds, errors and reduced motion")
    }
}
