import Combine
import Foundation

@MainActor
private final class Recorder: AudioRecording {
    func start() async throws { }
    func stop() throws -> Data { Data([1]) }
    func cancel() { }
}

@MainActor
private final class Service: SpeechServing {
    var reply: CheckedContinuation<String, Error>?
    func transcribe(wav: Data) async throws -> String {
        try await withCheckedThrowingContinuation { reply = $0 }
    }
    func polish(text: String) async throws -> String { text }
}

@MainActor
private final class Target: DictationTextTarget {
    let applicationName = "测试输入位置"
    var supportsConfirmation = true
    var confirmed = true
    var allowed = true
    var fails = false
    var warning: String?
    var writes = 0
    func validate() throws { if !allowed { throw DictationError("输入位置已变化") } }
    func insert(_ text: String) throws {
        if fails { throw DictationError("粘贴失败") }
        writes += 1
    }
    func confirms(_ text: String) -> Bool { confirmed }
    func stopObserving() { }
    func finishInsertion(confirmed: Bool) -> String? { warning }
}

@MainActor
private final class Fixture {
    let service = Service()
    let target = Target()
    let session: DictationSession
    let insertion: DictationInsertion
    let feedback: DictationFeedback

    init(verificationLimit: Double = 0.04, reduceMotion: Bool = false) {
        session = DictationSession(recorder: Recorder(), service: service)
        insertion = DictationInsertion(verificationLimit: verificationLimit)
        feedback = DictationFeedback(session: session, insertion: insertion,
                                     holdDuration: 0.06, fadeDuration: 0.1, attentionHoldDuration: 0.08,
                                     reduceMotion: { reduceMotion })
        session.onResult = { [weak self] in self?.insertion.complete($0) }
    }

    func begin(withTarget: Bool = true) async throws {
        insertion.cancel()
        if withTarget { insertion.prepare(target) }
        else { insertion.abandon("文件转写仅供查看和复制。") }
        session.submitAudio(Data([1]))
        try await wait { self.service.reply != nil }
    }

    func finish(_ text: String = "保留本轮原文") {
        service.reply?.resume(returning: text)
        service.reply = nil
    }

    func cancel() { insertion.cancel(); session.cancel() }
}

@MainActor
private func wait(_ predicate: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 3
    while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    guard predicate() else { throw DictationError("等待渐隐或会话状态超时") }
}

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
    precondition(condition(), message)
}

@main
private enum CompletionFeedbackTests {
    @MainActor
    static func main() async throws {
        for scenario in ["empty", "import", "asrFailure", "changedTarget", "writeFailure", "unconfirmed", "compatibility", "clipboardWarning"] {
            for hover in [false, true] {
                let f = Fixture()
                defer { f.cancel() }
                f.target.fails = scenario == "writeFailure"
                f.target.confirmed = scenario != "unconfirmed"
                f.target.supportsConfirmation = scenario != "compatibility"
                if scenario == "clipboardWarning" { f.target.warning = "剪贴板未完整恢复" }
                try await f.begin(withTarget: scenario != "import")
                f.feedback.setHovered(hover)
                f.target.allowed = scenario != "changedTarget"
                if scenario == "asrFailure" {
                    f.service.reply?.resume(throwing: DictationError("转写失败"))
                    f.service.reply = nil
                } else { f.finish(scenario == "empty" ? "" : "保留本轮原文") }
                try await wait { !f.session.isBusy && !f.insertion.isDelivering }
                let raw = f.session.rawText
                let notice = f.session.notice
                let message = f.insertion.message
                let inserted = f.insertion.didInsert
                let writes = f.target.writes
                if hover {
                    try await Task.sleep(nanoseconds: 250_000_000)
                    expect(f.feedback.opacity == 1, "已结束的提示在悬停时应保留：\(scenario)")
                    f.feedback.setHovered(false)
                    try await wait { f.feedback.opacity > 0 && f.feedback.opacity < 1 }
                    f.feedback.setHovered(true)
                    try await Task.sleep(nanoseconds: 250_000_000)
                    expect(f.feedback.opacity == 1, "重新悬停应中止渐隐：\(scenario)")
                    f.feedback.setHovered(false)
                }
                try await wait { f.feedback.opacity == 0 }
                expect(f.session.rawText == raw && f.session.notice == notice && f.insertion.message == message,
                       "收起提示不能删除原文或错误原因：\(scenario)")
                expect(f.target.writes == writes && f.insertion.didInsert == inserted,
                       "收起提示不能重复回填或伪报成功：\(scenario)")
                f.feedback.setHovered(true)
                expect(f.feedback.opacity == 0, "迟到的鼠标事件不能重开提示")
            }
        }

        let pending = Fixture(verificationLimit: 2)
        defer { pending.cancel() }
        pending.target.confirmed = false
        try await pending.begin()
        try await Task.sleep(nanoseconds: 250_000_000)
        expect(pending.feedback.opacity == 1, "处理过程中不能自动隐藏")
        pending.finish()
        try await wait { pending.insertion.isDelivering }
        try await Task.sleep(nanoseconds: 250_000_000)
        expect(pending.feedback.opacity == 1, "回填确认期间不能自动隐藏")
        pending.target.confirmed = true
        try await wait { pending.feedback.opacity == 0 }

        for action in ["cancel", "newRound", "dismiss"] {
            let f = Fixture()
            defer { f.cancel() }
            try await f.begin()
            f.finish("")
            try await wait { f.feedback.opacity > 0 && f.feedback.opacity < 1 }
            if action == "cancel" { f.cancel() }
            else if action == "dismiss" { f.feedback.dismiss() }
            else { try await f.begin() }
            try await Task.sleep(nanoseconds: 250_000_000)
            expect(f.feedback.opacity == (action == "newRound" ? 1 : 0), "旧提示不能干扰取消、关闭或新一轮")
            if action == "newRound" { f.finish(); try await wait { f.feedback.opacity == 0 } }
        }

        let reduced = Fixture(reduceMotion: true)
        defer { reduced.cancel() }
        var opacity: [Double] = []
        let observer = reduced.feedback.$opacity.sink { opacity.append($0) }
        reduced.session.reportInputFailure("麦克风权限未开启")
        try await wait { reduced.feedback.opacity == 0 }
        expect(!opacity.contains { $0 > 0 && $0 < 1 } && !reduced.session.notice.isEmpty,
               "减少动态效果时也应收起错误提示，保留具体原因")
        observer.cancel()
        print("PASS: all terminal outcomes dismiss, hover/re-entry, processing visibility, retained results and cancellation")
    }
}
