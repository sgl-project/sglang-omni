import Combine
import Foundation

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
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
        return "整理结果"
    }
}

@MainActor
private final class Target: DictationTextTarget {
    let applicationName = "Codex"
    var confirmed = false
    var fails = false
    var warning: String?
    var writes: [String] = []
    func validate() throws {}
    func insert(_ text: String) throws {
        if fails { throw DictationError("粘贴失败") }
        writes.append(text)
    }
    func confirms(_ text: String) -> Bool { confirmed }
    func stopObserving() {}
    func finishInsertion(confirmed: Bool) -> String? { warning }
}

@MainActor
private final class Fixture {
    let service = Service()
    let target = Target()
    let session: DictationSession
    let insertion: DictationInsertion
    let feedback: DictationFeedback
    init(verificationLimit: Double = 2, reduceMotion: Bool = false) {
        session = DictationSession(recorder: Recorder(), service: service)
        insertion = DictationInsertion(verificationLimit: verificationLimit)
        feedback = DictationFeedback(session: session, insertion: insertion,
                                     holdDuration: 0.06, fadeDuration: 0.08,
                                     reduceMotion: { reduceMotion })
        session.onResult = { [weak self] in self?.insertion.complete($0) }
    }
    func begin(withTarget: Bool = true) async throws {
        insertion.cancel()
        if withTarget { insertion.prepare(target) }
        session.submitAudio(Data([1]))
        try await wait { self.service.reply != nil }
    }
    func finish(_ text: String = "原文") { service.reply?.resume(returning: text); service.reply = nil }
    func cancel() { insertion.cancel(); session.cancel() }
}

@MainActor
private func wait(_ predicate: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 3
    while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    try expect(predicate(), "等待状态超时")
}

@main
private enum FeedbackTests {
    @MainActor
    static func main() async throws {
        let success = Fixture()
        var alpha: [Double] = []
        let observer = success.feedback.$opacity.sink { alpha.append($0) }
        try expect(success.feedback.opacity == 0, "启动时不显示浮条")
        try await success.begin()
        try expect(success.feedback.opacity == 1, "转写期间显示浮条")
        success.finish()
        try await wait { success.insertion.isDelivering }
        try await Task.sleep(nanoseconds: 180_000_000)
        try expect(success.session.phase == .ready && success.feedback.opacity == 1,
                   "ASR ready 不能触发淡出，必须等回填确认")
        success.target.confirmed = true
        try await wait { success.insertion.didInsert }
        try expect(success.feedback.opacity == 1, "成功后先停留")
        try await wait { success.feedback.opacity == 0 }
        try expect(alpha.contains { $0 > 0 && $0 < 1 }, "成功经过渐隐而非直接消失")
        try expect(success.session.rawText == "原文" && success.session.resultText == "原文",
                   "淡出只隐藏浮条，结果仍可主动查看")
        try expect(success.target.writes == ["原文"] && success.insertion.didInsert,
                   "淡出不取消、重复插入或改变成功状态")
        observer.cancel()

        for duringFade in [false, true] {
            let next = Fixture()
            next.target.confirmed = true
            try await next.begin()
            next.finish()
            try await wait { next.insertion.didInsert }
            if duringFade { try await wait { next.feedback.opacity < 1 } }
            next.insertion.cancel()
            next.session.toggleRecording()
            try await wait { next.session.phase == .recording }
            try await Task.sleep(nanoseconds: 180_000_000)
            try expect(next.feedback.opacity == 1, "旧停留/淡出任务不能隐藏新录音")
            next.cancel()
            try expect(next.feedback.opacity == 0, "取消立即隐藏")
        }

        for duringFade in [false, true] {
            let cancelled = Fixture()
            cancelled.target.confirmed = true
            try await cancelled.begin()
            cancelled.finish()
            try await wait { cancelled.insertion.didInsert }
            if duringFade { try await wait { cancelled.feedback.opacity < 1 } }
            cancelled.cancel()
            try await Task.sleep(nanoseconds: 180_000_000)
            try expect(cancelled.feedback.opacity == 0 && cancelled.session.rawText.isEmpty,
                       "取消在停留/淡出阶段均使旧任务失效")
        }

        for scenario in ["unconfirmed", "writeFailure", "noTarget", "empty", "asrFailure", "polishFailure", "clipboardWarning"] {
            let failure = Fixture(verificationLimit: 0.03)
            failure.target.confirmed = scenario != "unconfirmed"
            failure.target.fails = scenario == "writeFailure"
            if scenario == "clipboardWarning" { failure.target.warning = "剪贴板未完整恢复" }
            failure.session.polishEnabled = scenario == "polishFailure"
            failure.service.polishFails = scenario == "polishFailure"
            try await failure.begin(withTarget: scenario != "noTarget")
            if scenario == "asrFailure" {
                failure.service.reply?.resume(throwing: DictationError("ASR 失败"))
                failure.service.reply = nil
            } else { failure.finish(scenario == "empty" ? "" : "原文") }
            try await wait { !failure.session.isBusy && !failure.insertion.isDelivering }
            try await Task.sleep(nanoseconds: 180_000_000)
            if scenario == "polishFailure" {
                try await wait { failure.feedback.opacity == 0 }
                try expect(failure.insertion.didInsert && !failure.session.notice.isEmpty,
                           "回退原文已确认填入时应淡出，仍保留整理失败原因")
            } else {
                try expect(failure.feedback.opacity == 1, "异常不能自动隐去：\(scenario)")
            }
            let raw = failure.session.rawText
            failure.feedback.dismiss()
            try expect(failure.feedback.opacity == 0 && failure.session.rawText == raw,
                       "主动关闭提示保留本轮结果")
            failure.cancel()
        }

        let reduced = Fixture(reduceMotion: true)
        var reducedAlpha: [Double] = []
        let reducedObserver = reduced.feedback.$opacity.sink { reducedAlpha.append($0) }
        reduced.target.confirmed = true
        try await reduced.begin()
        reduced.finish()
        try await wait { reduced.feedback.opacity == 0 }
        try expect(reduced.insertion.didInsert && !reducedAlpha.contains { $0 > 0 && $0 < 1 },
                   "减少动态效果时成功提示停留后直接收起")
        reducedObserver.cancel()
        print("PASS: confirmed delivery, fading, result retention, new rounds, cancellation, failures and reduced motion")
    }
}
