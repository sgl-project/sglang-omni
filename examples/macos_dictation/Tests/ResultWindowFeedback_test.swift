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
    let applicationName = "test editor"
    var confirmed = true
    var writes = 0
    func validate() throws { }
    func insert(_ text: String) throws { writes += 1 }
    func confirms(_ text: String) -> Bool { confirmed }
    func stopObserving() { }
}

@MainActor
private final class Fixture {
    let service = Service()
    let target = Target()
    let session: DictationSession
    let insertion = DictationInsertion(verificationLimit: 2)
    let result: ResultWindowFeedback
    let floating: DictationFeedback

    init(reduceMotion: Bool = false) {
        session = DictationSession(recorder: Recorder(), service: service)
        result = ResultWindowFeedback(session: session, insertion: insertion,
                                      holdDuration: 0.06, fadeDuration: 0.1,
                                      attentionHoldDuration: 0.08, reduceMotion: { reduceMotion })
        floating = DictationFeedback(session: session, insertion: insertion,
                                     holdDuration: 0.06, fadeDuration: 0.1)
        session.onResult = { [weak self] in self?.insertion.complete($0) }
    }

    func begin(withTarget: Bool = true) async throws {
        insertion.cancel()
        if withTarget { insertion.prepare(target) }
        session.submitAudio(Data([1]))
        try await wait { self.service.reply != nil }
    }

    func finish() {
        service.reply?.resume(returning: "保留结果文字")
        service.reply = nil
    }

    func cancel() {
        insertion.cancel()
        session.cancel()
        finish()
    }
}

@MainActor
private func wait(_ predicate: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 3
    while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    precondition(predicate(), "Timed out waiting for presentation state")
}

@main
private enum ResultWindowFeedbackTests {
    @MainActor
    static func main() async throws {
        let completed = Fixture()
        defer { completed.cancel() }
        completed.result.show()
        try await Task.sleep(nanoseconds: 250_000_000)
        precondition(completed.result.opacity == 1, "Manually opened idle results must stay visible")
        try await completed.begin()
        completed.floating.setHovered(true)
        completed.finish()
        try await wait { completed.result.opacity > 0 && completed.result.opacity < 1 }
        try await wait { completed.result.opacity == 0 }
        precondition(completed.floating.opacity == 1, "Floating hover must remain independent")
        precondition(completed.session.rawText == "保留结果文字" && completed.target.writes == 1,
                     "Hiding must preserve text and must not repeat delivery")
        completed.result.show()
        try await Task.sleep(nanoseconds: 250_000_000)
        precondition(completed.result.opacity == 1, "Reopening completed results must stay visible")

        let pending = Fixture()
        defer { pending.cancel() }
        pending.target.confirmed = false
        try await pending.begin()
        pending.result.show()
        try await Task.sleep(nanoseconds: 250_000_000)
        precondition(pending.result.opacity == 1, "Recognition must not dismiss the result window")
        pending.finish()
        try await wait { pending.insertion.isDelivering }
        try await Task.sleep(nanoseconds: 250_000_000)
        precondition(pending.result.opacity == 1, "Pending paste confirmation must not dismiss results")
        pending.target.confirmed = true
        try await wait { pending.result.opacity == 0 }

        for action in ["reopen", "newRound", "cancel"] {
            let f = Fixture()
            defer { f.cancel() }
            try await f.begin()
            f.finish()
            try await wait { f.result.opacity > 0 && f.result.opacity < 1 }
            if action == "reopen" { f.result.show() }
            else if action == "newRound" { try await f.begin() }
            else { f.cancel() }
            try await Task.sleep(nanoseconds: 250_000_000)
            precondition(f.result.opacity == (action == "cancel" ? 0 : 1),
                         "A stale fade must not override an explicit reopen or a new round")
            if action == "newRound" {
                f.finish()
                try await wait { f.result.opacity == 0 }
            }
        }

        let imported = Fixture()
        defer { imported.cancel() }
        try await imported.begin(withTarget: false)
        imported.result.show()
        imported.finish()
        try await wait { imported.result.opacity == 0 }
        precondition(imported.session.rawText == "保留结果文字", "An unpasted result must remain available")

        let failure = Fixture(reduceMotion: true)
        defer { failure.cancel() }
        failure.result.show()
        var values: [Double] = []
        let observer = failure.result.$opacity.sink { values.append($0) }
        failure.session.reportInputFailure("测试错误")
        try await wait { failure.result.opacity == 0 }
        precondition(!values.contains { $0 > 0 && $0 < 1 }, "Reduce Motion must skip animation")
        precondition(failure.session.notice == "测试错误", "Hiding must preserve the error")
        failure.result.show()
        try await Task.sleep(nanoseconds: 250_000_000)
        precondition(failure.result.opacity == 1, "Reopened errors must remain available to read")
        observer.cancel()
        print("PASS: result-window fade, independent hover, manual reopening, delivery, cancellation, new rounds and Reduce Motion")
    }
}
