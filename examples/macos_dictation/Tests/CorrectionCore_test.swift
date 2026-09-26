import Foundation

@MainActor
private final class Recorder: AudioRecording {
    var failsStart = false
    var failsStop = false
    func start() async throws { if failsStart { throw DictationError("start failed") } }
    func stop() throws -> Data {
        if failsStop { throw DictationError("stop failed") }
        return Data([1])
    }
    func cancel() { }
}

@MainActor
private final class Service: SpeechServing {
    var reply = "把张三改成张珊，珊瑚的珊"
    var fails = false
    func transcribe(wav: Data) async throws -> String {
        if fails { throw DictationError("ASR failed") }
        return reply
    }
    func polish(text: String) async throws -> String { fatalError("Correction must not use polishing") }
}

@MainActor
private final class Corrector: TextCorrecting {
    var result = CorrectionResult(status: .ok, correctedText: "明天和张珊开会。")
    var hold = false
    var reply: CheckedContinuation<CorrectionResult, Error>?
    var inputs: [String] = []
    var backgrounds: [String] = []
    var fails = false
    func correct(original: String, instruction: String, personalBackground: String) async throws -> CorrectionResult {
        inputs.append(original)
        backgrounds.append(personalBackground)
        if fails { throw DictationError("LLM failed") }
        if hold { return try await withCheckedThrowingContinuation { reply = $0 } }
        return result
    }
}

@MainActor
private final class Target: TextRevisionTarget {
    var confirmed = true
    var valid = true
    var writes = 0
    var applied: TextRevision?
    var throwsAfterWrite = false
    func confirmInsertion() async -> Bool { confirmed }
    func prepare(original: String) throws { if !valid { throw DictationError("Changed editor") } }
    func apply(_ revision: TextRevision) async throws -> Bool {
        guard valid else { throw DictationError("Changed editor") }
        writes += 1
        if throwsAfterWrite { throw DictationError("No acknowledgement") }
        applied = revision
        return confirmed
    }
    func cancel() { }
}

@MainActor
private func wait(_ predicate: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 3
    while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    precondition(predicate(), "Timeout")
}

@main
private enum CorrectionCoreTests {
    @MainActor
    static func main() async throws {
        for (old, new) in [("明天和张三开会。", "明天和张珊开会。"), ("a👨‍👩‍👧‍👦b", "a👩🏽‍💻b"),
                           ("Cafe\u{301}", "Café"), ("第一句。\n第二句。", "第一句。"),
                           ("abc", ""), ("abc", "abc!"), ("abc", "abc")] {
            let edit = TextRevision(original: old, corrected: new)
            let patched = try TextDraft(value: old, selection: edit.range).inserting(edit.replacement)
            precondition(patched.utf16.elementsEqual(new.utf16), "Unicode patch must preserve exact bytes")
        }
        let patch = TextRevision(original: "明天和张三开会。", corrected: "明天和张珊开会。")
        precondition(patch.replacement == "珊" && patch.range.length == 1)

        var gesture = DoubleOptionGesture()
        func tap(_ time: Double, key: UInt16 = 58) -> Bool {
            precondition(!gesture.option(key: key, down: true, otherModifiers: false, time: time))
            return gesture.option(key: key, down: false, otherModifiers: false, time: time + 0.05)
        }
        precondition(!tap(0) && tap(0.2))
        precondition(!tap(0.4), "Third tap must not retrigger")
        gesture.reset()
        precondition(!tap(1))
        gesture.reset() // Option+Space, mouse input, focus or permission change.
        precondition(!tap(1.2))
        gesture.reset()
        precondition(!tap(2, key: 61) && tap(2.2, key: 61))
        gesture.reset()
        precondition(!tap(3) && !tap(3.2, key: 61), "Mixed sides must not trigger")
        gesture.reset()
        precondition(!gesture.option(key: 58, down: true, otherModifiers: false, time: 4))
        precondition(!gesture.option(key: 58, down: false, otherModifiers: false, time: 4.5))
        precondition(!tap(4.6), "Long hold must not count")

        let service = Service(), corrector = Corrector(), target = Target()
        let session = CorrectionSession(recorder: Recorder(), service: service, makeCorrector: { corrector })
        session.remember("明天和张三开会。")
        session.trackInsertion(target)
        try await wait { session.canReplace }
        session.toggleRecording(personalBackground: "术语")
        try await wait { session.phase == .recording }
        session.toggleRecording(personalBackground: "")
        try await wait { session.phase == .ready }
        precondition(target.writes == 1 && target.applied?.replacement == "珊")
        precondition(session.lastText == "明天和张珊开会。")
        precondition(session.recording.polishEnabled == false)
        precondition(corrector.backgrounds == ["术语"], "Stop-time preferences must not replace the captured background")

        corrector.result = CorrectionResult(status: .ok, correctedText: "今天和张珊开会。")
        target.valid = false
        session.toggleRecording(personalBackground: "")
        try await wait { session.phase == .recording }
        session.toggleRecording(personalBackground: "")
        try await wait { session.phase == .ready }
        precondition(target.writes == 1 && session.lastText == "明天和张珊开会。")
        precondition(session.correctedText == "今天和张珊开会。", "Copy-only output must remain available")

        corrector.hold = true
        session.toggleRecording(personalBackground: "")
        try await wait { session.phase == .recording }
        session.toggleRecording(personalBackground: "")
        try await wait { corrector.reply != nil }
        session.clear()
        corrector.reply?.resume(returning: corrector.result)
        corrector.reply = nil
        try await Task.sleep(nanoseconds: 20_000_000)
        precondition(session.lastText == nil && session.correctedText == nil && target.writes == 1,
                     "Late correction must not resurrect a cleared round")

        for scenario in ["noChange", "unchangedOK", "asrError", "emptyASR", "llmError", "startError", "stopError", "unconfirmed", "writeError", "deletion"] {
            let recorder = Recorder(), service = Service(), corrector = Corrector(), target = Target()
            let session = CorrectionSession(recorder: recorder, service: service, makeCorrector: { corrector })
            defer { session.clear() }
            let original = "明天和张三开会。"
            session.remember(original)
            session.trackInsertion(target)
            try await wait { session.canReplace }
            recorder.failsStart = scenario == "startError"
            recorder.failsStop = scenario == "stopError"
            service.fails = scenario == "asrError"
            if scenario == "emptyASR" { service.reply = "" }
            corrector.fails = scenario == "llmError"
            target.confirmed = scenario != "unconfirmed"
            target.throwsAfterWrite = scenario == "writeError"
            if scenario == "noChange" { corrector.result = CorrectionResult(status: .noChange, correctedText: original) }
            if scenario == "unchangedOK" { corrector.result = CorrectionResult(status: .ok, correctedText: original) }
            if scenario == "deletion" { corrector.result = CorrectionResult(status: .ok, correctedText: "") }
            session.toggleRecording(personalBackground: "")
            try await wait { session.phase == .recording || session.phase == .failed }
            if session.phase == .recording { session.toggleRecording(personalBackground: "") }
            try await wait { !session.isBusy }
            if scenario == "deletion" {
                precondition(session.lastText == "" && session.didReplace && !session.canReplace)
            } else {
                precondition(session.lastText == original && !session.didReplace, "Failure/no-change must preserve committed text: \(scenario)")
            }
            if ["noChange", "unchangedOK", "asrError", "emptyASR", "llmError", "startError", "stopError"].contains(scenario) {
                precondition(target.writes == 0, "Must not write for \(scenario)")
            } else { precondition(target.writes == 1) }
        }

        let missing = CorrectionSession(recorder: Recorder(), service: Service(), makeCorrector: { Corrector() })
        missing.toggleRecording(personalBackground: "")
        precondition(missing.phase == .failed && missing.lastText == nil)
        missing.clear()

        let unavailable = CorrectionSession(recorder: Recorder(), service: Service(), makeCorrector: { Corrector() })
        unavailable.remember("明天和张三开会。")
        unavailable.noteUnavailableTarget("输入框不支持选区")
        unavailable.toggleRecording(personalBackground: "")
        try await wait { unavailable.phase == .recording }
        unavailable.toggleRecording(personalBackground: "")
        try await wait { unavailable.phase == .ready }
        precondition(!unavailable.didReplace && unavailable.notice.contains("输入框不支持选区"),
                     "Capture failures must be visible instead of silently looking like a successful correction")
        unavailable.clear()

        let request = try OllamaCorrector.request(original: "原文", instruction: "改意见", personalBackground: "背景")
        let body = try JSONSerialization.jsonObject(with: request.httpBody!) as! [String: Any]
        precondition(body["format"] is [String: Any] && body["stream"] as? Bool == false)
        func response(_ text: String, reason: String = "stop") throws -> Data {
            let content = try JSONSerialization.data(withJSONObject: ["scope": "local", "text": text])
            return try JSONSerialization.data(withJSONObject: ["done": true, "done_reason": reason,
                "message": ["content": String(decoding: content, as: UTF8.self)]])
        }
        let deletion = try await OllamaCorrector(transport: LocalHTTPTransport())
            .correct(original: "删掉", instruction: "删除全部", personalBackground: "")
        precondition(deletion.correctedText == "")
        do {
            _ = try OllamaCorrector.decode(response("partial", reason: "length"), original: "原文", instruction: "修改")
            preconditionFailure("Truncated output must fail")
        } catch { }
        print("PASS: Unicode patches, double Option, correction delivery, copy fallback, cancellation and JSON validation")
    }
}
