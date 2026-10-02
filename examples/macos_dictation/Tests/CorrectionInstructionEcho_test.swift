import Foundation

@MainActor
private final class Recorder: AudioRecording {
    func start() async throws { }
    func stop() throws -> Data { Data([1]) }
    func cancel() { }
}

@MainActor
private final class Speech: SpeechServing {
    let instruction: String
    init(_ instruction: String) { self.instruction = instruction }
    func transcribe(wav: Data) async throws -> String { instruction }
    func polish(text: String) async throws -> String { fatalError("Correction must not polish its instruction") }
}

@MainActor
private final class DraftTarget: TextRevisionTarget {
    var value: String
    var writes = 0
    init(_ value: String) { self.value = value }
    func confirmInsertion() async -> Bool { true }
    func prepare(original: String) throws { precondition(value == original) }
    func apply(_ revision: TextRevision) async throws -> Bool {
        value = try TextDraft(value: value, selection: revision.range).inserting(revision.replacement)
        writes += 1
        return true
    }
    func cancel() { }
}

@main
private enum CorrectionInstructionEchoTests {
    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        guard predicate() else { throw DictationError("Timed out") }
    }

    @MainActor
    static func check(original: String, instruction: String, response: String, expected: String?) async throws {
        let content = try JSONSerialization.data(withJSONObject: ["scope": "local", "text": response])
        let bytes = try JSONSerialization.data(withJSONObject: ["done": true, "done_reason": "stop",
            "message": ["content": String(decoding: content, as: UTF8.self)]])
        HTTPStub.reset(["/api/chat": .http(200, String(decoding: bytes, as: UTF8.self))])
        let corrector = OllamaCorrector(transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self]))
        let target = DraftTarget(original)
        let session = CorrectionSession(recorder: Recorder(), service: Speech(instruction), makeCorrector: { corrector })
        defer { session.clear() }
        session.remember(original)
        session.trackInsertion(target)
        try await wait { session.canReplace }
        session.toggleRecording(personalBackground: "项目 SGLang-Omni")
        try await wait { session.phase == .recording }
        session.toggleRecording(personalBackground: "")
        try await wait { session.phase == .ready || session.phase == .failed }
        if let expected {
            guard session.didReplace, target.value == expected, target.writes == 1 else {
                throw DictationError("Expected only revised original; got: \(target.value)")
            }
        } else {
            guard session.phase == .failed, session.correctedText == nil,
                  session.notice.contains("修改意见"), target.value == original, target.writes == 0 else {
                throw DictationError("An echoed instruction reached the editor: \(target.value)")
            }
        }
    }

    @MainActor
    static func main() async throws {
        let original = "大家好，我们是S G浪组合。"
        let instruction = "S.G. Lang的Lang是L.A.N.G."
        let expected = "大家好，我们是SGLang组合。"
        // An ordinary edit must reject both a verbatim and a reformatted instruction echo.
        for response in ["明天下午两点见。\n\n时间往后推一个小时。", "明天下午三点见。修改意见：时间 往后 推一个小时"] {
            try await check(original: "明天下午两点见。", instruction: "时间往后推一个小时。", response: response, expected: nil)
        }
        // The exact spelling directive independently identifies a unique original span.
        try await check(original: original, instruction: instruction,
                        response: original + "\n\nS.G.Lang的Lang是L.A.N.G.", expected: expected)
        try await check(original: "明天下午两点见。", instruction: "把两点改成三点。",
                        response: "明天下午三点见。", expected: "明天下午三点见。")
        try await check(original: "明天下午两点见。", instruction: "时间往后推一个小时。",
                        response: "明天下午三点见。", expected: "明天下午三点见。")
        try await check(original: "备注：待定。", instruction: "把备注改成“把两点改成三点”。",
                        response: "备注：把两点改成三点。", expected: "备注：把两点改成三点。")
        print("PASS: echoed instructions never reach the draft; scoped spelling changes only the original word; legitimate edits still apply")
        if ProcessInfo.processInfo.environment["OMNI_CORRECTION_LIVE_TEST"] == "1" {
            let corrector = OllamaCorrector(transport: LocalHTTPTransport())
            let result = try await corrector.correct(original: original, instruction: instruction, personalBackground: "项目 SGLang-Omni")
            guard result.correctedText == expected else { throw DictationError("Live correction mismatch: \(result.correctedText)") }
            print("PASS: explicit spelling through the correction pipeline: \(result.correctedText)")
            print("Single functional case; no microphone or external editor access, model-only accuracy or latency claim.")
        }
    }
}
