import Foundation

@MainActor
private final class Recorder: AudioRecording {
    var starts = 0
    func start() async throws { starts += 1 }
    func stop() throws -> Data { Data([1]) }
    func cancel() { }
}

@MainActor
private final class Target: DictationTextTarget, TextRevisionTarget {
    let applicationName = "editor"
    let supportsConfirmation = false
    var value = ""
    var writes = 0
    func validate() throws { }
    func insert(_ text: String) throws { value = text; writes += 1 }
    func confirms(_ text: String) -> Bool { false }
    func stopObserving() { }
    func confirmInsertion() async -> Bool { !value.isEmpty }
    func prepare(original: String) throws { precondition(value == original) }
    func apply(_ revision: TextRevision) async throws -> Bool { value = revision.corrected; writes += 1; return true }
    func cancel() { }
}

@MainActor
private func wait(_ predicate: () -> Bool) async throws {
    let deadline = ProcessInfo.processInfo.systemUptime + 5
    while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
        try await Task.sleep(nanoseconds: 1_000_000)
    }
    precondition(predicate(), "Timed out")
}

@main
private enum CorrectionIntegrationTests {
    @MainActor
    static func main() async throws {
        HTTPStub.reset(["/v1/audio/transcriptions": .http(200, #"{"text":"明天和张三开会。"}"#)])
        let domain = "omni-correction-composition-\(UUID())"
        let defaults = UserDefaults(suiteName: domain)!
        defer { defaults.removePersistentDomain(forName: domain) }
        let recorder = Recorder(), target = Target()
        let state = ClientState(preferences: ClientPreferences(defaults: defaults),
                                service: LocalSpeechService(transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self])),
                                recorder: recorder, makeTextTarget: { _, _ in target })
        state.captureRevisionTarget = { _ in target }
        defer { state.cancel(); state.warmup.stop() }
        precondition(!state.session.polishEnabled)
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        state.toggleRecording()
        try await wait { state.correction.canReplace }
        precondition(state.correction.lastText == "明天和张三开会。" && target.writes == 1)
        HTTPStub.set(.http(200, #"{"text":"人名最后一个字是珊瑚的珊。"}"#), for: "/v1/audio/transcriptions")
        let payload = try JSONSerialization.data(withJSONObject: ["scope": "local", "text": "明天和张珊开会。"])
        let reply = String(decoding: payload, as: UTF8.self)
        let bytes = try JSONSerialization.data(withJSONObject: ["done": true, "done_reason": "stop", "message": ["content": reply]])
        HTTPStub.set(.http(200, String(decoding: bytes, as: UTF8.self)), for: "/api/chat")
        state.toggleCorrection()
        try await wait { state.correction.phase == .recording }
        state.toggleRecording() // Ordinary shortcut must not start another microphone session.
        precondition(recorder.starts == 2 && state.isBusy)
        state.toggleCorrection()
        try await wait { state.correction.phase == .ready }
        precondition(target.value == "明天和张珊开会。" && target.writes == 2)
        precondition(state.session.rawText == "明天和张三开会。", "Correction must preserve the ASR original")
        precondition(HTTPStub.requests.filter { $0.url.path == "/api/chat" }.count == 1,
                     "Polishing off must still allow one explicit correction request")
        state.toggleRecording()
        try await wait { state.session.phase == .recording }
        precondition(state.correction.lastText == nil, "A new ordinary round must clear the old anchor")
        state.cancel()
        print("PASS: dictation-to-correction composition, shared microphone exclusion, polishing independence and new-round reset")
    }
}
