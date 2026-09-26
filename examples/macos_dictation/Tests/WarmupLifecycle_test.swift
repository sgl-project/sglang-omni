import Foundation

private func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
}

@MainActor
private final class Recorder: AudioRecording {
    func start() async throws {}
    func stop() throws -> Data { Data([1]) }
    func cancel() {}
}

@main
private enum WarmupLifecycleTests {
    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while !predicate(), ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        try check(predicate(), "Timed out waiting for lifecycle state")
    }

    static func chatBodies() throws -> [[String: Any]] {
        try HTTPStub.requests.filter { $0.url.path == "/api/chat" }.map {
            try JSONSerialization.jsonObject(with: Data($0.body.utf8)) as! [String: Any]
        }
    }

    @MainActor
    static func scenario(_ name: String) async throws {
        let domain = "local.omni.warmup-lifecycle-test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: domain)!
        defer { defaults.removePersistentDomain(forName: domain) }
        let preferences = ClientPreferences(defaults: defaults)
        preferences.polishEnabled = true
        let configuration = ServiceConfiguration()
        let models = String(decoding: try JSONSerialization.data(withJSONObject: ["data": [["id": configuration.asr.model]]]), as: UTF8.self)
        let tags = String(decoding: try JSONSerialization.data(withJSONObject: ["models": [["name": configuration.polish.model]]]), as: UTF8.self)
        let warmReply = HTTPStub.Reply.http(200, #"{"done":true,"done_reason":"length"}"#)
        HTTPStub.reset([
            "/api/chat": .held,
            "/health": .http(200, #"{"status":"healthy"}"#),
            "/v1/models": .http(200, models),
            "/api/tags": .http(200, tags),
            "/v1/audio/transcriptions": .http(200, #"{"text":"保留这段原文。"}"#),
        ])
        let service = LocalSpeechService(configuration: configuration, transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self]))
        let state = ClientState(preferences: preferences, service: service, recorder: Recorder(),
                                makeTextTarget: { _, _ in throw DictationError("No real editor access in this fixture") })
        defer { state.cancel(); state.warmup.stop() }
        try await wait { HTTPStub.heldCount == 1 }
        let initial = try chatBodies()
        try check(initial.count == 1 && (initial[0]["options"] as? [String: Any])?["num_predict"] as? Int == 1,
                  "Saved enabled preference must start one warmup during initialization")
        state.checkServices(retryWarmup: false)
        try await wait { !state.checking }
        try check(HTTPStub.heldCount == 1 && HTTPStub.cancelledCount == 0,
                  "Startup service check must preserve the initial warmup")

        state.session.toggleRecording()
        try await wait { state.session.phase == .recording && HTTPStub.cancelledCount == 1 }
        HTTPStub.releaseHeld(warmReply)
        try await wait { HTTPStub.heldCount == 0 }
        HTTPStub.set(warmReply, for: "/api/chat")

        switch name {
        case "recording cancellation":
            state.cancel()
        case "ASR failure":
            HTTPStub.set(.failure(.cannotConnectToHost), for: "/v1/audio/transcriptions")
            state.session.toggleRecording()
            try await wait { state.session.phase == .failed }
        case "ASR cancellation":
            HTTPStub.set(.held, for: "/v1/audio/transcriptions")
            state.session.toggleRecording()
            try await wait { HTTPStub.heldCount == 1 }
            state.cancel()
            try await wait { HTTPStub.cancelledCount == 2 }
            HTTPStub.releaseHeld(.http(200, #"{"text":"迟到结果。"}"#))
            try await wait { HTTPStub.heldCount == 0 }
        case "foreground completion", "foreground cancellation", "changed model":
            HTTPStub.set(.held, for: "/api/chat")
            if name == "changed model" {
                state.polishModelDraft = "next-model"
                state.saveServiceConfiguration()
                try await wait { !state.checking }
            }
            state.session.toggleRecording()
            try await wait { state.session.phase == .polishing && HTTPStub.heldCount == 1 }
            HTTPStub.set(warmReply, for: "/api/chat")
            if name == "foreground cancellation" {
                state.cancel()
                try await wait { HTTPStub.cancelledCount == 2 }
            }
            HTTPStub.releaseHeld(.http(200, #"{"done":true,"done_reason":"stop","message":{"content":"保留这段原文。"}}"#))
            try await wait { HTTPStub.heldCount == 0 && !state.session.isBusy }
        default:
            throw DictationError("Unknown scenario")
        }

        let reachedPolish = name.hasPrefix("foreground") || name == "changed model"
        if !reachedPolish || name == "changed model" {
            try await wait { state.warmup.status.contains("已完成") }
        }
        // Let late canceled completions and all queued phase publications settle.
        try await Task.sleep(nanoseconds: 50_000_000)
        let bodies = try chatBodies()
        let warmups = bodies.filter { ($0["options"] as? [String: Any])?["num_predict"] as? Int == 1 }
        try check(warmups.count == (name.hasPrefix("foreground") ? 1 : 2),
                  "Interrupted warmup must retry unless foreground polishing took over: \(name)")
        try check(bodies.count - warmups.count == (reachedPolish ? 1 : 0), "Unexpected foreground request")
        if name == "changed model" {
            try check(warmups.last?["model"] as? String == "next-model", "New configuration must still warm on idle")
            try check(bodies[1]["model"] as? String == configuration.polish.model, "Current round must keep its captured model")
        }
        if name == "recording cancellation" {
            try check(!HTTPStub.requests.contains { $0.url.path == "/v1/audio/transcriptions" }, "Canceled recording must not reach ASR")
        }
        print("PASS: startup warmup preserved; \(name)")
    }

    @MainActor
    static func main() async throws {
        setbuf(stdout, nil)
        for name in ["recording cancellation", "ASR failure", "ASR cancellation",
                     "foreground completion", "foreground cancellation", "changed model"] {
            try await scenario(name)
        }
    }
}
