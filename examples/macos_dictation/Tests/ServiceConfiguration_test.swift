import Foundation

private func check(_ value: @autoclosure () -> Bool, _ message: String) throws {
    if !value() { throw DictationError(message) }
}

private final class Requests: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [(URL, String)] = []
    func append(_ request: URLRequest) {
        var body = request.httpBody ?? Data()
        if body.isEmpty, let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                body.append(contentsOf: buffer.prefix(count))
            }
        }
        lock.lock()
        values.append((request.url!, String(decoding: body, as: UTF8.self)))
        lock.unlock()
    }
    var all: [(URL, String)] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

private final class MockHTTP: URLProtocol {
    static let requests = Requests()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.requests.append(request)
        let body: String
        switch request.url!.path {
        case "/v1/audio/transcriptions": body = #"{"text":"可以换快捷键吗？"}"#
        case "/api/chat": body = #"{"done":true,"done_reason":"stop","message":{"content":"可以换快捷键吗？"}}"#
        case "/health": body = #"{"status":"healthy"}"#
        case "/v1/models": body = #"{"data":[{"id":"asr-b"}]}"#
        case "/api/tags": body = #"{"models":[{"name":"polish-b"}]}"#
        default: body = "{}"
        }
        client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: 200,
                                                            httpVersion: nil, headerFields: nil)!, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@MainActor
private final class Recorder: AudioRecording {
    var gate: CheckedContinuation<Void, Never>?
    func start() async throws { await withCheckedContinuation { gate = $0 } }
    func stop() throws -> Data { Data([1]) }
    func cancel() { gate?.resume(); gate = nil }
}

@main
private enum ServiceConfigurationTests {
    @MainActor
    static func wait(_ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(3)
        while !predicate(), Date() < deadline { try await Task.sleep(nanoseconds: 1_000_000) }
        try check(predicate(), "等待状态超时")
    }

    @MainActor
    static func main() async {
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        for address in ["https://example.com", "http://127.0.0.1.evil.test", "http://user@localhost",
                        "http://localhost:0", "http://localhost:65536", "http://localhost/api/chat",
                        "http://localhost?redirect=example.com", "file:///tmp/server"] {
            do { _ = try LocalModelConfiguration(baseURL: address, model: "model") }
            catch { continue }
            throw DictationError("应拒绝非本机 origin：\(address)")
        }
        let a = ServiceConfiguration(
            asr: try LocalModelConfiguration(baseURL: "http://127.0.0.1:9001/", model: "asr-a"),
            polish: try LocalModelConfiguration(baseURL: "http://localhost:9002", model: "polish-a"))
        let b = ServiceConfiguration(
            asr: try LocalModelConfiguration(baseURL: "http://127.0.0.1:9101", model: "asr-b"),
            polish: try LocalModelConfiguration(baseURL: "http://[::1]:9102", model: "polish-b"))
        let domain = "local.omni.configuration-test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: domain)!
        defer { defaults.removePersistentDomain(forName: domain) }
        let preferences = ClientPreferences(defaults: defaults)
        preferences.serviceConfiguration = b
        try check(ClientPreferences(defaults: UserDefaults(suiteName: domain)!).serviceConfiguration == b,
                  "模型与地址应独立保存并通过验证后恢复")
        defaults.set(Data(#"{"asr":{"baseURL":"https://example.com","model":"x"},"polish":{"baseURL":"http://localhost","model":"y"}}"#.utf8), forKey: "serviceConfiguration")
        try check(preferences.serviceConfiguration == ServiceConfiguration(), "损坏或远程配置不得用于发送资料")

        let transport = LocalHTTPTransport(protocolClasses: [MockHTTP.self])
        let service = LocalSpeechService(configuration: a, transport: transport)
        let recorder = Recorder()
        let session = DictationSession(recorder: recorder, service: service)
        defer { session.cancel() }
        session.polishEnabled = true
        session.toggleRecording()
        try await wait { recorder.gate != nil }
        service.configure(b)
        recorder.gate?.resume()
        recorder.gate = nil
        try await wait { session.phase == .recording }
        session.toggleRecording()
        try await wait { session.phase == .ready }
        let first = MockHTTP.requests.all
        try check(first.count == 2 && first[0].0.port == 9001 && first[0].1.contains("asr-a")
                  && first[1].0.port == 9002 && first[1].1.contains("polish-a"),
                  "录音中切换配置，本轮 ASR 与 LLM 必须继续使用开始时的配置")
        session.submitAudio(Data([1]))
        try await wait { session.phase == .ready }
        let next = Array(MockHTTP.requests.all.suffix(2))
        try check(next[0].0.port == 9101 && next[0].1.contains("asr-b")
                  && next[1].0.port == 9102 && next[1].1.contains("polish-b"), "下一轮应使用新的独立配置")
        let health = await service.health()
        try check(health.asr == "可连接" && health.ollama.contains("模型已安装"), "健康检查应使用相同配置")
        let probes = MockHTTP.requests.all.suffix(3).map(\.0)
        try check(probes.contains(b.asr.endpoint("health")) && probes.contains(b.asr.endpoint("v1/models"))
                  && probes.contains(b.polish.endpoint("api/tags")),
                  "健康检查不能使用写死的默认地址")

        var calls: [PolishWarmup.Request] = []
        var gates: [CheckedContinuation<Void, Never>] = []
        let warmup = PolishWarmup.configured { request in
            calls.append(request)
            await withCheckedContinuation { gates.append($0) }
        }
        defer { warmup.stop() }
        warmup.update(enabled: true, personalBackground: "固定资料", busy: false, model: a.polish)
        try await wait { gates.count == 1 }
        warmup.update(enabled: true, personalBackground: "固定资料", busy: false, model: b.polish)
        try await wait { gates.count == 2 }
        gates[0].resume()
        try await Task.sleep(nanoseconds: 10_000_000)
        try check(warmup.status.contains("正在"), "旧模型的迟到预热不能覆盖新状态")
        gates[1].resume()
        try await wait { warmup.status.contains("已完成") }
        let moved = try LocalModelConfiguration(baseURL: "http://localhost:9202", model: b.polish.model)
        warmup.update(enabled: true, personalBackground: "固定资料", busy: true, model: moved)
        try check(calls.count == 2, "听写中不能启动新配置预热")
        warmup.update(enabled: true, personalBackground: "固定资料", busy: false, model: moved)
        try await wait { gates.count == 3 }
        try check(calls.map(\.model) == [a.polish, b.polish, moved], "模型或地址变化必须重新预热")
        gates[2].resume()
        try await wait { warmup.status.contains("已完成") }
        print("PASS: validated local configuration, persistence, per-round snapshots, shared health endpoints and model-aware warmup")
    }
}
