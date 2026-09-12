import Foundation
import Network

private func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw DictationError(message) }
}

/// A disposable loopback HTTP fixture exercises URLSession's real redirect delegate.
/// Synthetic GET requests only; no model ports, external hosts or user data.
private final class RedirectServer {
    private let queue = DispatchQueue(label: "local.omni.dictation.redirect-test")
    private let listener: NWListener
    private var connections: [NWConnection] = []
    private var origin: URL?
    private var failure: String?
    private var paths: [String] = []
    var url: URL? { queue.sync { origin } }
    var error: String? { queue.sync { failure } }
    var requests: [String] { queue.sync { paths } }

    init() throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: .any)
        listener = try NWListener(using: parameters)
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                if let port = self.listener.port { self.origin = URL(string: "http://127.0.0.1:\(port.rawValue)") }
            case .failed(let error), .waiting(let error): self.failure = error.localizedDescription
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { connection.cancel(); return }
            self.connections.append(connection)
            connection.start(queue: self.queue)
            self.receive(connection, data: Data())
        }
        listener.start(queue: queue)
    }

    func stop() {
        queue.sync {
            listener.cancel()
            connections.forEach { $0.cancel() }
            connections = []
        }
    }

    private func receive(_ connection: NWConnection, data: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { [weak self] chunk, _, ended, error in
            guard let self, error == nil else { connection.cancel(); return }
            var data = data
            if let chunk { data.append(chunk) }
            guard data.range(of: Data("\r\n\r\n".utf8)) != nil else {
                if ended || data.count > 16_384 { connection.cancel() }
                else { self.receive(connection, data: data) }
                return
            }
            let line = String(decoding: data, as: UTF8.self).components(separatedBy: "\r\n")[0]
            let parts = line.split(separator: " ")
            guard parts.count >= 2, let origin = self.origin else { connection.cancel(); return }
            let path = String(parts[1])
            self.paths.append(path)
            let response: String
            if path == "/redirect-302" || path == "/redirect-307" {
                let code = path.hasSuffix("302") ? 302 : 307
                response = "HTTP/1.1 \(code) Redirect\r\nLocation: \(origin.absoluteString)/destination\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            } else {
                response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK"
            }
            connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}

@MainActor
private final class Recorder: AudioRecording {
    func start() async throws {}
    func stop() throws -> Data { Data([1]) }
    func cancel() {}
}

@main
private enum HTTPTransportTests {
    @MainActor
    static func wait(_ condition: () -> Bool) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 5
        while !condition(), ProcessInfo.processInfo.systemUptime < deadline {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        try check(condition(), "等待 HTTP 测试状态超时")
    }

    @MainActor
    static func expectFailure(_ message: String, operation: () async throws -> Void) async throws {
        do { try await operation() }
        catch let error as DictationError {
            try check(error.message.contains(message), "错误信息缺少 \(message)：\(error.message)")
            return
        }
        throw DictationError("应失败但成功了：\(message)")
    }

    @MainActor
    static func main() async throws {
        setbuf(stdout, nil)
        print("Checking HTTP response errors and cancellation")
        try await transportErrors()
        print("Checking pipeline fallback")
        try await pipelineFallback()
        print("Checking redirects on a disposable loopback server")
        try await redirects()
        print("PASS: HTTP status/response errors, timeout/connection failures, task cancellation, raw fallback and real redirect refusal")
    }

    @MainActor
    static func transportErrors() async throws {
        HTTPStub.reset(["/probe": .http(200, "ok")])
        let transport = LocalHTTPTransport(protocolClasses: [HTTPStub.self])
        let request = URLRequest(url: URL(string: "http://127.0.0.1/probe")!)
        let data = try await transport.send(request, stage: "测试 ASR")
        try check(data == Data("ok".utf8), "成功响应必须原样交给上层解码")
        for status in [400, 503] {
            HTTPStub.set(.http(status, "private backend error body"), for: "/probe")
            try await expectFailure("测试 ASR 返回 HTTP \(status)") {
                _ = try await transport.send(request, stage: "测试 ASR")
            }
        }
        for (reply, message) in [
            (HTTPStub.Reply.failure(.timedOut), "测试 ASR 请求超时"),
            (.failure(.cannotConnectToHost), "无法连接 测试 ASR"),
            (.nonHTTP, "测试 ASR 没有返回 HTTP 响应"),
        ] {
            HTTPStub.set(reply, for: "/probe")
            try await expectFailure(message) { _ = try await transport.send(request, stage: "测试 ASR") }
        }
        HTTPStub.set(.failure(.cancelled), for: "/probe")
        do {
            _ = try await transport.send(request, stage: "测试 ASR")
            throw DictationError("URL cancellation 必须转换为 CancellationError")
        } catch is CancellationError {}

        HTTPStub.set(.held, for: "/probe")
        let task = Task { try await transport.send(request, stage: "测试 ASR") }
        defer { task.cancel() }
        try await wait { HTTPStub.heldCount == 1 }
        task.cancel()
        do {
            _ = try await task.value
            throw DictationError("取消 Task 后不能返回成功结果")
        } catch is CancellationError {}
        try await wait { HTTPStub.cancelledCount == 1 }
        HTTPStub.releaseHeld(.http(200, "late response"))
        try await wait { HTTPStub.heldCount == 0 }
    }

    @MainActor
    static func pipelineFallback() async throws {
        let raw = "这个 PR 不要合并。"
        for reply in [HTTPStub.Reply.http(503, "unavailable"), .failure(.timedOut), .http(200, "{}"),
                      .http(200, #"{"done":true,"done_reason":"stop","message":{"content":""}}"#)] {
            HTTPStub.reset([
                "/v1/audio/transcriptions": .http(200, "{\"text\":\"\(raw)\"}"),
                "/api/chat": reply,
            ])
            let service = LocalSpeechService(transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self]))
            let session = DictationSession(recorder: Recorder(), service: service)
            defer { session.cancel() }
            session.polishEnabled = true
            var results: [String] = []
            session.onResult = { results.append($0) }
            session.submitAudio(Data([1]))
            try await wait { session.phase == .ready || session.phase == .failed }
            try check(session.phase == .ready && session.rawText == raw && session.resultText == raw
                      && !session.hasPolishedResult && session.notice.contains("已保留原文") && results == [raw],
                      "真实 HTTP/解码失败必须只交付一次原文，不能丢失或声称整理成功")
            try check(HTTPStub.requests.map(\.url.path) == ["/v1/audio/transcriptions", "/api/chat"],
                      "回退必须来自实际 ASR → Ollama 请求链")
        }
        HTTPStub.reset(["/v1/audio/transcriptions": .http(200, "{broken json")])
        let service = LocalSpeechService(transport: LocalHTTPTransport(protocolClasses: [HTTPStub.self]))
        try await expectFailure("Omni ASR 返回格式异常") { _ = try await service.transcribe(wav: Data([1])) }
    }

    @MainActor
    static func redirects() async throws {
        let server = try RedirectServer()
        defer { server.stop() }
        try await wait { server.url != nil || server.error != nil }
        if let error = server.error { throw DictationError("无法启动回环 HTTP 测试服务：\(error)") }
        let origin = server.url!
        let transport = LocalHTTPTransport()
        for code in [302, 307] {
            var request = URLRequest(url: origin.appendingPathComponent("redirect-\(code)"))
            request.timeoutInterval = 5
            try await expectFailure("HTTP \(code)") { _ = try await transport.send(request, stage: "重定向测试") }
        }
        try check(server.requests == ["/redirect-302", "/redirect-307"],
                  "拒绝重定向必须阻止发送第二个请求，即使目的地仍在本机")
        // Positive control: the destination is reachable, so rejection is not a network failure.
        var request = URLRequest(url: origin.appendingPathComponent("destination"))
        request.timeoutInterval = 5
        let data = try await transport.send(request, stage: "回环测试")
        try check(data == Data("OK".utf8) && server.requests.last == "/destination", "回环目标应可直接访问")
    }
}
