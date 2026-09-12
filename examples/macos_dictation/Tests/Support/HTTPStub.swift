import Foundation

/// All requests are intercepted. Unconfigured paths fail instead of contacting a server.
/// State and URLProtocol callbacks share one queue, including cancellation and held replies.
final class HTTPStub: URLProtocol {
    enum Reply {
        case http(Int, String)
        case failure(URLError.Code)
        case nonHTTP
        case held
    }
    struct Request {
        let url: URL
        let body: String
    }
    private static let queue = DispatchQueue(label: "local.omni.dictation.http-tests")
    private static var routes: [String: Reply] = [:]
    private static var recorded: [Request] = []
    private static var pending: [HTTPStub] = []
    private static var cancellations = 0
    private var stopped = false

    static func reset(_ replies: [String: Reply]) {
        queue.sync {
            routes = replies
            recorded = []
            pending = []
            cancellations = 0
        }
    }
    static func set(_ reply: Reply, for path: String) { queue.sync { routes[path] = reply } }
    static var requests: [Request] { queue.sync { recorded } }
    static var heldCount: Int { queue.sync { pending.count } }
    static var cancelledCount: Int { queue.sync { cancellations } }
    static func releaseHeld(_ reply: Reply) {
        queue.async {
            let held = pending
            pending = []
            held.forEach { $0.deliver(reply) }
        }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.queue.async {
            guard !self.stopped else { return }
            Self.recorded.append(Request(url: self.request.url!, body: Self.body(self.request)))
            self.deliver(Self.routes[self.request.url!.path] ?? .failure(.resourceUnavailable))
        }
    }
    override func stopLoading() {
        Self.queue.async {
            self.stopped = true
            if Self.pending.contains(where: { $0 === self }) { Self.cancellations += 1 }
        }
    }

    private func deliver(_ reply: Reply) {
        guard !stopped else { return }
        switch reply {
        case .http(let code, let body):
            let response = HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: "HTTP/1.1",
                                           headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        case .failure(let code):
            client?.urlProtocol(self, didFailWithError: URLError(code))
        case .nonHTTP:
            let response = URLResponse(url: request.url!, mimeType: nil, expectedContentLength: 0, textEncodingName: nil)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
        case .held:
            Self.pending.append(self)
        }
    }

    private static func body(_ request: URLRequest) -> String {
        if let data = request.httpBody { return String(decoding: data, as: UTF8.self) }
        guard let stream = request.httpBodyStream else { return "" }
        stream.open()
        defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(contentsOf: buffer.prefix(count))
        }
        return String(decoding: data, as: UTF8.self)
    }
}
