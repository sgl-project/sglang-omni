import Foundation

// Reject redirects even if a loopback server redirects to another local endpoint.
private final class NoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

@MainActor
public final class LocalHTTPTransport {
    private let session: URLSession

    /// Protocol classes are injectable for deterministic HTTP tests, without a running server.
    public init(protocolClasses: [AnyClass]? = nil) {
        let config = URLSessionConfiguration.ephemeral
        config.connectionProxyDictionary = [:]
        config.urlCache = nil
        config.httpCookieStorage = nil
        config.timeoutIntervalForRequest = 180
        config.timeoutIntervalForResource = 180
        if let protocolClasses { config.protocolClasses = protocolClasses }
        session = URLSession(configuration: config, delegate: NoRedirect(), delegateQueue: nil)
    }

    func send(_ request: URLRequest, stage: String) async throws -> Data {
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw DictationError("\(stage) 没有返回 HTTP 响应。") }
            guard (200..<300).contains(http.statusCode) else {
                throw DictationError("\(stage) 返回 HTTP \(http.statusCode)，请检查服务终端和模型配置。")
            }
            return data
        } catch let error as URLError {
            if error.code == .cancelled { throw CancellationError() }
            if error.code == .timedOut { throw DictationError("\(stage) 请求超时，请查看服务终端后重试。") }
            throw DictationError("无法连接 \(stage)，请确认对应的本地服务已启动。")
        }
    }
}
