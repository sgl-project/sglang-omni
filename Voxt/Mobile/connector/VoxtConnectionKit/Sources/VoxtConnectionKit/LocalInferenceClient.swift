import Foundation

public struct LocalInferenceClient: Sendable {
    public struct Health: Decodable, Equatable, Sendable {
        public var ok: Bool
        public var app: String
        public var version: String
    }
    public enum Failure: Error, Equatable { case invalidEndpoint, unauthorized, unavailable, invalidResponse }
    private let baseURL: URL
    private let token: String
    private let session: URLSession

    public init(baseURL: URL, token: String, session: URLSession = .shared) throws {
        guard baseURL.scheme == "http", baseURL.host == "127.0.0.1",
              baseURL.user == nil, baseURL.password == nil,
              baseURL.path.isEmpty || baseURL.path == "/",
              baseURL.query == nil, baseURL.fragment == nil else {
            throw Failure.invalidEndpoint
        }
        self.baseURL = baseURL
        self.token = token
        self.session = session
    }

    public func health() async throws -> Health {
        var request = URLRequest(url: baseURL.appending(path: "v1/health"))
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 5
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request, delegate: NoRedirects())
        } catch {
            if error is CancellationError || (error as? URLError)?.code == .cancelled { throw error }
            throw Failure.unavailable
        }
        guard let http = response as? HTTPURLResponse else { throw Failure.invalidResponse }
        if http.statusCode == 401 || http.statusCode == 403 { throw Failure.unauthorized }
        if http.statusCode >= 500 { throw Failure.unavailable }
        guard http.statusCode == 200,
              let health = try? JSONDecoder().decode(Health.self, from: data),
              health.app == "Voxt", !health.version.isEmpty else { throw Failure.invalidResponse }
        guard health.ok else { throw Failure.unavailable }
        return health
    }
}

/// A local API redirect must never carry the credential to another endpoint.
private final class NoRedirects: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
