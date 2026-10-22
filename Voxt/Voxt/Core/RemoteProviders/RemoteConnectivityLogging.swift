// RemoteConnectivityLogging.swift
// Provides Remote Connectivity Logging for remote provider configuration.

import Foundation

enum RemoteProviderConnectivityTestLogging {
    static func sanitizedEndpointForLog(_ endpoint: String) -> String {
        RemoteEndpointSecurityPolicy.sanitizedForLog(endpoint)
    }

    static func logHTTPRequest(context: String, request: URLRequest, bodyPreview: String) {
        let method = request.httpMethod ?? "GET"
        let url = redactedURLString(request.url)
        let headers = redactedHeaders(request.allHTTPHeaderFields ?? [:])
        VoxtLog.network(
            "Network test request. context=\(context), method=\(method), url=\(url), headers=\(headers), body=\(truncateLogText(bodyPreview, limit: 700))",
            verbose: true
        )
    }

    static func logHTTPResponse(context: String, response: HTTPURLResponse, data: Data) {
        let url = redactedURLString(response.url)
        let headers = redactedHeaders(response.allHeaderFields.reduce(into: [String: String]()) { partialResult, pair in
            partialResult[String(describing: pair.key)] = String(describing: pair.value)
        })
        let payload = String(data: data, encoding: .utf8) ?? "<non-utf8 \(data.count) bytes>"
        VoxtLog.network(
            "Network test response. context=\(context), status=\(response.statusCode), url=\(url), headers=\(headers), body=\(truncateLogText(payload, limit: 700))",
            verbose: true
        )
    }

    private static func redactedHeaders(_ headers: [String: String]) -> String {
        let redacted = sanitizedHeadersForLog(headers)
        if let data = try? JSONSerialization.data(withJSONObject: redacted, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            return text
        }
        return "\(redacted)"
    }

    static func sanitizedHeadersForLog(_ headers: [String: String]) -> [String: String] {
        VoxtLogRedactor.redactedHTTPHeaders(headers)
    }

    private static func redactedURLString(_ url: URL?) -> String {
        guard let url else { return "<nil>" }
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
            return url.absoluteString
        }
        components.queryItems = components.queryItems?.map {
            URLQueryItem(name: $0.name, value: "<redacted>")
        }
        components.user = nil
        components.password = nil
        components.fragment = nil
        return components.string ?? url.absoluteString
    }

    private static func truncateLogText(_ text: String, limit: Int) -> String {
        if text.count <= limit { return text }
        return String(text.prefix(limit)) + "...(truncated)"
    }
}
