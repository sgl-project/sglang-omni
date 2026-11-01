// LocalAPIHTTP.swift
// Provides the minimal HTTP/1.1 request parsing and response encoding used by the Local API server.

import Foundation

nonisolated struct LocalAPIHTTPRequest: Equatable, Sendable {
    var method: String
    var path: String
    var query: [String: String]
    /// Header names are stored lowercased.
    var headers: [String: String]
    var body: Data

    init(method: String, path: String, query: [String: String] = [:], headers: [String: String] = [:], body: Data = Data()) {
        self.method = method
        self.path = path
        self.query = query
        self.headers = Dictionary(headers.map { ($0.key.lowercased(), $0.value) }, uniquingKeysWith: { _, last in last })
        self.body = body
    }

    func header(_ name: String) -> String? {
        headers[name.lowercased()]
    }
}

nonisolated enum LocalAPIHTTPParseResult: Equatable, Sendable {
    case incomplete
    case complete(LocalAPIHTTPRequest, consumed: Int)
    /// The HTTP status to answer with before closing the connection.
    case failure(Int)
}

nonisolated enum LocalAPIHTTPParser {
    static let maxHeaderBytes = 16 * 1024
    private static let headerTerminator = Data("\r\n\r\n".utf8)

    /// Parses one request from the start of `data`. Only `Content-Length` bodies are accepted;
    /// clients that stream chunked bodies get 411.
    static func parse(_ data: Data, maxBodyBytes: Int) -> LocalAPIHTTPParseResult {
        guard let terminator = data.range(of: headerTerminator) else {
            return data.count > maxHeaderBytes ? .failure(431) : .incomplete
        }
        guard terminator.lowerBound - data.startIndex <= maxHeaderBytes else { return .failure(431) }
        guard let head = String(data: data[data.startIndex..<terminator.lowerBound], encoding: .utf8) else {
            return .failure(400)
        }

        var lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ", omittingEmptySubsequences: true)
        guard requestLine.count == 3, requestLine[2].hasPrefix("HTTP/1."),
              let components = URLComponents(string: String(requestLine[1])) else {
            return .failure(400)
        }

        var headers: [String: String] = [:]
        for line in lines where !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { return .failure(400) }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }

        if let encoding = headers["transfer-encoding"], encoding.lowercased() != "identity" {
            return .failure(411)
        }
        var bodyLength = 0
        if let lengthText = headers["content-length"] {
            guard let length = Int(lengthText), length >= 0 else { return .failure(400) }
            bodyLength = length
        }
        guard bodyLength <= maxBodyBytes else { return .failure(413) }

        let bodyStart = terminator.upperBound
        guard data.endIndex - bodyStart >= bodyLength else { return .incomplete }
        let body = Data(data[bodyStart..<(bodyStart + bodyLength)])

        var query: [String: String] = [:]
        for item in components.queryItems ?? [] { query[item.name] = item.value ?? "" }

        let request = LocalAPIHTTPRequest(
            method: String(requestLine[0]).uppercased(),
            path: components.path,
            query: query,
            headers: headers,
            body: body
        )
        return .complete(request, consumed: bodyStart + bodyLength - data.startIndex)
    }
}

nonisolated struct LocalAPIHTTPResponse: Equatable, Sendable {
    var status: Int
    var contentType: String
    var body: Data

    static func json(status: Int, _ object: some Encodable) -> LocalAPIHTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let body = (try? encoder.encode(object)) ?? Data("{}".utf8)
        return LocalAPIHTTPResponse(status: status, contentType: "application/json; charset=utf-8", body: body)
    }

    static func error(status: Int, _ message: String) -> LocalAPIHTTPResponse {
        json(status: status, ["error": message])
    }

    func serialized() -> Data {
        var head = "HTTP/1.1 \(status) \(Self.reasonPhrase(status))\r\n"
        head += "Content-Type: \(contentType)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Cache-Control: no-store\r\n"
        head += "Connection: close\r\n\r\n"
        return Data(head.utf8) + body
    }

    static func reasonPhrase(_ status: Int) -> String {
        switch status {
        case 200: "OK"
        case 400: "Bad Request"
        case 401: "Unauthorized"
        case 403: "Forbidden"
        case 404: "Not Found"
        case 409: "Conflict"
        case 405: "Method Not Allowed"
        case 411: "Length Required"
        case 413: "Content Too Large"
        case 415: "Unsupported Media Type"
        case 422: "Unprocessable Content"
        case 431: "Request Header Fields Too Large"
        case 503: "Service Unavailable"
        default: status < 500 ? "Client Error" : "Internal Server Error"
        }
    }
}

/// A failure whose message the Local API returns to the client as is.
struct LocalAPIError: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}
