import Foundation

struct ResumableDownloadDescriptor {
    let sourceURL: URL
    let destinationURL: URL
    let relativePath: String
    let expectedSize: Int64?
    let requiresExpectedSizeMatch: Bool
    let userAgent: String
    let bearerToken: String?
    let disableProxy: Bool
    let policy: ResumableDownloadPolicy
    let protocolClasses: [AnyClass]?

    init(
        sourceURL: URL,
        destinationURL: URL,
        relativePath: String,
        expectedSize: Int64?,
        requiresExpectedSizeMatch: Bool = true,
        userAgent: String,
        bearerToken: String? = nil,
        disableProxy: Bool,
        policy: ResumableDownloadPolicy = .default,
        protocolClasses: [AnyClass]? = nil
    ) {
        self.sourceURL = sourceURL
        self.destinationURL = destinationURL
        self.relativePath = relativePath
        self.expectedSize = expectedSize
        self.requiresExpectedSizeMatch = requiresExpectedSizeMatch
        self.userAgent = userAgent
        self.bearerToken = bearerToken
        self.disableProxy = disableProxy
        self.policy = policy
        self.protocolClasses = protocolClasses
    }
}

struct ResumableDownloadPolicy {
    let resumeThresholdBytes: Int64
    let stallTimeout: Duration
    let stallPollInterval: Duration
    let maxRecoveryAttempts: Int
    let initialBackoffSeconds: Double
    let maxBackoffSeconds: Double

    static let `default` = ResumableDownloadPolicy(
        resumeThresholdBytes: 64 * 1024 * 1024,
        stallTimeout: .seconds(45),
        stallPollInterval: .seconds(5),
        maxRecoveryAttempts: 5,
        initialBackoffSeconds: 1,
        maxBackoffSeconds: 30
    )
}

struct ResumableDownloadState: Codable, Equatable {
    let relativePath: String
    let sourceURL: String
    let expectedSize: Int64
    let etag: String
    let downloadedBytes: Int64
    let updatedAt: Date
    let rangeSupported: Bool
}

struct ResumableDownloadResult: Equatable {
    let bytesDownloaded: Int64
    let resumedFromBytes: Int64
    let rangeSupported: Bool
}

enum ResumableDownloadAttemptOutcome {
    case completed(ResumableDownloadResult)
    case restartFromZero(reason: String)
    case recoverableFailure(reason: String)
    case fatal(Error)
}

enum ResumableDownloadError: LocalizedError {
    case badServerResponse
    case unexpectedStatusCode(Int)
    case missingETag
    case invalidContentRange(String)
    case inconsistentExpectedSize(expected: Int64, actual: Int64)
    case inconsistentPartialState

    var errorDescription: String? {
        switch self {
        case .badServerResponse:
            return "Invalid response from model server."
        case .unexpectedStatusCode(let code):
            return "Download failed (HTTP \(code))."
        case .missingETag:
            return "Download server did not provide a stable ETag for resume."
        case .invalidContentRange(let value):
            return "Download resume returned invalid Content-Range: \(value)"
        case .inconsistentExpectedSize(let expected, let actual):
            return "Download size mismatch (expected \(expected), got \(actual))."
        case .inconsistentPartialState:
            return "Stored partial download state is inconsistent."
        }
    }

    var isRetryable: Bool {
        switch self {
        case .unexpectedStatusCode(let code):
            return code >= 500 || code == 429 || code == 408
        default:
            return false
        }
    }
}


struct ResumableResponseMetadata {
    let statusCode: Int
    let etag: String?
    let acceptRanges: Bool
    let contentLength: Int64?
    let contentRange: (start: Int64, end: Int64, total: Int64?)?

    nonisolated init(response: HTTPURLResponse) {
        statusCode = response.statusCode
        etag = response.value(forHTTPHeaderField: "ETag")
            ?? response.value(forHTTPHeaderField: "X-Linked-Etag")
        let acceptRangesValue = response.value(forHTTPHeaderField: "Accept-Ranges")?.lowercased() ?? ""
        acceptRanges = acceptRangesValue.contains("bytes")
        contentLength = response.expectedContentLength > 0 ? response.expectedContentLength : nil
        contentRange = Self.parseContentRange(response.value(forHTTPHeaderField: "Content-Range"))
    }

    private nonisolated static func parseContentRange(_ value: String?) -> (start: Int64, end: Int64, total: Int64?)? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.lowercased().hasPrefix("bytes ") else { return nil }
        let payload = trimmed.dropFirst("bytes ".count)
        let pieces = payload.split(separator: "/")
        guard pieces.count == 2 else { return nil }
        let rangePart = pieces[0].split(separator: "-")
        guard rangePart.count == 2,
              let start = Int64(rangePart[0]),
              let end = Int64(rangePart[1])
        else {
            return nil
        }
        let total = pieces[1] == "*" ? nil : Int64(pieces[1])
        return (start, end, total)
    }
}


enum ResumableDownloadLoopError: Error {
    case restartFromZero(reason: String)
    case recoverable(reason: String)

    var restartReason: String? {
        switch self {
        case .restartFromZero(let reason):
            return reason
        case .recoverable:
            return nil
        }
    }

    var recoverableReason: String? {
        switch self {
        case .restartFromZero:
            return nil
        case .recoverable(let reason):
            return reason
        }
    }
}
