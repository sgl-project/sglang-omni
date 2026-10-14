// LocalAPIRouter.swift
// Provides token checks and request routing for the Local API server.

import Foundation
import Security

nonisolated struct LocalAPIDictationRequest: Equatable, Sendable {
    var audio: Data
    var fileExtension: String
    var enhance: Bool
}

nonisolated enum LocalAPIRoute: Equatable, Sendable {
    case health
    case capabilities
    case dictation(LocalAPIDictationRequest)
    case reject(LocalAPIHTTPResponse)
}

nonisolated struct LocalAPIRouter: Sendable {
    /// Audio types AVAudioFile can read, mapped to the file extension it needs.
    static let audioExtensions: [String: String] = [
        "audio/wav": "wav",
        "audio/wave": "wav",
        "audio/x-wav": "wav",
        "audio/mp4": "m4a",
        "audio/m4a": "m4a",
        "audio/x-m4a": "m4a",
        "audio/aac": "aac",
        "audio/mpeg": "mp3",
        "audio/x-caf": "caf",
    ]

    let token: String
    let maxBodyBytes: Int

    func route(_ request: LocalAPIHTTPRequest) -> LocalAPIRoute {
        guard isAuthorized(request) else {
            return .reject(.error(status: 401, "Missing or wrong access token."))
        }

        switch request.path {
        case "/v1/health":
            guard request.method == "GET" else { return .reject(.error(status: 405, "Use GET.")) }
            return .health
        case "/v1/capabilities":
            guard request.method == "GET" else { return .reject(.error(status: 405, "Use GET.")) }
            return .capabilities
        case "/v1/dictations":
            guard request.method == "POST" else { return .reject(.error(status: 405, "Use POST.")) }
            guard !request.body.isEmpty else { return .reject(.error(status: 400, "The request has no audio.")) }
            let mediaType = request.header("content-type")?
                .split(separator: ";").first?
                .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
            guard let fileExtension = Self.audioExtensions[mediaType] else {
                return .reject(.error(status: 415, "Send audio/wav or audio/mp4."))
            }
            return .dictation(LocalAPIDictationRequest(
                audio: request.body,
                fileExtension: fileExtension,
                enhance: request.query["enhance"] != "0"
            ))
        default:
            return .reject(.error(status: 404, "Unknown path."))
        }
    }

    private func isAuthorized(_ request: LocalAPIHTTPRequest) -> Bool {
        guard !token.isEmpty, let header = request.header("authorization"), header.hasPrefix("Bearer ") else {
            return false
        }
        return Self.constantTimeEquals(Array(header.dropFirst("Bearer ".count).utf8), Array(token.utf8))
    }

    private static func constantTimeEquals(_ lhs: [UInt8], _ rhs: [UInt8]) -> Bool {
        var difference = UInt8(truncatingIfNeeded: lhs.count ^ rhs.count)
        for index in 0..<max(lhs.count, rhs.count) {
            let a = index < lhs.count ? lhs[index] : 0
            let b = index < rhs.count ? rhs[index] : 0
            difference |= a ^ b
        }
        return difference == 0
    }

    /// 32 random bytes, base64url without padding (43 characters).
    static func generateToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        if SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) != errSecSuccess {
            bytes = (0..<32).map { _ in UInt8.random(in: .min ... .max) }
        }
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
