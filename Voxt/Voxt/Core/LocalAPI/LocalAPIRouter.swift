// LocalAPIRouter.swift
// Provides token checks and request routing for the Local API server.

import Foundation
import Security

nonisolated struct LocalAPIDictationRequest: Equatable, Sendable {
    var audio: Data
    var fileExtension: String
    var enhance: Bool
    var language: ASRLanguageSelection = .userSettings
}

nonisolated struct LocalAPIComposeRequest: Equatable, Sendable {
    var audio: Data
    var fileExtension: String
    var options: LocalAPIComposeOptions
    var language: ASRLanguageSelection = .userSettings
}

private struct LocalAPIComposeRequestBody: Decodable {
    var audio: String
    var contentType: String
    var options: LocalAPIComposeOptions
    var language: String? = nil
}

nonisolated enum LocalAPIRoute: Equatable, Sendable {
    case health
    case capabilities
    /// The built-in compose instructions, for a client's prompt editor.
    case composePrompts
    case dictation(LocalAPIDictationRequest)
    case historySync(HistorySyncRequest)
    case compose(LocalAPIComposeRequest)
    case correction(LocalAPICorrectionRequest)
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
            guard let language = Self.languageSelection(request.query["language"]) else {
                return .reject(.error(status: 400, "Set language to auto or a language name such as English."))
            }
            return .dictation(LocalAPIDictationRequest(
                audio: request.body,
                fileExtension: fileExtension,
                enhance: request.query["enhance"] != "0",
                language: language
            ))
        case "/v1/history/sync":
            guard request.method == "POST" else { return .reject(.error(status: 405, "Use POST.")) }
            let mediaType = request.header("content-type")?
                .split(separator: ";").first?
                .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
            guard mediaType == "application/json" else {
                return .reject(.error(status: 415, "Send application/json."))
            }
            switch HistorySyncRequest.parse(request.body) {
            case let .success(parsed):
                return .historySync(parsed)
            case let .failure(message):
                return .reject(.error(status: 400, message))
            }
        case "/v1/compose/prompts":
            guard request.method == "GET" else { return .reject(.error(status: 405, "Use GET.")) }
            return .composePrompts
        case "/v1/compose":
            guard request.method == "POST" else { return .reject(.error(status: 405, "Use POST.")) }
            let mediaType = request.header("content-type")?
                .split(separator: ";").first?
                .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
            guard mediaType == "application/json" else {
                return .reject(.error(status: 415, "Send application/json."))
            }
            let body: LocalAPIComposeRequestBody
            do {
                body = try JSONDecoder().decode(LocalAPIComposeRequestBody.self, from: request.body)
            } catch let LocalAPIComposeDecoding.styleExamples(message) {
                return .reject(.error(status: 400, message))
            } catch {
                return .reject(.error(status: 400, "The compose request is not valid JSON."))
            }
            guard let audio = Data(base64Encoded: body.audio), !audio.isEmpty else {
                return .reject(.error(status: 400, "The request has no audio."))
            }
            guard let fileExtension = Self.audioExtensions[body.contentType.lowercased()] else {
                return .reject(.error(status: 415, "Send audio/wav or audio/mp4."))
            }
            if let reason = body.options.validationError() {
                return .reject(.error(status: 400, reason))
            }
            guard let language = Self.languageSelection(body.language) else {
                return .reject(.error(status: 400, "Set language to auto or a language name such as English."))
            }
            return .compose(LocalAPIComposeRequest(
                audio: audio,
                fileExtension: fileExtension,
                options: body.options,
                language: language
            ))
        case "/v1/corrections":
            guard request.method == "POST" else { return .reject(.error(status: 405, "Use POST.")) }
            let mediaType = request.header("content-type")?
                .split(separator: ";").first?
                .trimmingCharacters(in: .whitespaces).lowercased() ?? ""
            guard mediaType == "application/json" else {
                return .reject(.error(status: 415, "Send application/json."))
            }
            let object: Any
            do {
                object = try JSONSerialization.jsonObject(with: request.body)
            } catch {
                return .reject(.error(status: 400, "The corrections request is not valid JSON."))
            }
            guard let fields = object as? [String: Any] else {
                return .reject(.error(status: 400, "The corrections request is not valid JSON."))
            }
            guard let original = fields["original"] as? String, let edited = fields["edited"] as? String else {
                return .reject(.error(status: 400, "original and edited must be strings."))
            }
            guard !original.isEmpty, !edited.isEmpty else {
                return .reject(.error(status: 400, "The original and edited text must not be empty."))
            }
            guard original.count <= 2_000, edited.count <= 2_000 else {
                return .reject(.error(status: 400, "Use at most 2000 characters for each passage."))
            }
            guard original != edited else {
                return .reject(.error(status: 400, "The original and edited text are the same."))
            }
            return .correction(LocalAPICorrectionRequest(original: original, edited: edited))
        default:
            return .reject(.error(status: 404, "Unknown path."))
        }
    }

    /// No value keeps the user's language settings; "auto" detects the language; a name such as
    /// "English" transcribes in that language. nil for a value that is not a language name.
    static func languageSelection(_ value: String?) -> ASRLanguageSelection? {
        guard let value else { return .userSettings }
        let name = value.trimmingCharacters(in: .whitespaces)
        if name.lowercased() == "auto" { return .request(nil) }
        guard (1...40).contains(name.count),
              name.allSatisfy({ $0.isLetter || $0 == " " || $0 == "-" }) else { return nil }
        return .request(name)
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
