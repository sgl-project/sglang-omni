// VoxtLogRedactor.swift
// Redacts sensitive values before logs are written to OSLog or local files.

import Foundation
import Logging

enum VoxtLogRedactor {
    nonisolated private static let sensitiveHTTPHeaderNames: Set<String> = [
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-api-access-key",
        "x-api-app-key"
    ]

    private static let sensitivePatterns: [String] = [
        #"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]+"#,
        #"(?i)([?&](?:api_key|apikey|key|token|access_token|refresh_token|secret|password)=)[^&#\s]+"#,
        #"(?i)((?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|oauth[-_ ]?token|bearer[-_ ]?token|token|secret|password)\s*[:=]\s*)[^\s,;&]+"#
    ]

    private nonisolated static let compiledPatterns: [NSRegularExpression] = sensitivePatterns.compactMap {
        try? NSRegularExpression(pattern: $0)
    }

    nonisolated static func redact(_ text: String, privacy: VoxtLogPrivacy = .automatic) -> String {
        switch privacy {
        case .visible:
            return text
        case .sensitive:
            return "<redacted>"
        case .preview(let limit):
            return preview(text, limit: limit)
        case .automatic:
            return redactSensitiveValues(in: text)
        }
    }

    nonisolated static func preview(_ text: String, limit: Int = 1_200) -> String {
        let normalized = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return "<empty>" }
        let redacted = redactSensitiveValues(in: normalized)
        guard redacted.count > limit else { return redacted }
        let endIndex = redacted.index(redacted.startIndex, offsetBy: max(0, limit))
        return "\(redacted[..<endIndex])… [truncated]"
    }

    nonisolated static func redactedMetadata(_ metadata: Logger.Metadata?) -> Logger.Metadata? {
        guard let metadata else { return nil }
        var result: Logger.Metadata = [:]
        for (key, value) in metadata {
            result[key] = isSensitiveMetadataKey(key) ? .string("<redacted>") : redactedMetadataValue(value)
        }
        return result
    }

    nonisolated static func redactedHTTPHeaders(_ headers: [String: String]) -> [String: String] {
        headers.reduce(into: [String: String]()) { result, pair in
            if isSensitiveHTTPHeaderName(pair.key) {
                result[pair.key] = "<redacted>"
            } else {
                result[pair.key] = redactSensitiveValues(in: pair.value)
            }
        }
    }

    private nonisolated static func redactedMetadataValue(_ value: Logger.Metadata.Value) -> Logger.Metadata.Value {
        switch value {
        case .string(let string):
            return .string(redactSensitiveValues(in: string))
        case .stringConvertible(let value):
            return .string(redactSensitiveValues(in: String(describing: value)))
        case .array(let values):
            return .array(values.map(redactedMetadataValue))
        case .dictionary(let dictionary):
            var result: Logger.Metadata = [:]
            for (key, value) in dictionary {
                result[key] = isSensitiveMetadataKey(key) ? .string("<redacted>") : redactedMetadataValue(value)
            }
            return .dictionary(result)
        }
    }

    /// Structured metadata separates the key from the value: scanning only the
    /// string value cannot recognize ["apiKey": "opaque-secret"]. Match exact
    /// normalized keys so safe counters such as tokenCount remain useful.
    private nonisolated static func isSensitiveMetadataKey(_ key: String) -> Bool {
        let normalized = key.lowercased().filter { $0.isLetter || $0.isNumber }
        return sensitiveMetadataKeys.contains(normalized)
    }

    private nonisolated static let sensitiveMetadataKeys: Set<String> = [
            "authorization", "proxyauthorization", "cookie", "setcookie",
            "apikey", "xapikey", "xapiaccesskey", "xapiappkey", "accesstoken",
            "refreshtoken", "idtoken", "oauthtoken", "bearertoken", "token",
            "password", "proxypassword", "secret", "clientsecret", "credential", "credentials"
    ]

    private nonisolated static func isSensitiveHTTPHeaderName(_ name: String) -> Bool {
        let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return sensitiveHTTPHeaderNames.contains(normalized)
            || normalized.hasSuffix("-api-key")
            || normalized.hasSuffix("-access-key")
            || normalized.hasSuffix("-app-key")
            || normalized.contains("token")
            || normalized.contains("secret")
            || normalized.contains("credential")
    }

    private nonisolated static func redactSensitiveValues(in text: String) -> String {
        var output = text
        let home = NSHomeDirectory()
        if !home.isEmpty {
            output = output.replacingOccurrences(of: home, with: "~")
        }
        for pattern in compiledPatterns {
            let range = NSRange(output.startIndex..<output.endIndex, in: output)
            output = pattern.stringByReplacingMatches(
                in: output,
                options: [],
                range: range,
                withTemplate: "$1<redacted>"
            )
        }
        return output
    }
}
