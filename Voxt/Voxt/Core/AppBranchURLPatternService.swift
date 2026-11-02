// AppBranchURLPatternService.swift
// Provides App Branch URLPattern Service for core app behavior.

import Foundation

enum AppBranchURLPatternService {
    struct URLGroupPromptMatch: Equatable {
        let groupID: UUID
        let groupName: String
        let pattern: String
        let prompt: String
    }

    struct URLGroupMatch: Equatable {
        let groupID: UUID
        let groupName: String
        let pattern: String
    }

    nonisolated static func canonicalizedPattern(_ value: String) -> String {
        var normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized.hasPrefix("https://") {
            normalized.removeFirst("https://".count)
        } else if normalized.hasPrefix("http://") {
            normalized.removeFirst("http://".count)
        }
        if !normalized.contains("/") {
            normalized += "/*"
        } else if normalized.hasSuffix("/") {
            normalized += "*"
        }
        return normalized
    }

    nonisolated static func normalizedPattern(_ value: String) -> String {
        canonicalizedPattern(value)
    }

    nonisolated static func isValidWildcardURLPattern(_ pattern: String) -> Bool {
        let value = normalizedPattern(pattern)
        guard !value.isEmpty else { return false }
        guard !value.contains("://") else { return false }
        guard !value.contains(" ") else { return false }
        guard value.contains(".") else { return false }
        guard value.contains("/") else { return false }

        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyz0123456789-._/*")
        return value.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    nonisolated static func normalizedURLForMatching(_ rawURL: String?) -> String? {
        guard let rawURL else { return nil }
        let trimmed = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let withScheme = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
        if let components = URLComponents(string: withScheme), let host = components.host?.lowercased() {
            let path = components.path.isEmpty ? "/" : components.path.lowercased()
            return "\(host)\(path)"
        }
        return trimmed.lowercased()
    }

    nonisolated static func wildcardMatches(pattern: String, candidate: String) -> Bool {
        let normalizedPattern = pattern.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalizedPattern.isEmpty else { return false }

        let escaped = NSRegularExpression.escapedPattern(for: normalizedPattern)
        let regexPattern = "^" + escaped.replacingOccurrences(of: "\\*", with: ".*") + "$"
        guard let regex = try? NSRegularExpression(pattern: regexPattern) else { return false }
        let range = NSRange(location: 0, length: (candidate as NSString).length)
        return regex.firstMatch(in: candidate.lowercased(), options: [], range: range) != nil
    }

    nonisolated static func firstPromptMatch(
        groups: [AppBranchGroup],
        urlsByID: [UUID: String],
        normalizedURL: String
    ) -> URLGroupPromptMatch? {
        for group in groups {
            for urlID in group.urlPatternIDs {
                guard let pattern = urlsByID[urlID], wildcardMatches(pattern: pattern, candidate: normalizedURL) else {
                    continue
                }
                let prompt = group.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                if !prompt.isEmpty {
                    return URLGroupPromptMatch(
                        groupID: group.id,
                        groupName: group.name,
                        pattern: pattern,
                        prompt: prompt
                    )
                }
            }
        }
        return nil
    }

    nonisolated static func firstGroupMatch(
        groups: [AppBranchGroup],
        urlsByID: [UUID: String],
        normalizedURL: String
    ) -> URLGroupMatch? {
        for group in groups {
            for urlID in group.urlPatternIDs {
                guard let pattern = urlsByID[urlID], wildcardMatches(pattern: pattern, candidate: normalizedURL) else {
                    continue
                }
                return URLGroupMatch(
                    groupID: group.id,
                    groupName: group.name,
                    pattern: pattern
                )
            }
        }
        return nil
    }
}
