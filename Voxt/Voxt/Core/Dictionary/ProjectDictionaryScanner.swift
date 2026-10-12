// ProjectDictionaryScanner.swift
// Provides Project Dictionary Scanner for dictionary matching and learning.

import Foundation

struct ProjectDictionaryDiscovery: Equatable {
    let roots: [URL]
    let terms: [String]
}

enum ProjectDictionaryScanner {
    private static let contentCharacterLimit = 32_768
    private static let maxScannedFiles = 240
    private static let maxReturnedTerms = 64
    private static let minimumScore = 2
    private static let decoratedCharacters = "._+-/"
    private static let separatorCharacters = CharacterSet(charactersIn: decoratedCharacters)
    private static let pathTokenRegex = makeRegex(#"\b[\p{L}\p{N}][\p{L}\p{N}._+\-/]{2,39}\b"#)
    private static let richTextTokenRegex = makeRegex(
        #"\b(?:[A-Z]{2,}[A-Za-z0-9._+\-/]*|[A-Za-z0-9]+(?:[._+\-/][A-Za-z0-9]+)+|[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+|[A-Za-z]*\d+[A-Za-z0-9._+\-/]*)\b"#
    )
    private static let hanTokenRegex = makeRegex(#"\p{Han}{2,12}"#)
    private static let stopwords: Set<String> = [
        "assistant", "cache", "claude", "codex", "config", "configuration",
        "conversation", "data", "directory", "document", "docs", "file", "files",
        "folder", "folders", "history", "instruction", "instructions", "message",
        "messages", "output", "path", "paths", "project", "projects", "prompt",
        "prompts", "readme", "response", "session", "settings", "system", "temp",
        "text", "tmp", "user", "workspace"
    ]

    static func scanDefaultContextDirectories(
        fileManager: FileManager = .default
    ) -> ProjectDictionaryDiscovery {
        let home = fileManager.homeDirectoryForCurrentUser
        let roots = [".codex", ".claude"]
            .map { home.appendingPathComponent($0, isDirectory: true) }
            .filter { directoryExists(at: $0, fileManager: fileManager) }
        return scanContextDirectories(roots, fileManager: fileManager)
    }

    static func scanContextDirectory(
        named directoryName: String,
        fileManager: FileManager = .default
    ) -> ProjectDictionaryDiscovery {
        let root = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(directoryName, isDirectory: true)
        return scanContextDirectories([root], fileManager: fileManager)
    }

    static func scanContextDirectories(
        _ directories: [URL],
        fileManager: FileManager = .default
    ) -> ProjectDictionaryDiscovery {
        let roots = directories.filter { directoryExists(at: $0, fileManager: fileManager) }
        guard !roots.isEmpty else {
            return ProjectDictionaryDiscovery(roots: [], terms: [])
        }

        var scores: [String: Int] = [:]
        var preferredSurfaces: [String: String] = [:]
        var scannedFiles = 0

        func record(terms: [String], weight: Int) {
            for term in terms {
                let normalized = normalize(term)
                guard isAcceptedTerm(term, normalized: normalized) else { continue }
                let singleMentionBonus = weight == 1 && term.hasProjectDictionaryDecoration ? 1 : 0
                scores[normalized, default: 0] += weight + singleMentionBonus
                preferredSurfaces[normalized] = preferredSurface(
                    existing: preferredSurfaces[normalized],
                    candidate: term
                )
            }
        }

        for root in roots {
            record(
                terms: candidateTerms(in: root.lastPathComponent, allowPlainLowercase: false),
                weight: 3
            )

            guard let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                options: [.skipsPackageDescendants],
                errorHandler: { _, _ in true }
            ) else {
                continue
            }

            while let url = enumerator.nextObject() as? URL {
                let relativePath = url.path.replacingOccurrences(of: root.path + "/", with: "")
                record(
                    terms: candidateTerms(in: relativePath, allowPlainLowercase: true),
                    weight: 3
                )

                guard scannedFiles < maxScannedFiles else { break }
                let resourceValues = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                guard resourceValues?.isRegularFile == true else { continue }
                guard (resourceValues?.fileSize ?? 0) <= contentCharacterLimit else { continue }
                guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]) else { continue }
                guard !looksBinary(data) else { continue }
                guard let text = String(data: data.prefix(contentCharacterLimit), encoding: .utf8) else { continue }
                scannedFiles += 1
                record(terms: candidateTerms(in: text, allowPlainLowercase: false), weight: 1)
            }
        }

        let terms = scores
            .filter { $0.value >= minimumScore }
            .sorted { lhs, rhs in
                if lhs.value != rhs.value {
                    return lhs.value > rhs.value
                }
                return (preferredSurfaces[lhs.key] ?? lhs.key)
                    .localizedCaseInsensitiveCompare(preferredSurfaces[rhs.key] ?? rhs.key) == .orderedAscending
            }
            .prefix(maxReturnedTerms)
            .compactMap { preferredSurfaces[$0.key] }

        return ProjectDictionaryDiscovery(roots: roots, terms: terms)
    }

    static func candidateTerms(in text: String, allowPlainLowercase: Bool) -> [String] {
        var terms: [String] = []
        let segments = allowPlainLowercase
            ? text.components(separatedBy: CharacterSet(charactersIn: "/\\")).filter { !$0.isEmpty }
            : [text]

        for segment in segments {
            let nsRange = NSRange(segment.startIndex..<segment.endIndex, in: segment)
            for regex in [allowPlainLowercase ? pathTokenRegex : richTextTokenRegex, hanTokenRegex] {
                let matches = regex.matches(in: segment, range: nsRange)
                for match in matches {
                    guard let range = Range(match.range, in: segment) else { continue }
                    let token = String(segment[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                    guard isCandidateToken(token, allowPlainLowercase: allowPlainLowercase) else { continue }
                    terms.append(token)
                }
            }
        }

        return uniqueTerms(terms)
    }

    private static func uniqueTerms(_ terms: [String]) -> [String] {
        var seen = Set<String>()
        return terms.filter { term in
            let normalized = normalize(term)
            guard !normalized.isEmpty, seen.insert(normalized).inserted else { return false }
            return true
        }
    }

    private static func preferredSurface(existing: String?, candidate: String) -> String {
        guard let existing else { return candidate }
        let existingDecorated = existing.hasProjectDictionaryDecoration
        let candidateDecorated = candidate.hasProjectDictionaryDecoration
        if candidateDecorated, !existingDecorated {
            return candidate
        }
        if candidate.count > existing.count, candidateDecorated == existingDecorated {
            return candidate
        }
        return existing
    }

    private static func isAcceptedTerm(_ term: String, normalized: String) -> Bool {
        guard !normalized.isEmpty else { return false }
        guard !stopwords.contains(normalized) else { return false }
        if term.unicodeScalars.contains(where: isHanScalar) {
            return term.count >= 2
        }
        guard term.rangeOfCharacter(from: .letters) != nil else { return false }
        return term.count >= 3 && term.count <= 40
    }

    private static func isCandidateToken(_ token: String, allowPlainLowercase: Bool) -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if trimmed.unicodeScalars.contains(where: isHanScalar) {
            return trimmed.count >= 2
        }

        let hasLetter = trimmed.rangeOfCharacter(from: .letters) != nil
        guard hasLetter else { return false }

        let hasUppercase = trimmed.contains(where: \.isUppercase)
        let hasDigit = trimmed.contains(where: \.isNumber)
        let hasSeparator = trimmed.rangeOfCharacter(from: separatorCharacters) != nil
        if allowPlainLowercase {
            return trimmed.count >= 3
        }
        return hasUppercase || hasDigit || hasSeparator
    }

    private static func normalize(_ term: String) -> String {
        term.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private static func directoryExists(at url: URL, fileManager: FileManager) -> Bool {
        var isDirectory = ObjCBool(false)
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }

    private static func looksBinary(_ data: Data) -> Bool {
        let sample = data.prefix(512)
        guard !sample.isEmpty else { return false }
        let nullCount = sample.reduce(into: 0) { count, byte in
            if byte == 0 {
                count += 1
            }
        }
        return nullCount * 20 > sample.count
    }

    nonisolated private static func isHanScalar(_ scalar: UnicodeScalar) -> Bool {
        (0x4E00...0x9FFF).contains(scalar.value)
    }

    private static func makeRegex(_ pattern: String) -> NSRegularExpression {
        do {
            return try NSRegularExpression(pattern: pattern)
        } catch {
            VoxtLog.dictionaryWarning("Project dictionary regex initialization failed for pattern \(pattern): \(error.localizedDescription)")
            if let fallback = try? NSRegularExpression(pattern: "$^") {
                return fallback
            }
            fatalError("ProjectDictionaryScanner could not initialize its fallback regex.")
        }
    }
}

private extension String {
    var hasProjectDictionaryDecoration: Bool {
        contains(where: { $0.isUppercase || ProjectDictionaryScannerDecorations.characters.contains($0) })
    }
}

private enum ProjectDictionaryScannerDecorations {
    static let characters = "._+-/"
}
