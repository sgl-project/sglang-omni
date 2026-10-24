// DoubaoDictionaryPayloadBuilder.swift
// Provides Doubao Dictionary Payload Builder for remote provider configuration.

import Foundation

enum DoubaoDictionaryMode: String, Codable, CaseIterable, Identifiable {
    case off
    case requestScoped

    var id: String { rawValue }
}

struct DoubaoDictionaryRequestPayload: Equatable {
    var hotwords: [String]
    var correctWords: [String: String]

    init(
        hotwords: [String] = [],
        correctWords: [String: String] = [:]
    ) {
        self.hotwords = hotwords
        self.correctWords = correctWords
    }

    var isEmpty: Bool {
        hotwords.isEmpty && correctWords.isEmpty
    }

    var usesNativeCorrections: Bool {
        !correctWords.isEmpty
    }
}

enum DoubaoDictionaryRequestPayloadBuilder {
    static func build(
        configuration: RemoteProviderConfiguration,
        entries: [DictionaryEntry],
        dictionaryEnabled: Bool
    ) -> DoubaoDictionaryRequestPayload {
        guard dictionaryEnabled else { return .init() }

        switch configuration.doubaoDictionaryModeValue {
        case .off:
            return .init()
        case .requestScoped:
            return buildRequestScoped(configuration: configuration, entries: entries)
        }
    }

    private static func buildRequestScoped(
        configuration: RemoteProviderConfiguration,
        entries: [DictionaryEntry]
    ) -> DoubaoDictionaryRequestPayload {
        let activeEntries = entries.filter { $0.status == .active }
        var hotwords: [String] = []
        var hotwordSeen = Set<String>()
        var correctWords: [String: String] = [:]

        for entry in activeEntries {
            guard let term = cleanedWord(entry.term) else { continue }
            let normalizedTerm = DictionaryStore.normalizeTerm(term)
            if configuration.doubaoEnableRequestHotwords,
               !normalizedTerm.isEmpty,
               hotwordSeen.insert(normalizedTerm).inserted {
                hotwords.append(term)
            }

            guard configuration.doubaoEnableRequestCorrections else { continue }
            for replacement in entry.replacementTerms {
                guard let alias = cleanedWord(replacement.text) else { continue }
                let normalizedAlias = DictionaryStore.normalizeTerm(alias)
                guard !normalizedAlias.isEmpty, normalizedAlias != normalizedTerm else { continue }
                correctWords[alias] = term
            }
        }

        return .init(hotwords: Array(hotwords.prefix(5_000)), correctWords: correctWords)
    }

    private static func cleanedWord(_ rawValue: String) -> String? {
        let cleaned = rawValue
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return nil }
        return cleaned
    }
}
