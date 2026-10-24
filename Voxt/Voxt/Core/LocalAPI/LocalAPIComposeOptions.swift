// LocalAPIComposeOptions.swift
// Provides the validated options of a Local API compose request. The rules mirror the
// phone's CompositionOptions exactly so a request the keyboard allowed is always accepted
// here, and nothing else is.

import Foundation

nonisolated enum LocalAPIComposeMode: String, Codable, Equatable, Sendable, CaseIterable {
    case dictate, translate, edit, write
}

nonisolated enum LocalAPIComposeStyle: String, Codable, Equatable, Sendable, CaseIterable {
    case automatic, formal, casual, concise
}

/// One phone-kept wording pair. It exists only on the compose request. Nothing here is stored.
nonisolated struct LocalAPIStyleExample: Equatable, Sendable {
    var before: String
    var after: String

    static func digitSequences(in text: String) -> [String] {
        var sequences: [String] = []
        var current = ""
        for character in text {
            let isDigit = character.unicodeScalars.allSatisfy { $0.value >= 48 && $0.value <= 57 }
            if isDigit {
                current.append(character)
            } else if !current.isEmpty {
                sequences.append(current)
                current = ""
            }
        }
        if !current.isEmpty { sequences.append(current) }
        return sequences
    }
}

nonisolated enum LocalAPIComposeDecoding: Error {
    case styleExamples(String)
}

nonisolated struct LocalAPIComposeOptions: Equatable, Sendable {
    static let maximumTargetLanguageCount = 64
    static let maximumSelectedTextCount = 8_000
    static let maximumContextCount = 2_000
    static let maximumVocabularyCount = 200
    static let maximumVocabularyTermCount = 100
    static let maximumCustomPromptCount = 4_000

    /// One regional variant per language base; anything else is rejected.
    static let allowedLanguageVariants: Set<String> = [
        "en-US", "en-GB", "en-AU", "en-CA",
        "zh-Hans", "zh-Hant",
        "es-ES", "es-MX",
        "fr-FR", "fr-CA",
        "pt-PT", "pt-BR",
    ]

    var mode: LocalAPIComposeMode = .dictate
    var targetLanguage: String?
    var selectedText: String = ""
    var contextBefore: String = ""
    var contextAfter: String = ""
    var style: LocalAPIComposeStyle = .automatic
    var vocabulary: [String] = []
    var languageVariants: [String] = []
    var styleExamples: [LocalAPIStyleExample] = []
    /// A paired phone's Developer prompt for this mode. It replaces the built-in instruction
    /// for this one request; nothing is stored. nil uses the built-in prompt.
    var customPrompt: String?

    init(
        mode: LocalAPIComposeMode = .dictate,
        targetLanguage: String? = nil,
        selectedText: String = "",
        contextBefore: String = "",
        contextAfter: String = "",
        style: LocalAPIComposeStyle = .automatic,
        vocabulary: [String] = [],
        languageVariants: [String] = [],
        styleExamples: [LocalAPIStyleExample] = []
    ) {
        self.mode = mode
        self.targetLanguage = targetLanguage
        self.selectedText = selectedText
        self.contextBefore = contextBefore
        self.contextAfter = contextAfter
        self.style = style
        self.vocabulary = vocabulary
        self.languageVariants = languageVariants
        self.styleExamples = styleExamples
    }

    /// nil when the options satisfy every rule, otherwise the HTTP rejection reason.
    func validationError() -> String? {
        if mode == .edit, selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Select the text you want to edit first."
        }
        if mode == .translate, targetLanguage == nil {
            return "Choose a translation language first."
        }
        if let targetLanguage, !Self.isValidTargetLanguage(targetLanguage) {
            return "Choose a valid translation language."
        }
        if selectedText.count > Self.maximumSelectedTextCount
            || contextBefore.count > Self.maximumContextCount
            || contextAfter.count > Self.maximumContextCount {
            return "Select a shorter passage to work on."
        }
        if vocabulary.count > Self.maximumVocabularyCount
            || !vocabulary.allSatisfy({ term in
                !term.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && term.count <= Self.maximumVocabularyTermCount
            }) {
            return "Use at most 200 dictionary entries, each up to 100 characters."
        }
        if !languageVariants.allSatisfy({ Self.allowedLanguageVariants.contains($0) }) {
            return "Choose a supported language variant."
        }
        let variantBases = languageVariants.map { variant in
            variant.split(separator: "-").first.map(String.init) ?? variant
        }
        if Set(variantBases).count != variantBases.count {
            return "Pick at most one variant per language."
        }
        if let styleError = Self.styleExampleError(styleExamples) {
            return styleError
        }
        if let customPrompt, customPrompt.count > Self.maximumCustomPromptCount {
            return "Use a custom prompt of at most 4,000 characters."
        }
        return nil
    }

    /// Semantic limits for pairs that already decoded. Structure errors are thrown earlier.
    static func styleExampleError(_ examples: [LocalAPIStyleExample]) -> String? {
        if examples.count > 8 {
            return "Use at most 8 personal style examples."
        }
        var seen: Set<String> = []
        for example in examples {
            if example.before.isEmpty || example.after.isEmpty
                || example.before.count > 120 || example.after.count > 120 {
                return "A personal style example must be 1 to 120 characters."
            }
            if example.before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || example.after.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "A personal style example must not be blank."
            }
            if example.before == example.after {
                return "The personal style example texts must differ."
            }
            if LocalAPIStyleExample.digitSequences(in: example.before)
                != LocalAPIStyleExample.digitSequences(in: example.after) {
                return "Personal style examples must keep the same digit sequences."
            }
            let identity = example.before + "\u{0}" + example.after
            if !seen.insert(identity).inserted {
                return "Personal style examples must not repeat."
            }
        }
        return nil
    }

    /// A well-formed BCP-47 label whose base subtag is an ISO language. This says nothing
    /// about which languages the selected models actually support.
    static func isValidTargetLanguage(_ tag: String) -> Bool {
        let base = tag.split(separator: "-").first.map(String.init)?.lowercased() ?? ""
        return tag.count <= maximumTargetLanguageCount
            && tag.range(of: "^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$", options: .regularExpression) != nil
            && Locale.isoLanguageCodes.contains(base)
    }
}

extension LocalAPIComposeOptions: Decodable {
    enum CodingKeys: String, CodingKey {
        case mode, targetLanguage, selectedText, contextBefore, contextAfter, style, vocabulary, languageVariants, styleExamples, customPrompt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        mode = try container.decodeIfPresent(LocalAPIComposeMode.self, forKey: .mode) ?? .dictate
        targetLanguage = try container.decodeIfPresent(String.self, forKey: .targetLanguage)
        selectedText = try container.decodeIfPresent(String.self, forKey: .selectedText) ?? ""
        contextBefore = try container.decodeIfPresent(String.self, forKey: .contextBefore) ?? ""
        contextAfter = try container.decodeIfPresent(String.self, forKey: .contextAfter) ?? ""
        style = try container.decodeIfPresent(LocalAPIComposeStyle.self, forKey: .style) ?? .automatic
        vocabulary = try container.decodeIfPresent([String].self, forKey: .vocabulary) ?? []
        languageVariants = try container.decodeIfPresent([String].self, forKey: .languageVariants) ?? []
        styleExamples = try Self.decodeStyleExamples(from: container)
        let prompt = try container.decodeIfPresent(String.self, forKey: .customPrompt)?.trimmingCharacters(in: .whitespacesAndNewlines)
        customPrompt = prompt?.isEmpty == false ? prompt : nil
    }

    private static func decodeStyleExamples(from container: KeyedDecodingContainer<CodingKeys>) throws -> [LocalAPIStyleExample] {
        guard container.contains(.styleExamples) else { return [] }
        if try container.decodeNil(forKey: .styleExamples) {
            throw LocalAPIComposeDecoding.styleExamples("The personal style request has invalid fields.")
        }
        let decoded: [StyleExampleObject]
        do {
            decoded = try container.decode([StyleExampleObject].self, forKey: .styleExamples)
        } catch let error as LocalAPIComposeDecoding {
            throw error
        } catch {
            throw LocalAPIComposeDecoding.styleExamples("The personal style request has invalid fields.")
        }
        return decoded.map { LocalAPIStyleExample(before: $0.before, after: $0.after) }
    }
}

private struct StyleExampleObject: Decodable {
    var before: String
    var after: String

    private struct AnyKey: CodingKey {
        var stringValue: String
        var intValue: Int?
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) {
            self.intValue = intValue
            self.stringValue = "\(intValue)"
        }
    }

    private enum Field: String, CodingKey { case before, after }

    init(from decoder: Decoder) throws {
        let present = try decoder.container(keyedBy: AnyKey.self)
        guard Set(present.allKeys.map(\.stringValue)) == ["before", "after"] else {
            throw LocalAPIComposeDecoding.styleExamples("The personal style request has invalid fields.")
        }
        let fields = try decoder.container(keyedBy: Field.self)
        guard let before = try? fields.decode(String.self, forKey: .before),
              let after = try? fields.decode(String.self, forKey: .after) else {
            throw LocalAPIComposeDecoding.styleExamples("The personal style request has invalid fields.")
        }
        self.before = before
        self.after = after
    }
}
