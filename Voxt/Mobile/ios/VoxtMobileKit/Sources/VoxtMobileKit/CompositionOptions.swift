import Foundation

/// Per-recording intent. Context is supplied by the keyboard, never inferred from the Mac's frontmost app.
public struct CompositionOptions: Codable, Equatable, Sendable {
    public enum Mode: String, Codable, CaseIterable, Sendable {
        case dictate, translate, edit, write

        public var title: String {
            switch self {
            case .dictate: VoxtL10n.shared.string("Dictate")
            case .translate: VoxtL10n.shared.string("Translate")
            case .edit: VoxtL10n.shared.string("Edit selection")
            case .write: VoxtL10n.shared.string("Help me write")
            }
        }
    }

    public enum Style: String, Codable, CaseIterable, Sendable {
        case automatic, formal, casual, concise

        public var title: String { VoxtL10n.shared.string(rawValue.capitalized) }
    }

    public struct ValidationError: Error, Equatable, LocalizedError {
        public let message: String
        public var errorDescription: String? { message }
    }

    public var mode: Mode
    public var targetLanguage: String?
    public var selectedText: String
    public var contextBefore: String
    public var contextAfter: String
    public var style: Style
    public var vocabulary: [String]
    public var languageVariants: [String]
    public var styleExamples: [PersonalStyleExample]
    public var literalDictation: Bool
    /// Developer mode's prompt for this mode, attached by the app when the request is sent.
    /// The Mac uses it instead of its built-in instruction for this request only.
    public var customPrompt: String?

    public static let maximumCustomPromptCount = 4_000
    public static let supportedLanguageVariants = ["en-US", "en-GB", "en-AU", "en-CA", "zh-Hans", "zh-Hant", "es-ES", "es-MX", "fr-FR", "fr-CA", "pt-PT", "pt-BR"]

    public init(mode: Mode = .dictate, targetLanguage: String? = nil, selectedText: String = "", contextBefore: String = "", contextAfter: String = "", style: Style = .automatic, vocabulary: [String] = [], languageVariants: [String] = [], styleExamples: [PersonalStyleExample] = [], literalDictation: Bool = false) {
        self.mode = mode
        self.targetLanguage = targetLanguage
        self.selectedText = selectedText
        self.contextBefore = contextBefore
        self.contextAfter = contextAfter
        self.style = style
        self.vocabulary = vocabulary
        self.languageVariants = languageVariants
        self.styleExamples = styleExamples
        self.literalDictation = literalDictation
    }

    private enum CodingKeys: String, CodingKey {
        case mode, targetLanguage, selectedText, contextBefore, contextAfter, style, vocabulary, languageVariants, styleExamples, literalDictation, customPrompt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        mode = try c.decode(Mode.self, forKey: .mode)
        targetLanguage = try c.decodeIfPresent(String.self, forKey: .targetLanguage)
        selectedText = try c.decode(String.self, forKey: .selectedText)
        contextBefore = try c.decode(String.self, forKey: .contextBefore)
        contextAfter = try c.decode(String.self, forKey: .contextAfter)
        style = try c.decode(Style.self, forKey: .style)
        vocabulary = try c.decode([String].self, forKey: .vocabulary)
        languageVariants = try c.decodeIfPresent([String].self, forKey: .languageVariants) ?? []
        styleExamples = try c.decodeIfPresent([PersonalStyleExample].self, forKey: .styleExamples) ?? []
        literalDictation = try c.decodeIfPresent(Bool.self, forKey: .literalDictation) ?? false
        customPrompt = try c.decodeIfPresent(String.self, forKey: .customPrompt)
    }

    /// Only this exact legacy request can safely use the older dictation endpoint.
    public var isPlainDictation: Bool { self == CompositionOptions() }

    public func validate() throws {
        guard styleExamples.count <= 8, styleExamples.allSatisfy(\.isValid),
              !styleExamples.indices.contains(where: { index in styleExamples[..<index].contains(styleExamples[index]) }) else {
            throw ValidationError(message: "Use up to 8 different style examples, each up to 120 characters, keeping the same numbers.")
        }
        let bases = languageVariants.map { $0.split(separator: "-").first.map(String.init) ?? "" }
        guard languageVariants.allSatisfy(Self.supportedLanguageVariants.contains), Set(bases).count == bases.count else {
            throw ValidationError(message: "Choose one supported variant per language.")
        }
        if mode == .edit && selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw ValidationError(message: "Select the text you want to edit first.")
        }
        if mode == .translate && targetLanguage == nil {
            throw ValidationError(message: "Choose a translation language first.")
        }
        if let targetLanguage {
            let base = targetLanguage.split(separator: "-").first.map(String.init)?.lowercased() ?? ""
            guard targetLanguage.count <= 64,
                  targetLanguage.range(of: "^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$", options: .regularExpression) != nil,
                  Locale.isoLanguageCodes.contains(base) else {
                throw ValidationError(message: "Choose a valid translation language.")
            }
        }
        guard selectedText.count <= 8_000, contextBefore.count <= 2_000, contextAfter.count <= 2_000 else {
            throw ValidationError(message: "Select a shorter passage to work on.")
        }
        guard (customPrompt?.count ?? 0) <= Self.maximumCustomPromptCount else {
            throw ValidationError(message: "Use a custom prompt of at most 4,000 characters.")
        }
        guard vocabulary.count <= 200, vocabulary.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.count <= 100 }) else {
            throw ValidationError(message: "Use at most 200 dictionary entries, each up to 100 characters.")
        }
    }
}
