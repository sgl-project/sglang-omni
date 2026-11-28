import Foundation

// The Go core's JSON types (ios/voxtcore/api.go): field names match its
// JSON tags.

/// The language model a request goes to: a preset with the user's
/// overrides and the key read from the Keychain.
public struct CoreProvider: Codable, Equatable, Sendable {
    public var preset: String
    public var baseURL: String?
    public var model: String?
    public var key: String?

    public init(preset: String, baseURL: String? = nil, model: String? = nil, key: String? = nil) {
        self.preset = preset
        self.baseURL = baseURL
        self.model = model
        self.key = key
    }
}

/// A dictionary entry: the term as it must be written and what speech
/// recognition tends to make of it.
public struct Term: Codable, Equatable, Hashable, Identifiable, Sendable {
    public var id = UUID()
    public var term: String
    public var aliases: [String]

    public init(term: String, aliases: [String] = []) {
        self.term = term
        self.aliases = aliases
    }

    enum CodingKeys: String, CodingKey { case term, aliases }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        term = try c.decode(String.self, forKey: .term)
        aliases = try c.decodeIfPresent([String].self, forKey: .aliases) ?? []
    }
}

/// A spoken trigger and the text it stands for.
public struct Snippet: Codable, Equatable, Hashable, Identifiable, Sendable {
    public var id = UUID()
    public var trigger: String
    public var text: String

    public init(trigger: String, text: String) {
        self.trigger = trigger
        self.text = text
    }

    enum CodingKeys: String, CodingKey { case trigger, text }
}

/// A feature's settings, as the core reads them.
public struct CoreFeature: Codable, Equatable, Sendable {
    public var cleanup: Bool
    public var structure: Bool?
    public var prompt: String?
}

/// An earlier turn of a Rewrite conversation.
public struct CoreTurn: Codable, Equatable, Sendable {
    public var instruction: String
    public var source: String?
    public var title: String?
    public var answer: String

    public init(instruction: String, source: String?, title: String?, answer: String) {
        self.instruction = instruction
        self.source = source
        self.title = title
        self.answer = answer
    }
}

/// One feature's language-model step.
public struct ProcessRequest: Codable, Equatable, Sendable {
    public var task: String
    public var text: String
    public var typed: Bool?
    public var instruction: String?
    public var targetLanguage: String?
    public var mainLanguage: String
    public var otherLanguages: [String]?
    public var feature: CoreFeature
    public var notes: String?
    public var terms: [Term]?
    public var snippets: [Snippet]?
    public var previous: [CoreTurn]?
    public var provider: CoreProvider
}

/// The step's answer.
public struct ProcessResult: Codable, Equatable, Sendable {
    public var text: String
    public var title: String?
    public var profile: String?
    public var degraded: Bool?
    public var warning: String?
    public var retried: Bool?
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var latencyMillis: Int
}

/// A recording for a remote speech service.
public struct SpeechRequest: Codable, Equatable, Sendable {
    public var path: String
    public var endpoint: String?
    public var model: String?
    public var key: String?
    public var language: String?
    public var prompt: String?
    public var terms: [Term]?
}

/// A remote speech service's text.
public struct SpeechResult: Codable, Equatable, Sendable {
    public var text: String
    public var language: String?
    public var model: String?
    public var latencyMillis: Int
}

/// A language model service the settings offer.
public struct Preset: Codable, Equatable, Hashable, Identifiable, Sendable {
    public var id: String
    public var name: String
    public var baseURL: String?
    public var keyOptional: Bool?
    public var models: [String]?
    public var defaultModel: String?
    public var keyURL: String?
}

/// A language the pickers offer.
public struct Language: Codable, Equatable, Hashable, Identifiable, Sendable {
    public var code: String
    public var name: String
    public var chinese: String
    public var native: String?

    public var id: String { code }

    /// The name to show in the app's language (Simplified Chinese or
    /// English).
    public func displayName(chinese useChinese: Bool) -> String {
        useChinese && !chinese.isEmpty ? chinese : name
    }
}

/// A connection test's result.
public struct TestResult: Codable, Equatable, Sendable {
    public var latencyMillis: Int
}
