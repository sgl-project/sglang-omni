import Foundation

/// A feature of a voice session, as on macOS: Transcribe writes down what
/// was said, Translate translates it, Rewrite follows a spoken instruction
/// over a text (or writes from scratch).
public enum Feature: String, Codable, CaseIterable, Identifiable, Sendable {
    case transcribe, translate, rewrite

    public var id: String { rawValue }
}

/// Where speech is recognised.
public enum SpeechEngineKind: String, Codable, Sendable {
    /// Apple's speech recognition, on the device when it can (offline).
    case onDevice
    /// An OpenAI-compatible speech service (/v1/audio/transcriptions).
    case remote
}

/// The iOS app's settings, kept as one JSON document (SettingsStore).
/// Field names follow the macOS settings where they mean the same, so that
/// a later sync can map them (issue #4684, milestone 4). Keys are never
/// here: they live in the Keychain (KeyStore).
public struct AppSettings: Codable, Equatable, Sendable {
    public struct General: Codable, Equatable, Sendable {
        public init() {}

        /// The user's main language ("" follows the device's).
        public var mainLanguage = ""
        public var otherLanguages: [String] = []
    }

    public struct FeatureSettings: Codable, Equatable, Sendable {
        public init() {}

        public var cleanup = true
        public var structure = false
        public var prompt = ""
    }

    public struct Translate: Codable, Equatable, Sendable {
        public init() {}

        public var cleanup = true
        public var prompt = ""
        public var targetLanguage = "en"
    }

    public struct LanguageModel: Codable, Equatable, Sendable {
        public init() {}

        public init(preset: String, baseURL: String = "", model: String = "") {
            self.preset = preset
            self.baseURL = baseURL
            self.model = model
        }

        /// A preset's id (VoxtCore.presets).
        public var preset = "openai"
        public var baseURL = ""
        public var model = ""
    }

    public struct Speech: Codable, Equatable, Sendable {
        public init() {}

        public var engine = SpeechEngineKind.onDevice
        /// Only on the device: never send the audio to Apple's servers.
        public var requireOffline = true
        public var endpoint = ""
        public var model = ""
    }

    public struct Dictionary: Codable, Equatable, Sendable {
        public init() {}

        public var enabled = true
        /// Markdown notes for the language model.
        public var notes = ""
        public var terms: [Term] = []
    }

    public var version = 1
    public var general = General()
    public var transcribe = FeatureSettings()
    public var translate = Translate()
    public var rewrite = FeatureSettings()
    public var languageModel = LanguageModel()
    public var speech = Speech()
    public var dictionary = Dictionary()
    public var snippets: [Snippet] = []
    /// The feature the voice screen opens with.
    public var lastFeature = Feature.transcribe

    public init() {}
}

public extension AppSettings {
    /// The main language: the setting, else the device's preferred one
    /// (a lang code: "zh", "en").
    func mainLanguage(preferred: [String] = Locale.preferredLanguages) -> String {
        if !general.mainLanguage.isEmpty {
            return general.mainLanguage
        }
        guard let first = preferred.first else {
            return "en"
        }
        return Locale(identifier: first).language.languageCode?.identifier ?? "en"
    }

    /// The glossary the requests carry: none when the dictionary is off.
    var activeTerms: [Term] {
        dictionary.enabled ? dictionary.terms.filter { !$0.term.trimmingCharacters(in: .whitespaces).isEmpty } : []
    }

    /// The notes the requests carry.
    var activeNotes: String? {
        dictionary.enabled && !dictionary.notes.isEmpty ? dictionary.notes : nil
    }
}

public extension AppSettings.LanguageModel {
    /// The core's setup of this language model, with its key.
    func provider(keys: KeyStore) -> CoreProvider {
        CoreProvider(preset: preset, baseURL: baseURL.isEmpty ? nil : baseURL,
                     model: model.isEmpty ? nil : model, key: keys.key(KeyAccount.languageModel(preset)))
    }
}

/// Loads and saves the settings document in a directory (Application
/// Support in the app, a temporary one in tests).
public final class SettingsStore: @unchecked Sendable {
    private let url: URL
    private let lock = NSLock()

    public init(directory: URL) {
        url = directory.appendingPathComponent("settings.json")
    }

    /// The saved settings; the defaults when there are none or they cannot
    /// be read (a broken file is kept aside, not overwritten).
    public func load() -> AppSettings {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: url) else {
            return AppSettings()
        }
        do {
            return try Self.decode(data)
        } catch {
            let aside = url.deletingPathExtension().appendingPathExtension("broken.json")
            try? FileManager.default.removeItem(at: aside)
            try? FileManager.default.moveItem(at: url, to: aside)
            return AppSettings()
        }
    }

    /// A document over the defaults: a field it lacks (written by an older
    /// version) or holds as null keeps its default instead of failing the
    /// whole document, which would reset the user's settings.
    static func decode(_ data: Data) throws -> AppSettings {
        let saved = try JSONSerialization.jsonObject(with: data)
        let defaults = try JSONSerialization.jsonObject(with: JSONEncoder().encode(AppSettings()))
        let merged = try JSONSerialization.data(withJSONObject: merge(saved, over: defaults))
        return try JSONDecoder().decode(AppSettings.self, from: merged)
    }

    private static func merge(_ value: Any, over base: Any) -> Any {
        guard var object = base as? [String: Any], let saved = value as? [String: Any] else {
            return value
        }
        for (key, v) in saved where !(v is NSNull) {
            object[key] = object[key].map { merge(v, over: $0) } ?? v
        }
        return object
    }

    public func save(_ settings: AppSettings) throws {
        lock.lock()
        defer { lock.unlock() }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(settings).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }
}
