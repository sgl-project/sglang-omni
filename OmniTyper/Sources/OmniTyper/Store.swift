// SPDX-License-Identifier: Apache-2.0
import Combine
import Foundation

enum VoiceMode: String, Codable, CaseIterable, Identifiable {
    case dictate, translate, edit, ask
    var id: String { rawValue }
    var title: String {
        switch self { case .dictate: return L("mode.dictate.title"); case .translate: return L("mode.translate.title")
        case .edit: return L("mode.edit.title"); case .ask: return L("mode.ask.title") }
    }
    var icon: String {
        switch self { case .dictate: return "waveform"; case .translate: return "character.bubble"
        case .edit: return "pencil.line"; case .ask: return "sparkles" }
    }
    var detail: String {
        switch self {
        case .dictate: return L("mode.dictate.detail")
        case .translate: return L("mode.translate.detail")
        case .edit: return L("mode.edit.detail")
        case .ask: return L("mode.ask.detail")
        }
    }
}

struct TextAPISettings: Codable, Equatable {
    var baseURL = "http://127.0.0.1:11434/v1"
    var model = ""
    var optionsJSON = "{}"

    func payload(apiKey: String, requireModel: Bool = true) throws -> [String: Any] {
        let url = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard url.unicodeScalars.count <= 2048,
              !url.unicodeScalars.contains(where: { CharacterSet.whitespacesAndNewlines.union(.controlCharacters).contains($0) }),
              let components = URLComponents(string: url), ["http", "https"].contains(components.scheme ?? ""),
              let host = components.host, !host.isEmpty,
              components.user == nil, components.password == nil, components.query == nil, components.fragment == nil,
              components.port == nil || (1...65535).contains(components.port!) else {
            throw Failure("error.baseURL")
        }
        guard model.unicodeScalars.count <= 256, !model.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
              !requireModel || !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Failure("error.model")
        }
        guard apiKey.utf8.count <= 4096, apiKey.unicodeScalars.allSatisfy({ (33...126).contains($0.value) }) else {
            throw Failure("error.apiKey")
        }
        guard let data = optionsJSON.data(using: .utf8), data.count <= 8192,
              let options = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(options.keys).isDisjoint(with: ["model", "messages", "stream"]) else {
            throw Failure("error.options")
        }
        return ["text_api_url": url, "text_model": model, "text_api_key": apiKey, "text_api_options": options]
    }
}

struct Preferences: Codable, Equatable {
    var pythonExecutable = ""
    var asrModel = "mlx-community/Qwen3-ASR-0.6B-4bit"
    // Note (Codex): Optional fields preserve decoding of libraries saved before these settings existed.
    var textAPI: TextAPISettings?
    var textSettings: TextAPISettings {
        get { textAPI ?? TextAPISettings() }
        set { textAPI = newValue }
    }
    var language = ""
    var targetLanguage = "English"
    var style = "verbatim"
    var instructions = ""
    var microphoneUID = ""
    var shortcutKeyCode: UInt16 = 49
    var shortcutModifiers: UInt64 = 786432 // Control + Option
    var holdToTalk = false
    var sounds = true
    var autoPaste = true
    var saveHistory = true
    var historyDays = 30 // 0 = forever
    var keepAudio = false
    var appearance = "system"
    // Note (Jiaxin Deng): nil follows the system language and keeps older libraries decodable.
    var uiLanguage: String?
    // Note (Jiaxin Deng): Remember prior grants to distinguish invalidated permissions from first use.
    var accessibilityWasTrusted: Bool?

    static func combinedInstructions(_ defaults: String, _ app: String) throws -> String {
        guard [defaults, app].allSatisfy({ $0.unicodeScalars.count <= 1000 && !$0.contains("\0") }) else {
            throw Failure("error.instructionsField")
        }
        let combined = [defaults, app].filter { !$0.isEmpty }.joined(separator: "\n")
        guard combined.unicodeScalars.count <= 2000 else {
            throw Failure("error.instructionsCombined")
        }
        return combined
    }
}

struct DictionaryEntry: Codable, Identifiable, Equatable {
    var id = UUID()
    var spoken: String
    var written: String
    var learned = false
    var isValid: Bool { Self.isValidPhrase(spoken) && Self.isValidPhrase(written) }
    static func isValidPhrase(_ text: String) -> Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && text.unicodeScalars.count <= 120 && !text.contains("\0")
    }
}

struct AppRule: Codable, Identifiable, Equatable {
    var id = UUID()
    var bundleID: String
    var name: String
    var style: String
    var instructions: String
}

struct HistoryEntry: Codable, Identifiable, Equatable {
    var id = UUID()
    var date = Date()
    var mode: VoiceMode
    var appName: String
    var rawText: String
    var text: String
    var duration: Double
    var audioFile: String?
    var warning: String?
    var units: Int { Self.countUnits(text) }
    static func countUnits(_ text: String) -> Int {
        func isCJK(_ scalar: Unicode.Scalar) -> Bool {
            (0x3400...0x9FFF).contains(scalar.value) || (0x3040...0x30FF).contains(scalar.value)
                || (0xAC00...0xD7AF).contains(scalar.value)
        }
        let cjk = text.unicodeScalars.filter(isCJK)
        let rest = String(String.UnicodeScalarView(text.unicodeScalars.map { isCJK($0) ? " " : $0 }))
        return cjk.count + rest.split(whereSeparator: { $0.isWhitespace }).count
    }
}

private struct SavedData: Codable {
    var version = 1
    var preferences = Preferences()
    var dictionary: [DictionaryEntry] = []
    var rules: [AppRule] = []
    var history: [HistoryEntry] = []
}

@MainActor
final class AppStore: ObservableObject {
    @Published var preferences = Preferences() {
        didSet {
            L10n.use(preferences.uiLanguage)
            if loaded { prune(); persist() }
        }
    }
    @Published var dictionary: [DictionaryEntry] = [] { didSet { persist() } }
    @Published var rules: [AppRule] = [] { didSet { persist() } }
    @Published var history: [HistoryEntry] = [] { didSet { persist() } }
    @Published var storageError = ""
    let directory: URL
    private var loaded = false
    private var canSave = true
    private var pruning = false
    private var file: URL { directory.appendingPathComponent("library.json") }
    var audioDirectory: URL { directory.appendingPathComponent("Audio", isDirectory: true) }

    init(directory: URL? = nil) {
        self.directory = directory ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("OmniTyper", isDirectory: true)
        do {
            // Note (Codex): Migration must not replace an existing OmniTyper library.
            let legacy = self.directory.deletingLastPathComponent().appendingPathComponent("OpenTypeless", isDirectory: true)
            if self.directory.lastPathComponent == "OmniTyper",
               !FileManager.default.fileExists(atPath: self.directory.path),
               FileManager.default.fileExists(atPath: legacy.path) {
                try FileManager.default.moveItem(at: legacy, to: self.directory)
            }
            try FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            if FileManager.default.fileExists(atPath: file.path) {
                let saved = try JSONDecoder().decode(SavedData.self, from: Data(contentsOf: file))
                guard saved.version == 1 else { throw CocoaError(.coderReadCorrupt) }
                preferences = saved.preferences; dictionary = saved.dictionary
                rules = saved.rules; history = saved.history
            }
        } catch {
            // Note (Codex): An unreadable library must never be overwritten with defaults.
            canSave = false
            storageError = L("error.libraryLoad", error.localizedDescription)
        }
        L10n.use(preferences.uiLanguage)
        loaded = true
        if canSave {
            // Note (Codex): A renamed checkout can invalidate the saved interpreter path.
            let previous = preferences.pythonExecutable
            let relocated = previous.replacingOccurrences(of: "/openTypeless/", with: "/OmniTyper/")
                .replacingOccurrences(of: "/OpenTypeless/", with: "/OmniTyper/")
            if relocated != previous, !FileManager.default.isExecutableFile(atPath: previous),
               FileManager.default.isExecutableFile(atPath: relocated) {
                preferences.pythonExecutable = relocated
            }
            prune(); removeOrphanedAudio()
        }
    }

    func persist() {
        guard loaded, canSave else { return }
        do {
            let data = SavedData(preferences: preferences, dictionary: dictionary, rules: rules, history: history)
            let encoded = try JSONEncoder().encode(data)
            try encoded.write(to: file, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
            storageError = ""
        } catch { storageError = L("error.librarySave", error.localizedDescription) }
    }

    @discardableResult
    func add(_ entry: HistoryEntry, recording: URL?) -> String? {
        guard preferences.saveHistory, canSave else {
            if let recording { try? FileManager.default.removeItem(at: recording) }
            return nil
        }
        var saved = entry
        var retentionError: String?
        if preferences.keepAudio, let recording {
            do {
                try FileManager.default.createDirectory(at: audioDirectory, withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                let name = "\(entry.id.uuidString).wav"
                let destination = audioDirectory.appendingPathComponent(name)
                try FileManager.default.copyItem(at: recording, to: destination)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path)
                saved.audioFile = name
            } catch {
                retentionError = L("error.audioRetain", error.localizedDescription)
                saved.warning = [saved.warning, retentionError].compactMap { $0 }.joined(separator: "\n")
            }
        }
        if retentionError == nil, let recording { try? FileManager.default.removeItem(at: recording) }
        history.insert(saved, at: 0)
        prune()
        if let retentionError { storageError = [storageError, retentionError].filter { !$0.isEmpty }.joined(separator: "\n") }
        return retentionError
    }

    func audioURL(for entry: HistoryEntry) -> URL? {
        guard let name = entry.audioFile, name == "\(entry.id.uuidString).wav" else { return nil }
        let path = audioDirectory.appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: path.path) ? path : nil
    }

    // Note (Jiaxin Deng): History preserves insertion failures that occur while the app window is hidden.
    func note(_ warning: String, on id: UUID) {
        guard canSave, let index = history.firstIndex(where: { $0.id == id }) else { return }
        let existing = history[index].warning ?? ""
        history[index].warning = existing.isEmpty ? warning : existing + " " + warning
    }

    func delete(_ ids: Set<UUID>) {
        for entry in history where ids.contains(entry.id) {
            if let url = audioURL(for: entry) { try? FileManager.default.removeItem(at: url) }
        }
        history.removeAll { ids.contains($0.id) }
    }

    func prune(now: Date = Date()) {
        guard !pruning, canSave else { return }
        pruning = true; defer { pruning = false }
        let cutoff = now.addingTimeInterval(-Double(max(0, preferences.historyDays)) * 86400)
        let expired = history.enumerated().filter { index, item in
            !preferences.saveHistory || index >= 1000 || (preferences.historyDays > 0 && item.date < cutoff)
        }.map { $0.element.id }
        if !expired.isEmpty { delete(Set(expired)) }
        if !preferences.keepAudio {
            for index in history.indices where history[index].audioFile != nil {
                if let url = audioURL(for: history[index]) { try? FileManager.default.removeItem(at: url) }
                history[index].audioFile = nil
            }
        }
    }

    private func removeOrphanedAudio() {
        let referenced = Set(history.compactMap(\.audioFile))
        guard let files = try? FileManager.default.contentsOfDirectory(at: audioDirectory,
                                                                      includingPropertiesForKeys: nil) else { return }
        for file in files where file.pathExtension == "wav" && !referenced.contains(file.lastPathComponent) {
            try? FileManager.default.removeItem(at: file)
        }
    }

    @discardableResult
    func addWord(spoken: String, written: String, learned: Bool = false, replacing id: UUID? = nil) -> Bool {
        let spoken = spoken.trimmingCharacters(in: .whitespacesAndNewlines)
        let written = written.trimmingCharacters(in: .whitespacesAndNewlines)
        guard DictionaryEntry.isValidPhrase(spoken), DictionaryEntry.isValidPhrase(written) else { return false }
        if let id {
            guard let index = dictionary.firstIndex(where: { $0.id == id }),
                  !dictionary.contains(where: { $0.id != id && $0.spoken.caseInsensitiveCompare(spoken) == .orderedSame }) else { return false }
            dictionary[index] = DictionaryEntry(id: id, spoken: spoken, written: written, learned: dictionary[index].learned)
        } else if let index = dictionary.firstIndex(where: { $0.spoken.caseInsensitiveCompare(spoken) == .orderedSame }) {
            dictionary[index].written = written
        } else if dictionary.count < 200 {
            dictionary.append(DictionaryEntry(spoken: spoken, written: written, learned: learned))
        } else { return false }
        return true
    }

    func importWords(_ text: String) throws -> Int {
        let rows = try DictionaryCSV.parse(text)
        var count = 0
        for (index, row) in rows.enumerated() {
            guard let first = row.first, !first.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            if index == 0 && ["spoken", "word", "phrase"].contains(first.lowercased()) { continue }
            let written = row.count > 1 && !row[1].isEmpty ? row[1] : first
            if addWord(spoken: first, written: written) { count += 1 }
        }
        return count
    }
}

enum DictionaryCSV {
    static func parse(_ text: String) throws -> [[String]] {
        guard text.utf8.count <= 1_000_000 else { throw Failure("error.dictionaryTooLarge") }
        var rows: [[String]] = [], row: [String] = [], field = "", quoted = false, endedQuote = false
        let chars = Array(text.replacingOccurrences(of: "\r\n", with: "\n"))
        var index = 0
        while index < chars.count {
            let ch = chars[index]
            if quoted {
                if ch == "\"" {
                    if index + 1 < chars.count && chars[index + 1] == "\"" { field.append("\""); index += 1 }
                    else { quoted = false; endedQuote = true }
                } else { field.append(ch) }
            } else if ch == "," || ch == "\n" {
                row.append(field); field = ""; endedQuote = false
                if ch == "\n" { rows.append(row); row = [] }
            } else if ch == "\"", field.isEmpty, !endedQuote { quoted = true }
            else {
                guard !endedQuote, ch != "\"" else { throw Failure("error.csvQuoting") }
                field.append(ch)
            }
            index += 1
        }
        guard !quoted else { throw Failure("error.csvUnclosed") }
        if !field.isEmpty || !row.isEmpty || endedQuote { row.append(field); rows.append(row) }
        return rows
    }
}

/// Stable diagnostic code with a message resolved in the current interface language.
struct Failure: LocalizedError {
    let code: String
    private let arguments: [String]

    init(_ code: String, _ arguments: String...) {
        self.code = code
        self.arguments = arguments
    }

    var errorDescription: String? {
        arguments.isEmpty ? L(code) : String(format: L10n.string(code), arguments: arguments)
    }
}
