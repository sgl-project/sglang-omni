// SPDX-License-Identifier: Apache-2.0
import Testing
import Foundation
@testable import OmniTyper

struct StoreTests {
    @Test @MainActor func textAPIConfigurationKeepsOldLibrariesAndKeysPrivate() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var old = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(Preferences())) as? [String: Any])
        old.removeValue(forKey: "textAPI")
        old["textModel"] = "mlx-community/Qwen3-1.7B-4bit"
        let restored = try JSONDecoder().decode(Preferences.self, from: JSONSerialization.data(withJSONObject: old))
        #expect(restored.textSettings.baseURL == "http://127.0.0.1:11434/v1")
        #expect(restored.textSettings.model.isEmpty)
        var settings = TextAPISettings(model: "my-ollama-model", optionsJSON: "{\"temperature\":0.3}")
        let payload = try settings.payload(apiKey: "secret")
        #expect(payload["text_model"] as? String == "my-ollama-model")
        #expect(payload["text_api_key"] as? String == "secret")
        settings.optionsJSON = "{\"messages\":[]}"
        #expect(throws: (any Error).self) { try settings.payload(apiKey: "") }
        settings.optionsJSON = "{}"
        settings.baseURL = "http://user:secret@localhost/v1"
        #expect(throws: (any Error).self) { try settings.payload(apiKey: "") }

        let store = AppStore(directory: root)
        let model = AppModel(store: store)
        defer { model.shutdown() }
        model.textAPIKey = "session-secret"
        store.preferences.textSettings.model = "my-ollama-model"
        #expect(model.textAPIKey == "session-secret")
        #expect(!String(decoding: try Data(contentsOf: root.appendingPathComponent("library.json")), as: UTF8.self).contains("session-secret"))
        store.preferences.textSettings.baseURL = "https://example.com/v1"
        #expect(model.textAPIKey.isEmpty)
        #expect(AppStore(directory: root).preferences.textSettings.model == "my-ollama-model")
    }

    @Test func huggingFaceEndpointValidationKeepsOldLibrariesAndOfficialDefault() throws {
        var old = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(Preferences())) as? [String: Any])
        old.removeValue(forKey: "hfEndpoint")
        let restored = try JSONDecoder().decode(Preferences.self, from: JSONSerialization.data(withJSONObject: old))
        #expect(restored.huggingFaceEndpoint.isEmpty)
        #expect(try Preferences.validatedHuggingFaceEndpoint("  ") == "")
        #expect(try Preferences.validatedHuggingFaceEndpoint(" https://hf-mirror.com ") == "https://hf-mirror.com")
        for invalid in ["file:///tmp/mirror", "https://user:secret@hf-mirror.com",
                        "https://hf-mirror.com?token=x", "https://hf-mirror.com#fragment",
                        "https://hf mirror.com", "https://hf-mirror.com:99999"] {
            #expect(throws: (any Error).self) { try Preferences.validatedHuggingFaceEndpoint(invalid) }
        }
    }

    @Test @MainActor func prepareModelsRejectsAnInvalidEndpointBeforeStarting() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(store: AppStore(directory: directory))
        defer { model.shutdown() }
        model.store.preferences.hfEndpoint = "file:///tmp/mirror"
        model.prepareModels()
        #expect(!model.error.isEmpty)
        #expect(!model.isBusy)
    }

    /// A failed insertion happens in another app, so its notice lands in a window
    /// the user is not looking at. History has to carry the reason instead.
    @Test @MainActor func insertionFailuresAreRecordedOnTheSavedEntry() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AppStore(directory: root)
        let entry = HistoryEntry(mode: .dictate, appName: "Claude", rawText: "hi", text: "Hi.", duration: 1)
        _ = store.add(entry, recording: nil)
        store.note("The focused field changed.", on: entry.id)
        #expect(store.history.first?.warning == "The focused field changed.")
        store.note("And again.", on: entry.id)
        #expect(store.history.first?.warning == "The focused field changed. And again.")
        store.note("ignored", on: UUID())
        #expect(store.history.count == 1)
    }

    @Test @MainActor func renamedAppMigratesLibraryAndRuntimeWithoutOverwritingNewData() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let legacy = root.appendingPathComponent("Library/OpenTypeless")
        let current = root.appendingPathComponent("Library/OmniTyper")
        let oldPython = root.appendingPathComponent("openTypeless/.venv/bin/python")
        let python = root.appendingPathComponent("OmniTyper/.venv/bin/python")
        try FileManager.default.createDirectory(at: python.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: python)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: python.path)
        let original = AppStore(directory: legacy)
        original.preferences.pythonExecutable = oldPython.path
        original.preferences.keepAudio = true
        original.preferences.historyDays = 0
        original.addWord(spoken: "S G Lang", written: "SGLang")
        let recording = root.appendingPathComponent("recording.wav")
        let audio = Data([1, 2, 3])
        try audio.write(to: recording)
        original.add(HistoryEntry(mode: .dictate, appName: "Test", rawText: "hello", text: "Hello.", duration: 1), recording: recording)

        let migrated = AppStore(directory: current)
        #expect(migrated.storageError.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: legacy.path))
        #expect(migrated.preferences.pythonExecutable == python.path)
        #expect(migrated.dictionary == original.dictionary)
        #expect(migrated.history == original.history)
        #expect(try Data(contentsOf: #require(migrated.audioURL(for: migrated.history[0]))) == audio)
        #expect(AppStore(directory: current).preferences.pythonExecutable == python.path)

        // Reappearing legacy data must not replace the new library or a custom runtime.
        AppStore(directory: legacy).addWord(spoken: "old", written: "Old")
        let legacyFile = legacy.appendingPathComponent("library.json")
        let legacyBytes = try Data(contentsOf: legacyFile)
        migrated.preferences.pythonExecutable = "/custom/runtime/bin/python"
        let reloaded = AppStore(directory: current)
        #expect(reloaded.dictionary == migrated.dictionary)
        #expect(reloaded.preferences.pythonExecutable == "/custom/runtime/bin/python")
        #expect(try Data(contentsOf: legacyFile) == legacyBytes)
    }

    @Test func csvHandlesQuotedCommasNewlinesAndRejectsCorruption() throws {
        let parsed = try DictionaryCSV.parse("spoken,written\n\"a,b\",\"line 1\nline 2\"\n\"a\"\"b\",c\n")
        #expect(parsed == [["spoken", "written"], ["a,b", "line 1\nline 2"], ["a\"b", "c"]])
        #expect(throws: (any Error).self) { try DictionaryCSV.parse("\"unfinished") }
        #expect(throws: (any Error).self) { try DictionaryCSV.parse("\"quoted\"junk,word") }
    }

    @Test @MainActor func persistenceRetentionAndPrivacyDeleteAudio() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        store.preferences.keepAudio = true
        let audio = directory.appendingPathComponent("temp.wav")
        try Data([1, 2, 3]).write(to: audio)
        let entry = HistoryEntry(mode: .dictate, appName: "Test", rawText: "hello", text: "Hello.", duration: 1)
        store.add(entry, recording: audio)
        #expect(!FileManager.default.fileExists(atPath: audio.path))
        let saved = try #require(store.audioURL(for: store.history[0]))
        #expect(FileManager.default.fileExists(atPath: saved.path))
        let reloaded = AppStore(directory: directory)
        #expect(reloaded.history[0].text == "Hello.")
        reloaded.preferences.keepAudio = false
        #expect(!FileManager.default.fileExists(atPath: saved.path))
        #expect(reloaded.history[0].audioFile == nil)
        reloaded.history[0].date = Date(timeIntervalSinceNow: -86400 * 40)
        reloaded.prune()
        #expect(reloaded.history.isEmpty)
        reloaded.add(entry, recording: nil)
        reloaded.preferences.saveHistory = false
        #expect(reloaded.history.isEmpty)
    }

    @Test @MainActor func corruptLibraryIsNotOverwritten() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("library.json")
        let badData = Data("not json".utf8)
        try badData.write(to: file)
        let store = AppStore(directory: directory)
        #expect(!store.storageError.isEmpty)
        store.addWord(spoken: "hello", written: "Hello")
        #expect(try Data(contentsOf: file) == badData)
    }

    @Test @MainActor func dictionaryImportDeduplicatesAndIgnoresHeader() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        _ = try store.importWords("spoken,written\nS G Lang,SGLang\nCUDA\n")
        store.addWord(spoken: "s g lang", written: "SGLang", learned: true)
        #expect(store.dictionary.count == 2)
        #expect(store.dictionary[0].written == "SGLang")
        #expect(store.dictionary[1].written == "CUDA")
    }

    @Test func wordCountIncludesCJK() {
        #expect(HistoryEntry.countUnits("你好 Swift world") == 4)
        #expect(HistoryEntry.countUnits("") == 0)
    }

    @Test @MainActor func invalidDictionaryEditsDoNotReplaceSavedWords() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        #expect(store.addWord(spoken: "S G Lang", written: "SGLang", learned: true))
        let entry = try #require(store.dictionary.first)
        for invalid in ["", "   ", "a\0b", String(repeating: "e\u{301}", count: 61)] {
            #expect(!store.addWord(spoken: invalid, written: "SGLang", replacing: entry.id))
            #expect(store.dictionary == [entry])
        }
        #expect(store.addWord(spoken: "s g lang", written: "SGLang-Omni", replacing: entry.id))
        #expect(store.dictionary[0].id == entry.id)
        #expect(store.dictionary[0].learned)
        #expect(AppStore(directory: directory).dictionary[0].written == "SGLang-Omni")
        #expect(try store.importWords("bad,\"\"\ninvalid,a\0b") == 1)
    }

    @Test func writingInstructionsHonorUnicodeAndCombinedLimit() throws {
        let thousand = String(repeating: "a", count: 1000)
        let accepted = try Preferences.combinedInstructions(thousand, String(repeating: "b", count: 999))
        #expect(accepted.unicodeScalars.count == 2000)
        for (defaults, app) in [(thousand, thousand), (thousand + "a", ""), ("x\0y", ""), (String(repeating: "e\u{301}", count: 501), "")] {
            #expect(throws: (any Error).self) { try Preferences.combinedInstructions(defaults, app) }
        }
    }

    @Test @MainActor func failedAudioRetentionPreservesRecordingAndWarning() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppStore(directory: directory)
        store.preferences.keepAudio = true
        try Data().write(to: store.audioDirectory) // A regular file prevents directory creation.
        let audio = directory.appendingPathComponent("recording.wav")
        try Data([1, 2, 3]).write(to: audio)
        let entry = HistoryEntry(mode: .dictate, appName: "Test", rawText: "hello", text: "Hello.", duration: 1)
        let warning = try #require(store.add(entry, recording: audio))
        #expect(FileManager.default.fileExists(atPath: audio.path))
        #expect(store.storageError.contains(warning))
        #expect(store.history[0].warning == warning)
        #expect(store.history[0].audioFile == nil)
        #expect(AppStore(directory: directory).history[0].warning == warning)
    }

    @Test @MainActor func retryUsesUpdatedRuntimeAndOriginalMode() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = AppStore(directory: directory)
        store.preferences.keepAudio = true
        let originalPython = directory.appendingPathComponent("missing-original-python").path
        let correctedPython = directory.appendingPathComponent("missing-corrected-python").path
        store.preferences.pythonExecutable = originalPython
        store.preferences.textSettings.model = "test-model"
        let audio = directory.appendingPathComponent("recording.wav")
        try Data([1, 2, 3]).write(to: audio)
        store.add(HistoryEntry(mode: .dictate, appName: "Original app", rawText: "hello", text: "Hello.", duration: 42), recording: audio)
        let model = AppModel(store: store)
        defer { model.shutdown(); try? FileManager.default.removeItem(at: directory) }
        model.retry(try #require(store.history.first))
        for _ in 0..<100 where model.isBusy { try await Task.sleep(nanoseconds: 10_000_000) }
        try #require(model.canRetry)
        #expect(model.error.contains(originalPython))
        store.preferences.pythonExecutable = correctedPython
        model.mode = .ask
        model.retryLast()
        for _ in 0..<100 where model.isBusy { try await Task.sleep(nanoseconds: 10_000_000) }
        #expect(model.canRetry)
        #expect(model.error.contains(correctedPython))
        #expect(model.mode == .dictate)
        #expect(model.lastApp == "Original app")
    }
}
