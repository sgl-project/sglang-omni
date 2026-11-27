import Foundation
import Testing
@testable import VoxtKit

@Suite struct StoresTests {
    @Test func wavHeaderAndSamples() {
        let data = WAV.encode([0, 1, -1, 2], sampleRate: 16_000)
        #expect(data.count == 44 + 8)
        #expect(String(decoding: data[0..<4], as: UTF8.self) == "RIFF")
        #expect(String(decoding: data[8..<16], as: UTF8.self) == "WAVEfmt ")
        let rate = data[24..<28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        #expect(UInt32(littleEndian: rate) == 16_000)
        let samples = data[44...].withUnsafeBytes { Array($0.bindMemory(to: Int16.self)) }
        #expect(samples == [0, Int16.max, -Int16.max, Int16.max], "clipped to -1...1")
        #expect(Recording(samples: [Float](repeating: 0, count: 8_000)).duration == 0.5)
    }

    @Test func settingsRoundTripAndABrokenFileIsKeptAside() throws {
        let dir = scratchDirectory()
        let store = SettingsStore(directory: dir)
        #expect(store.load() == AppSettings())
        var s = AppSettings()
        s.general.mainLanguage = "zh"
        s.languageModel = .init(preset: "deepseek", baseURL: "", model: "deepseek-chat")
        s.dictionary.terms = [Term(term: "SGLang", aliases: ["s g lang"])]
        try store.save(s)
        let loaded = store.load()
        #expect(loaded.general.mainLanguage == "zh" && loaded.languageModel.preset == "deepseek")
        #expect(loaded.dictionary.terms.map(\.term) == ["SGLang"])
        try Data("{".utf8).write(to: dir.appendingPathComponent("settings.json"))
        #expect(store.load() == AppSettings())
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("settings.broken.json").path))
    }

    @Test func aDocumentMissingFieldsKeepsTheOthers() throws {
        let dir = scratchDirectory()
        // An older version's document: no speech settings, a dictionary without notes.
        let old = #"{"languageModel":{"preset":"ollama"},"dictionary":{"terms":[{"term":"Voxt"}]},"lastFeature":"rewrite"}"#
        try Data(old.utf8).write(to: dir.appendingPathComponent("settings.json"))
        let s = SettingsStore(directory: dir).load()
        #expect(s.languageModel.preset == "ollama" && s.languageModel.model.isEmpty)
        #expect(s.dictionary.terms.map(\.term) == ["Voxt"] && s.dictionary.enabled)
        #expect(s.speech == AppSettings.Speech() && s.lastFeature == .rewrite)

        // A field held as null keeps its default, not the whole document reset.
        let nulls = #"{"speech":{"endpoint":null,"model":"whisper-1"},"general":{"mainLanguage":"ja"}}"#
        try Data(nulls.utf8).write(to: dir.appendingPathComponent("settings.json"))
        let n = SettingsStore(directory: dir).load()
        #expect(n.speech.endpoint.isEmpty && n.speech.model == "whisper-1" && n.general.mainLanguage == "ja")
    }

    @Test func mainLanguageFollowsTheDevice() {
        var s = AppSettings()
        #expect(s.mainLanguage(preferred: ["zh-Hans-CN"]) == "zh")
        #expect(s.mainLanguage(preferred: []) == "en")
        s.general.mainLanguage = "ja"
        #expect(s.mainLanguage(preferred: ["zh-Hans-CN"]) == "ja")
        s.dictionary.enabled = false
        s.dictionary.terms = [Term(term: "x")]
        s.dictionary.notes = "notes"
        #expect(s.activeTerms.isEmpty && s.activeNotes == nil)
    }

    @Test func historyIsNewestFirstAndBounded() throws {
        let store = HistoryStore(directory: scratchDirectory(), limit: 2)
        let a = HistoryEntry(feature: .transcribe, speech: "a", result: "A")
        let b = HistoryEntry(feature: .translate, speech: "b", result: "B")
        let c = HistoryEntry(feature: .rewrite, speech: "c", result: "C")
        try store.add(a)
        try store.add(b)
        try store.add(c)
        #expect(store.entries().map(\.result) == ["C", "B"])
        try store.delete([c.id])
        #expect(store.entries().map(\.result) == ["B"])
        try store.clear()
        #expect(store.entries().isEmpty)
    }

    @Test func memoryKeysAndAccounts() throws {
        let keys = MemoryKeyStore()
        try keys.setKey("sk-1", for: KeyAccount.languageModel("openai"))
        #expect(keys.key("llm.openai") == "sk-1")
        try keys.setKey("", for: "llm.openai")
        #expect(keys.key("llm.openai") == nil)
        #expect(KeyAccount.speech == "asr.remote")
    }

    @Test func recognizerLocales() {
        let us = Locale(identifier: "en_US")
        #expect(AppleSpeechEngine.localeIdentifier("en", current: us) == "en-US")
        #expect(AppleSpeechEngine.localeIdentifier("zh", current: us) == "zh-CN")
        #expect(AppleSpeechEngine.localeIdentifier("fr", current: us) == "fr")
        #expect(AppleSpeechEngine.localeIdentifier(nil, current: us) == "en_US")
    }

    @Test func meterLevel() {
        #expect(MicRecorder.level([]) == 0)
        #expect(MicRecorder.level([1, -1]) == 1)
        #expect(MicRecorder.level([0.0001]) == 0)
    }

    @Test func remoteEngineSendsTheWAVThroughTheCore() async throws {
        let core = FakeCore()
        let engine = RemoteSpeechEngine(core: core, endpoint: "", model: "", key: "k", scratch: scratchDirectory())
        let text = try await engine.transcribe(SpeechInput(recording: Recording(samples: [0, 0]), language: "en", terms: []))
        #expect(text == "remote")
    }
}
