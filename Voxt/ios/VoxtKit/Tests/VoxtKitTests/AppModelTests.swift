import Foundation
import Testing
@testable import VoxtKit
@testable import VoxtUI

@MainActor
@Suite struct AppModelTests {
    func model(_ dir: URL, core: FakeCore = FakeCore(), keys: KeyStore = MemoryKeyStore(), saveDelay: Duration = .zero) -> AppModel {
        AppModel(core: core, keys: keys, recorder: FakeRecorder(), directory: dir, saveDelay: saveDelay, microphoneAllowed: { true })
    }

    @Test func settingsAreSavedOnceTypingPausesAndReadBack() async {
        let dir = scratchDirectory()
        let typing = model(dir, saveDelay: .seconds(3600))
        typing.settings.languageModel.preset = "ollama"
        #expect(model(dir).settings.languageModel.preset == "openai", "not written on every change")
        typing.saveNow()
        #expect(model(dir).settings.languageModel.preset == "ollama", "saveNow writes at once")

        let first = model(dir)
        first.settings.languageModel.preset = "deepseek"
        first.settings.lastFeature = .rewrite
        await first.waitForSave()
        let again = model(dir)
        #expect(again.settings.languageModel.preset == "deepseek")
        #expect(again.session.feature == .rewrite, "the voice screen opens with the last feature")
    }

    @Test func aRefusedKeyReportsTheKeychainStatus() {
        let keys = RefusingKeyStore()
        let m = model(scratchDirectory(), keys: keys)
        #expect(m.saveKey("sk-1", for: KeyAccount.speech) == -34018 && keys.key(KeyAccount.speech) == nil)
        keys.refuse = false
        #expect(m.saveKey("sk-1", for: KeyAccount.speech) == nil && keys.key(KeyAccount.speech) == "sk-1")
    }

    @Test func theSessionReadsTheCurrentSettings() async throws {
        let core = FakeCore()
        let keys = MemoryKeyStore()
        let m = model(scratchDirectory(), core: core, keys: keys)
        m.settings.languageModel = .init(preset: "deepseek", baseURL: "", model: "deepseek-chat")
        #expect(m.saveKey("sk-d", for: KeyAccount.languageModel("deepseek")) == nil)
        m.session.feature = .translate
        m.session.source = "你好"
        m.session.translateTyped()
        await m.session.settled()
        let request = try #require(core.requests.first)
        #expect(request.provider == CoreProvider(preset: "deepseek", model: "deepseek-chat", key: "sk-d"))
        #expect(keys.key("llm.deepseek") == "sk-d")
        m.reloadHistory()
        #expect(m.history.count == 1)
        m.deleteHistory([m.history[0].id])
        #expect(m.history.isEmpty)
    }

    @Test func languageNamesFallBackToTheCode() {
        let m = model(scratchDirectory())
        #expect(m.languageName("xx") == "xx")
    }
}

/// A Keychain without the app's entitlement (errSecMissingEntitlement)
/// while refuse is set.
final class RefusingKeyStore: KeyStore, @unchecked Sendable {
    var refuse = true
    private let keys = MemoryKeyStore()

    func key(_ account: String) -> String? { keys.key(account) }

    func setKey(_ key: String?, for account: String) throws {
        if refuse {
            throw KeychainError(status: -34018)
        }
        try keys.setKey(key, for: account)
    }
}
