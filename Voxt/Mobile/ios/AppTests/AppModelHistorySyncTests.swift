import XCTest
@testable import VoxtMobile
import VoxtMobileKit

private final class SyncTestTokens: TokenStoring {
    var token: String?
    func read() throws -> String? { token }
    func write(_ value: String) throws { token = value }
    func delete() throws { token = nil }
}

@MainActor
private struct HistoryIdentityTranscriber: Transcribing {
    let id: UUID
    func transcribe(wav: Data) async throws -> (text: String, rawText: String) { ("Synced result", "Synced result") }
    func transcribeResult(wav: Data, options: CompositionOptions) async throws -> TranscriptionResult {
        .init(id: id, text: "Synced result", rawText: "Synced result")
    }
}

@MainActor
final class AppModelHistorySyncTests: XCTestCase {
    func testDictationUsesMacHistoryIDAndUnpairDisablesSync() async throws {
        let name = "app-sync.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = SyncTestTokens()
        try PairingStore(defaults: defaults, tokens: tokens).save(ServerConfig(baseURL: URL(string: "https://fixture.invalid")!, token: "fixture"))
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        let id = UUID()
        model.controller.setTranscriber(HistoryIdentityTranscriber(id: id))
        try model.controller.start()
        await model.controller.stop()
        XCTAssertEqual(model.history.first?.id, id)
        model.setHistorySyncEnabled(true)
        XCTAssertTrue(model.historySyncEnabled)
        XCTAssertEqual(HistorySyncStore(defaults: defaults).makeRequest()?.items.map(\.id), [id])
        model.unpair()
        XCTAssertFalse(model.historySyncEnabled)
        XCTAssertNil(HistorySyncStore(defaults: defaults).makeRequest())
    }

    func testHistoryOffDisablesSyncAndSyncRefreshDoesNotCountAsDictation() async throws {
        let name = "app-sync.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let tokens = SyncTestTokens()
        try PairingStore(defaults: defaults, tokens: tokens).save(ServerConfig(baseURL: URL(string: "https://fixture.invalid")!, token: "fixture"))
        let id = UUID(), serverID = UUID()
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens,
                             injectedHistoryExchange: { _ in
            .init(storeID: serverID, cursor: 1, hasMore: false,
                  items: [.init(id: id, text: "Mac history", createdAt: Date().timeIntervalSince1970)], deletedIDs: [])
        })
        model.setHistorySyncEnabled(true)
        await model.syncHistory()
        XCTAssertEqual(model.history.map(\.id), [id])
        XCTAssertEqual(model.appResultCount, 0)
        model.setHistoryRetention(.off)
        XCTAssertFalse(model.historySyncEnabled)
        XCTAssertTrue(model.history.isEmpty)
    }
}
