import XCTest
import VoxtMobileKit
@testable import VoxtMobile

private final class FeatureTokens: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}

/// Settings hides Developer prompts, correction learning and history sync when the Mac's
/// capabilities leave their endpoints out; a Mac that lists no endpoints keeps all three.
@MainActor
final class AppModelMacFeaturesTests: XCTestCase {
    private static let listing = #"{"api":{"major":1,"minor":0},"endpoints":["health","capabilities","dictations","compose","compose/prompts"],"locations":{"transcription":"local","text":"local"}}"#
    private static let legacy = #"{"transcription":"local","text":"local"}"#

    private func model(capabilities json: String, defaults: UserDefaults) throws -> AppModel {
        let tokens = FeatureTokens()
        try PairingStore(defaults: defaults, tokens: tokens).save(.init(baseURL: URL(string: "https://mac.example.test")!, token: "fake-token"))
        let health = try JSONDecoder().decode(VoxtClient.Health.self, from: Data(#"{"ok":true,"app":"Voxt","version":"1.16.0"}"#.utf8))
        let capabilities = try JSONDecoder().decode(VoxtClient.Capabilities.self, from: Data(json.utf8))
        return AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens,
                        injectedHealthProbe: { _ in health }, injectedCapabilitiesProbe: { _ in capabilities })
    }

    private func withDefaults(_ body: (UserDefaults) async throws -> Void) async throws {
        let name = "mac-features.\(UUID())"
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        try await body(defaults)
    }

    func testBeforeTheMacAnswersEveryFeatureShows() async throws {
        try await withDefaults { defaults in
            let model = try model(capabilities: Self.listing, defaults: defaults)
            XCTAssertTrue(VoxtClient.Feature.allCases.allSatisfy(model.macServes))
        }
    }

    func testFeaturesTheMacDoesNotListAreHiddenAndTheKeyboardIsTold() async throws {
        try await withDefaults { defaults in
            let model = try model(capabilities: Self.listing, defaults: defaults)
            await model.checkConnection(attempts: 1)
            XCTAssertTrue(model.macServes(.composePrompts))
            XCTAssertFalse(model.macServes(.corrections))
            XCTAssertFalse(model.macServes(.historySync))
            XCTAssertFalse(MacFeatures.serves(.corrections, in: defaults), "the keyboard reads the same list")
        }
    }

    func testAMacThatListsNoEndpointsKeepsEveryFeature() async throws {
        try await withDefaults { defaults in
            let model = try model(capabilities: Self.legacy, defaults: defaults)
            await model.checkConnection(attempts: 1)
            XCTAssertTrue(VoxtClient.Feature.allCases.allSatisfy(model.macServes))
            XCTAssertTrue(MacFeatures.serves(.historySync, in: defaults))
        }
    }
}
