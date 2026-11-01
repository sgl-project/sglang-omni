import XCTest
import VoxtMobileKit
import VoxtTailcat
@testable import VoxtMobile

private final class InvitationTokens: TokenStoring {
    var value: String?
    func read() throws -> String? { value }
    func write(_ value: String) throws { self.value = value }
    func delete() throws { value = nil }
}

@MainActor final class AppModelTailcatInvitationTests: XCTestCase {
    func testOldScannedPairingLinkDoesNotAuthorizePhone() throws {
        let name = "invitation-test.\(UUID())", tokens = InvitationTokens()
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let model = AppModel(injectedCapture: ScriptedCapturePublic(), injectedDefaults: defaults, injectedTokens: tokens)
        model.handle(url: URL(string: "voxtmobile://pair?url=http://127.0.0.1:9&token=fixture")!)
        XCTAssertNil(model.server)
        XCTAssertNil(tokens.value)
    }
}
