import Foundation
import Testing
@testable import VoxtMobileKit

/// Fields the Mac adds to its links later must not break pairing. Known fields stay required and
/// unique, and the QR code still never carries the access token.
struct LinkFieldsTests {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)
    private let qr = "voxtmobile://connect?v=1&tailcat=tc-fixture&port=48932&key=privkey:" + String(repeating: "a", count: 64)
        + "&proof=synthetic-proof&expires=1800000120"
    private let pair = "voxtmobile://pair?tailcat=tc-fixture&port=48931&token=fixture-bearer&client=nodekey:" + String(repeating: "a", count: 64)

    @Test func theQRCodeIgnoresFieldsItDoesNotKnow() throws {
        #expect((try? PairingBootstrap(url: URL(string: qr + "&name=Studio%20Mac&future=1")!, now: now)) != nil)
        #expect(throws: PairingBootstrap.Error.self) { try PairingBootstrap(url: URL(string: qr + "&proof=second")!, now: now) }
        #expect(throws: PairingBootstrap.Error.self) { try PairingBootstrap(url: URL(string: qr + "&token=forbidden")!, now: now) }
    }

    @Test func thePairLinkIgnoresFieldsItDoesNotKnow() throws {
        #expect((try? TailcatInvitation(pairingLink: URL(string: pair + "&future=1&name=Studio%20Mac")!)) != nil)
        #expect(throws: TailcatInvitation.Error.self) { try TailcatInvitation(pairingLink: URL(string: pair + "&token=second")!) }
        #expect(throws: TailcatInvitation.Error.self) {
            try TailcatInvitation(pairingLink: URL(string: pair.replacingOccurrences(of: "&token=fixture-bearer", with: ""))!)
        }
    }
}
