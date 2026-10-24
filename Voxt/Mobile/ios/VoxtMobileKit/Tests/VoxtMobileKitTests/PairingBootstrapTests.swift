import Foundation
import Testing
@testable import VoxtMobileKit

struct PairingBootstrapTests {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)
    private func link(extra: String = "") -> URL {
        URL(string: "voxtmobile://connect?v=1&tailcat=tc-fixture&port=48932&key=privkey:" + String(repeating: "a", count: 64) + "&proof=synthetic-proof&expires=1800000120" + extra)!
    }
    @Test func versionedOneScanAcceptsNativePrivateKeyAndExpiry() throws {
        let bootstrap = try PairingBootstrap(url: link(), now: now)
        #expect(bootstrap.address == "tc-fixture")
        #expect(bootstrap.port == 48932)
        #expect(bootstrap.temporaryKey == "privkey:" + String(repeating: "a", count: 64))
        #expect(bootstrap.expiresAt == now.addingTimeInterval(120))
    }
    @Test func malformedExpiredAndLegacyQRsAreRejected() {
        for text in [link(extra: "&proof=duplicate").absoluteString, link(extra: "&token=forbidden").absoluteString,
                     link().absoluteString.replacingOccurrences(of: "privkey:", with: "nodepriv:"),
                     link().absoluteString.replacingOccurrences(of: "v=1", with: "v=2"),
                     "voxtmobile://connect?tailcat=tc-fixture&port=48931&key=nodepriv:" + String(repeating: "a", count: 64) + "&proof=x"] {
            #expect(throws: PairingBootstrap.Error.self) { try PairingBootstrap(url: URL(string: text)!, now: now) }
        }
        #expect(throws: PairingBootstrap.Error.expired) { try PairingBootstrap(url: link(), now: now.addingTimeInterval(120)) }
    }
}
