import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct TailcatInvitationTests {
    private let publicKey = "nodekey:" + String(repeating: "a", count: 64)
    private func link(_ extra: [URLQueryItem] = [], port: String = "48931", client: String? = nil) -> URL {
        var url = URLComponents(string: "voxtmobile://pair")!
        url.queryItems = [
            .init(name: "tailcat", value: "tcAQID_test-address"),
            .init(name: "port", value: port),
            .init(name: "token", value: "fixture-bearer"),
            .init(name: "client", value: client ?? publicKey)
        ] + extra
        return url.url!
    }
    @Test func invitationUsesOnlyThisPhonesPrivateKey() throws {
        let invitation = try TailcatInvitation(pairingLink: link())
        let config = try invitation.server(clientKey: "privkey:phone-fixture", publicKey: publicKey)
        #expect(config.token == "fixture-bearer")
        #expect(config.tailcat?.address == "tcAQID_test-address")
        #expect(config.tailcat?.remotePort == 48931)
        #expect(config.tailcat?.clientKey == "privkey:phone-fixture")
        #expect(config.baseURL.host == "tailcat.invalid")
    }
    @Test func differentPhoneCannotConsumeAnInvitation() throws {
        let invitation = try TailcatInvitation(pairingLink: link())
        #expect(throws: TailcatInvitation.Error.differentDevice) {
            try invitation.server(clientKey: "privkey:other", publicKey: "nodekey:" + String(repeating: "b", count: 64))
        }
    }
    @Test func duplicateMixedMissingAndInvalidFieldsAreRejected() {
        #expect(throws: TailcatInvitation.Error.invalidLink) { try TailcatInvitation(pairingLink: link([.init(name: "port", value: "48932")])) }
        // Unknown fields are ignored (so the Mac can add fields later) and never used: the server
        // stays the tailcat address and the key stays this phone's own.
        for extra in [[URLQueryItem(name: "url", value: "https://elsewhere.invalid")], [.init(name: "clientKey", value: "privkey:from-link")]] {
            let config = try? TailcatInvitation(pairingLink: link(extra)).server(clientKey: "privkey:phone-fixture", publicKey: publicKey)
            #expect(config?.baseURL.host == "tailcat.invalid"); #expect(config?.tailcat?.clientKey == "privkey:phone-fixture")
        }
        for port in ["0", "65536", "-1", "banana", ""] {
            #expect(throws: TailcatInvitation.Error.invalidLink) { try TailcatInvitation(pairingLink: link(port: port)) }
        }
        for client in ["", "nodekey:short", "privkey:" + String(repeating: "a", count: 64)] {
            #expect(throws: TailcatInvitation.Error.invalidLink) { try TailcatInvitation(pairingLink: link(client: client)) }
        }
        for url in [URL(string: "https://pair?tailcat=x")!, URL(string: "voxtmobile://pair?tailcat=x")!] {
            #expect(throws: TailcatInvitation.Error.invalidLink) { try TailcatInvitation(pairingLink: url) }
        }
    }
}
