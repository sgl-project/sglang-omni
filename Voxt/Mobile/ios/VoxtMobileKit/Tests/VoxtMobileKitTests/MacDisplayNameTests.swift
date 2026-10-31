import Foundation
import Testing
@testable import VoxtMobileKit

/// "tailcat.invalid", the placeholder host of a Tailcat pairing, is not a name to show as
/// the Mac's. Show a name the Mac sent, or "Your Mac".
@Suite struct MacDisplayNameTests {
    private let publicKey = "nodekey:" + String(repeating: "a", count: 64)
    private func link(name: String? = nil) -> URL {
        var url = URLComponents(string: "voxtmobile://pair")!
        url.queryItems = [
            .init(name: "tailcat", value: "tcAQID_test-address"), .init(name: "port", value: "48931"),
            .init(name: "token", value: "fixture-bearer"), .init(name: "client", value: publicKey),
        ] + (name.map { [.init(name: "name", value: $0)] } ?? [])
        return url.url!
    }

    @Test func tailcatPairingNeverShowsThePlaceholderHost() throws {
        let config = try TailcatInvitation(pairingLink: link()).server(clientKey: "privkey:p", publicKey: publicKey)
        #expect(config.baseURL.host == "tailcat.invalid", "the transport placeholder itself is unchanged")
        #expect(config.displayName == nil)
    }

    @Test func aNameFromTheMacIsKeptAndShown() throws {
        let config = try TailcatInvitation(pairingLink: link(name: "  Studio Mac  ")).server(clientKey: "privkey:p", publicKey: publicKey)
        #expect(config.displayName == "Studio Mac")
        let stored = try JSONDecoder().decode(ServerConfig.self, from: JSONEncoder().encode(config))
        #expect(stored.displayName == "Studio Mac")
        let long = try TailcatInvitation(pairingLink: link(name: String(repeating: "m", count: 200))).server(clientKey: "privkey:p", publicKey: publicKey)
        #expect(long.displayName?.count == 64)
    }

    /// The name can arrive form-encoded (spaces as "+"). Query values are
    /// percent-decoded only: %20 is a space and a literal "+" stays "+" (no form decoding).
    @Test func nameIsPercentDecodedButPlusStaysLiteral() throws {
        let base = "voxtmobile://pair?tailcat=tcAQID_test-address&port=48931&token=fixture-bearer&client=\(publicKey)"
        let spaced = try TailcatInvitation(pairingLink: #require(URL(string: base + "&name=Studio%20Mac%20mini")))
            .server(clientKey: "privkey:p", publicKey: publicKey)
        #expect(spaced.displayName == "Studio Mac mini")
        let plus = try TailcatInvitation(pairingLink: #require(URL(string: base + "&name=C%2B%2B+Mac")))
            .server(clientKey: "privkey:p", publicKey: publicKey)
        #expect(plus.displayName == "C++ Mac".replacingOccurrences(of: " ", with: "+"))
    }

    @Test func olderStoredConfigsWithoutANameStillDecode() throws {
        let old = #"{"baseURL":"http:\/\/tailcat.invalid","token":"t","tailcat":{"address":"tc1","clientKey":"k","remotePort":1}}"#
        let config = try JSONDecoder().decode(ServerConfig.self, from: Data(old.utf8))
        #expect(config.displayName == nil)
    }

    @Test func aHostIsNeverTakenForTheMacsName() throws {
        #expect(try ServerConfig(baseURL: #require(URL(string: "https://studio.tail1.ts.net:8900")), token: "t").displayName == nil)
        #expect(try ServerConfig(baseURL: #require(URL(string: "http://127.0.0.1:8900")), token: "t").displayName == nil)
    }
}
