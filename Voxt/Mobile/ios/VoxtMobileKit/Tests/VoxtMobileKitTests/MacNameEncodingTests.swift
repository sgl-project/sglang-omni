import Foundation
import Testing
@testable import VoxtMobileKit

/// The links the Mac's helper writes (tailcat/host/testdata/pair-links.json, kept current by
/// the helper's Go test) must give back the Mac's name exactly: spaces, '+', apostrophes, CJK.
struct MacNameEncodingTests {
    private struct Sample: Decodable { let name: String; let link: String }

    @Test func theMacsNameSurvivesTheHelpersLink() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "tailcat/host/testdata/pair-links.json")
        let samples = try JSONDecoder().decode([Sample].self, from: Data(contentsOf: fixture))
        #expect(samples.count == 3)
        for sample in samples {
            let invitation = try TailcatInvitation(pairingLink: try #require(URL(string: sample.link)))
            let server = try invitation.server(clientKey: "privkey:phone", publicKey: "nodekey:" + String(repeating: "a", count: 64))
            #expect(server.name == sample.name)
            #expect(server.token == "owned+token")
        }
    }
}
