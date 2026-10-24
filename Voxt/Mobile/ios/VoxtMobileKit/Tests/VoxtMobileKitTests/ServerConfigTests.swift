import Foundation
import Testing
@testable import VoxtMobileKit

@Suite struct ServerConfigTests {
    @Test func bareHostGetsHTTPS() throws {
        let url = try ServerConfig.normalizeBaseURL("mac.example.test:8900")
        #expect(url.absoluteString == "https://mac.example.test:8900")
    }

    @Test func trimsWhitespaceAndTrailingSlash() throws {
        let url = try ServerConfig.normalizeBaseURL("  https://mac.example.test:8900/  ")
        #expect(url.absoluteString == "https://mac.example.test:8900")
    }

    @Test func keepsExplicitHTTPForLoopback() throws {
        let url = try ServerConfig.normalizeBaseURL("http://127.0.0.1:47910")
        #expect(url.absoluteString == "http://127.0.0.1:47910")
    }

    @Test func rejectsEmptyAndGarbage() {
        #expect(throws: ServerConfig.Error.invalidAddress) { try ServerConfig.normalizeBaseURL("   ") }
        #expect(throws: ServerConfig.Error.invalidAddress) { try ServerConfig.normalizeBaseURL("ftp://x") }
        #expect(throws: ServerConfig.Error.invalidAddress) { try ServerConfig.normalizeBaseURL("http://") }
    }

}
