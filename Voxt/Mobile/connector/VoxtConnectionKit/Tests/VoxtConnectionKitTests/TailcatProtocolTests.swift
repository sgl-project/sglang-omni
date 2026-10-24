import XCTest
@testable import VoxtConnectionKit

final class TailcatProtocolTests: XCTestCase {
    private let publicKey = "nodekey:" + String(repeating: "a", count: 64)

    func testConfigurationRetainsOpaqueStateAndExplicitNullOnFirstLaunch() throws {
        struct Configuration: Decodable { let version: Int; let state: String?; let allowedClients: [String]; let localPort: Int }
        for state in [nil, "{\"version\":1,\"fixture\":true}"] {
            let data = try TailcatConfiguration(state: state, allowedClients: [publicKey], localPort: 47920).encoded()
            let decoded = try JSONDecoder().decode(Configuration.self, from: data)
            XCTAssertEqual(decoded.version, 1)
            XCTAssertEqual(decoded.state, state)
            XCTAssertEqual(decoded.allowedClients, [publicKey])
            XCTAssertEqual(decoded.localPort, 47920)
            if state == nil { XCTAssertTrue(String(decoding: data, as: UTF8.self).contains("\"state\":null")) }
        }
    }

    func testEmptyAllowlistBadKeyAndPortAreRejected() {
        for clients in [[], ["banana"], [publicKey, publicKey]] {
            XCTAssertThrowsError(try TailcatConfiguration(state: nil, allowedClients: clients, localPort: 47920).encoded())
        }
        for port in [0, 1023, 65535] {
            XCTAssertThrowsError(try TailcatConfiguration(state: nil, allowedClients: [publicKey], localPort: port).encoded())
        }
    }

    func testReadyRequiresVersionAddressPortAndOpaqueState() throws {
        let ready = try TailcatReady(data: Data(#"{"version":1,"address":"tc-fixture","port":48931,"state":"opaque fixture"}"#.utf8))
        XCTAssertEqual(ready.address, "tc-fixture")
        XCTAssertEqual(ready.port, 48931)
        XCTAssertEqual(ready.state, "opaque fixture")
        for invalid in [#"{}"#, #"{"version":2,"address":"tc-fixture","port":48931,"state":"opaque"}"#, #"{"version":1,"address":"tc-fixture","port":0,"state":"opaque"}"#, #"{"version":1,"address":"tc-fixture","port":48931,"state":{}}"#, #"{"version":1,"address":"tc-fixture","port":48931,"state":"opaque","error":"private native detail"}"#] {
            XCTAssertThrowsError(try TailcatReady(data: Data(invalid.utf8)))
        }
    }
}
