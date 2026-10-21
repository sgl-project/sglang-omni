// LocalAPIConnectorLinkTests.swift
// Voxt hands its port and Access token to the signed Voxt Connector over a Unix socket,
// after one "Allow".

import XCTest
@testable import Voxt

final class LocalAPIConnectorLinkTests: XCTestCase {
    private func pair() -> (connector: Int32, voxt: Int32) {
        var fds: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds), 0)
        return (fds[0], fds[1])
    }

    private func exchange(_ server: LocalAPIConnectorLink) async -> [String] {
        let (connector, voxt) = pair()
        _ = #"{"v":1,"op":"connect"}"#.appending("\n").withCString { write(connector, $0, strlen($0)) }
        await server.handle(voxt)
        var lines: [String] = []
        var buffer = [UInt8](repeating: 0, count: 4096)
        var timeout = timeval(tv_sec: 1, tv_usec: 0)
        setsockopt(connector, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        let count = read(connector, &buffer, buffer.count)
        if count > 0 { lines = String(decoding: buffer[0..<count], as: UTF8.self).split(separator: "\n").map(String.init) }
        close(connector)
        return lines
    }

    func testRememberedSignedConnectorGetsCredentialsWithoutADialog() async {
        var asked = false
        let server = LocalAPIConnectorLink(verifyPeer: { _ in true }, isRemembered: { true }, remember: {},
                                         ask: { asked = true; return .deny }, credentials: { (47920, "fixture-token") })
        let lines = await exchange(server)
        XCTAssertEqual(lines, [#"{"alwaysAllowed":true,"port":47920,"status":"approved","token":"fixture-token","v":1}"#])
        XCTAssertFalse(asked)
    }

    func testFirstConnectionAsksOnceAndCanBeRemembered() async {
        var remembered = false
        let server = LocalAPIConnectorLink(verifyPeer: { _ in true }, isRemembered: { remembered }, remember: { remembered = true },
                                         ask: { .allowAlways }, credentials: { (47920, "fixture-token") })
        let lines = await exchange(server)
        XCTAssertEqual(lines.first, #"{"status":"pending","v":1}"#)
        XCTAssertEqual(lines.last, #"{"alwaysAllowed":true,"port":47920,"status":"approved","token":"fixture-token","v":1}"#)
        XCTAssertTrue(remembered)
        let once = LocalAPIConnectorLink(verifyPeer: { _ in true }, isRemembered: { false }, remember: {},
                                       ask: { .allowOnce }, credentials: { (47920, "fixture-token") })
        let onceLines = await exchange(once)
        XCTAssertEqual(onceLines.last, #"{"alwaysAllowed":false,"port":47920,"status":"approved","token":"fixture-token","v":1}"#,
                       "the Connector stores its launch flag only for Always Allow")
    }

    func testDeniedOrUnsignedConnectorGetsNoToken() async {
        let denied = LocalAPIConnectorLink(verifyPeer: { _ in true }, isRemembered: { false }, remember: {},
                                         ask: { .deny }, credentials: { (47920, "fixture-token") })
        let deniedLines = await exchange(denied)
        XCTAssertEqual(deniedLines.last, #"{"status":"denied","v":1}"#)
        XCTAssertFalse(deniedLines.joined().contains("fixture-token"))
        var asked = false
        let unsigned = LocalAPIConnectorLink(verifyPeer: { _ in false }, isRemembered: { true }, remember: {},
                                           ask: { asked = true; return .allowAlways }, credentials: { (47920, "fixture-token") })
        let unsignedLines = await exchange(unsigned)
        XCTAssertEqual(unsignedLines, [#"{"reason":"signature","status":"rejected","v":1}"#])
        XCTAssertFalse(asked, "no dialog for a process that isn't the signed Connector")
    }

    func testSocketIsPrivateToTheUser() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("cl-\(UUID().uuidString.prefix(8))")
        let server = LocalAPIConnectorLink(path: directory.appendingPathComponent("c.sock").path, verifyPeer: { _ in false },
                                         isRemembered: { false }, remember: {}, ask: { .deny }, credentials: { nil })
        try server.start()
        defer { server.stop(); try? FileManager.default.removeItem(at: directory) }
        let socketMode = try FileManager.default.attributesOfItem(atPath: directory.appendingPathComponent("c.sock").path)[.posixPermissions] as? Int
        let directoryMode = try FileManager.default.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int
        XCTAssertEqual(socketMode, 0o600)
        XCTAssertEqual(directoryMode, 0o700)
    }

    func testRequirementUsesVoxtsOwnTeamAndTheConnectorBundle() {
        XCTAssertEqual(LocalAPIConnectorLink.connectorRequirement(team: "ABCDE12345"),
                       #"anchor apple generic and certificate leaf[subject.OU] = "ABCDE12345" and identifier "com.voxt.VoxtConnector""#)
        XCTAssertEqual(LocalAPIConnectorLink.connectorRequirement(team: "ABCDE12345", identifier: "org.example.Connector"),
                       #"anchor apple generic and certificate leaf[subject.OU] = "ABCDE12345" and identifier "org.example.Connector""#)
    }

    func testInfoPlistCanNameARenamedConnector() {
        XCTAssertEqual(LocalAPIConnectorLink.connectorBundleIdentifier(info: nil), "com.voxt.VoxtConnector")
        XCTAssertEqual(LocalAPIConnectorLink.connectorBundleIdentifier(info: [:]), "com.voxt.VoxtConnector")
        XCTAssertEqual(LocalAPIConnectorLink.connectorBundleIdentifier(info: ["VoxtConnectorBundleIdentifier": "org.example.Connector"]),
                       "org.example.Connector")
        // The value goes into a code requirement, where a quote would end the identifier string.
        for bad: Any in [#"x" or identifier "y"#, "", 7] {
            XCTAssertEqual(LocalAPIConnectorLink.connectorBundleIdentifier(info: ["VoxtConnectorBundleIdentifier": bad]),
                           "com.voxt.VoxtConnector")
        }
    }

    func testUnsignedVoxtRejectsEveryConnectorWithoutADialog() async {
        var asked = false
        let server = LocalAPIConnectorLink(team: nil, isRemembered: { true }, remember: {},
                                         ask: { asked = true; return .allowAlways }, credentials: { (47920, "fixture-token") })
        let lines = await exchange(server)
        XCTAssertEqual(lines, [#"{"reason":"signature","status":"rejected","v":1}"#])
        XCTAssertFalse(asked)
    }

    func testSandboxedVoxtPutsTheSocketAtTheTopOfItsContainer() {
        let container = URL(fileURLWithPath: "/Users/someone-with-a-long-name/Library/Containers/com.voxt.Voxt/Data")
        let sandboxed = LocalAPIConnectorLink.socketPath(environment: ["APP_SANDBOX_CONTAINER_ID": "com.voxt.Voxt"], home: container)
        XCTAssertEqual(sandboxed, container.path + "/connector.sock")
        // A socket path holds at most 103 bytes; Application Support inside the container would not fit.
        XCTAssertLessThanOrEqual(sandboxed.utf8.count, 103)
        let home = URL(fileURLWithPath: "/Users/someone")
        XCTAssertEqual(LocalAPIConnectorLink.socketPath(environment: [:], home: home),
                       "/Users/someone/Library/Application Support/Voxt/connector.sock")
    }

    func testAPeerIsTrustedOnlyWhenEveryCheckPasses() {
        XCTAssertTrue(LocalAPIConnectorLink.acceptsPeer(runningCodeValid: true, bundleOnDiskValid: true, hardenedRuntime: true))
        XCTAssertFalse(LocalAPIConnectorLink.acceptsPeer(runningCodeValid: true, bundleOnDiskValid: false, hardenedRuntime: true),
                       "a bundle changed after signing is refused even though its running code is valid")
        XCTAssertFalse(LocalAPIConnectorLink.acceptsPeer(runningCodeValid: false, bundleOnDiskValid: true, hardenedRuntime: true))
        XCTAssertFalse(LocalAPIConnectorLink.acceptsPeer(runningCodeValid: true, bundleOnDiskValid: true, hardenedRuntime: false))
    }
}
