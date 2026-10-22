import Foundation
import Testing
@testable import VoxtConnectionKit

@Test func pairingCodeAcceptsGoPrivateKeyFormatAndRejectsExpiredOrTokenPayload() throws {
    let now = Date(timeIntervalSince1970: 1_800_000_000)
    let text = "voxtmobile://connect?v=1&tailcat=tc-owned&port=48932&key=privkey:" + String(repeating: "a", count: 64) + "&proof=owned-proof&expires=1800000120"
    let code = try ConnectorPairingCode(url: URL(string: text)!, now: now)
    #expect(code.expiresAt == now.addingTimeInterval(120))
    #expect(throws: (any Error).self) { try ConnectorPairingCode(url: URL(string: text)!, now: now.addingTimeInterval(120)) }
    #expect(throws: (any Error).self) { try ConnectorPairingCode(url: URL(string: text + "&token=forbidden")!, now: now) }
    #expect(throws: (any Error).self) { try ConnectorPairingCode(url: URL(string: text.replacingOccurrences(of: "privkey:", with: "nodepriv:"))!, now: now) }
}

/// Fields added later must not break pairing; the six known ones stay required and unique.
@Test func pairingCodeIgnoresFieldsItDoesNotKnow() throws {
    let now = Date(timeIntervalSince1970: 1_800_000_000)
    let text = "voxtmobile://connect?v=1&tailcat=tc-owned&port=48932&key=privkey:" + String(repeating: "a", count: 64) + "&proof=owned-proof&expires=1800000120"
    #expect((try? ConnectorPairingCode(url: URL(string: text + "&name=Studio%20Mac&future=1")!, now: now)) != nil)
    #expect(throws: (any Error).self) { try ConnectorPairingCode(url: URL(string: text + "&proof=second")!, now: now) }
    #expect(throws: (any Error).self) { try ConnectorPairingCode(url: URL(string: text.replacingOccurrences(of: "&proof=owned-proof", with: ""))!, now: now) }
}

@MainActor @Test(arguments: [false, true]) func pairingProcessPersistsBeforeAcknowledgingAndDeliversQR(refuseSave: Bool) async throws {
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let script = directory.appending(path: "owned-runtime.sh")
    let expiry = Int(Date().timeIntervalSince1970) + 120
    let qr = "voxtmobile://connect?v=1&tailcat=tc-owned&port=48932&key=privkey:" + String(repeating: "a", count: 64) + "&proof=owned-proof&expires=\(expiry)"
    try """
    #!/bin/sh
    [ "$1" = "--connector" ] || exit 1
    read configuration
    printf '%s\\n' '{"event":"ready","address":""}'
    read command
    printf '%s\\n' '{"event":"pairing","url":"\(qr)","expiresAt":\(expiry)}'
    printf '%s\\n' '{"event":"persist","id":"1","allowedClients":["nodekey:\(String(repeating: "b", count: 64))"],"state":"owned-state"}'
    read acknowledgement
    case "$acknowledgement" in *'"ok":\(refuseSave ? "false" : "true")'*) ;; *) exit 2;; esac
    printf '%s\\n' '\(refuseSave ? "{\"event\":\"pairingClosed\"}" : "{\"event\":\"paired\",\"address\":\"tc-normal\",\"client\":\"nodekey:" + String(repeating: "b", count: 64) + "\"}")'
    cat >/dev/null
    """.write(to: script, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
    var saved: ConnectorConfiguration?
    var events: [ConnectorPairingProcess.Event] = []
    let runtime = ConnectorPairingProcess(executableURL: script, persist: { if refuseSave { throw CocoaError(.fileWriteNoPermission) }; saved = $0 })
    runtime.onEvent = { events.append($0) }
    defer { runtime.stop() }
    try await runtime.start(localPort: 47920, token: "owned-token", configuration: .init(allowedClients: [], state: nil))
    try runtime.beginPairing()
    for _ in 0..<100 {
        if refuseSave && events.contains(.pairingClosed) { break }
        if events.contains(.paired(address: "tc-normal", client: "nodekey:" + String(repeating: "b", count: 64))) { break }
        try await Task.sleep(for: .milliseconds(10))
    }
    #expect(saved?.state == (refuseSave ? nil : "owned-state"))
    #expect(events.contains(.pairingClosed) == refuseSave)
    #expect(events.contains(.paired(address: "tc-normal", client: "nodekey:" + String(repeating: "b", count: 64))) == !refuseSave)
    #expect(events.contains { if case .pairing = $0 { return true }; return false })
}

@MainActor @Test(.enabled(if: ProcessInfo.processInfo.environment["CONNECTOR_LIVE_HELPER"] != nil,
                         "Requires an explicitly supplied owned helper"))
func swiftConsumesProductionGoPairingQRAndCancelsTemporaryNode() async throws {
    let helper = try #require(ProcessInfo.processInfo.environment["CONNECTOR_LIVE_HELPER"])
    var receivedCode = false, closed = false, failed = false, saves = 0
    let runtime = ConnectorPairingProcess(executableURL: URL(fileURLWithPath: helper), persist: { _ in saves += 1 })
    runtime.onEvent = { event in
        switch event {
        case .pairing: receivedCode = true
        case .pairingClosed: closed = true
        case .failed: failed = true
        default: break
        }
    }
    defer { runtime.stop() }
    try await runtime.start(localPort: 18080, token: "owned-unused-token", configuration: .init(allowedClients: [], state: nil))
    try runtime.beginPairing()
    for _ in 0..<500 {
        if receivedCode || failed { break }
        try await Task.sleep(for: .milliseconds(20))
    }
    #expect(receivedCode); #expect(!failed); #expect(saves == 0)
    try runtime.cancelPairing()
    for _ in 0..<100 {
        if closed || failed { break }
        try await Task.sleep(for: .milliseconds(20))
    }
    #expect(closed); #expect(!failed)
}

/// The helper puts this Mac's name on the link the phone stores, so the phone can show it.
@MainActor @Test(arguments: [("Studio Mac", "Studio Mac"), (String(repeating: "麦", count: 80), String(repeating: "麦", count: 64))])
func theHelperIsToldThisMacsName(given: String, sent: String) async throws {
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let script = directory.appending(path: "owned-runtime.sh"), received = directory.appending(path: "configuration.json")
    try """
    #!/bin/sh
    read configuration
    printf '%s\\n' "$configuration" > '\(received.path)'
    printf '%s\\n' '{"event":"ready","address":""}'
    cat >/dev/null
    """.write(to: script, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
    let runtime = ConnectorPairingProcess(executableURL: script, persist: { _ in }, macName: given)
    defer { runtime.stop() }
    try await runtime.start(localPort: 47920, token: "owned-token", configuration: .init(allowedClients: [], state: nil))
    for _ in 0..<100 where !FileManager.default.fileExists(atPath: received.path) { try await Task.sleep(for: .milliseconds(10)) }
    let configuration = try JSONSerialization.jsonObject(with: Data(contentsOf: received)) as? [String: Any]
    #expect(configuration?["name"] as? String == sent)
}

@Test func thisMacsNameIsReadable() {
    #expect(!(ConnectorPairingProcess.computerName() ?? "").isEmpty)
}
