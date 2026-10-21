// Runs Voxt's real LocalAPIConnectorLink inside the App Sandbox for 60 seconds.
import Foundation
import Security
setbuf(stdout, nil)
let environment = ProcessInfo.processInfo.environment
let path = LocalAPIConnectorLink.defaultPath
print("server sandbox container: \(environment["APP_SANDBOX_CONTAINER_ID"] ?? "none")")
print("server team: \(LocalAPIConnectorLink.ownTeamIdentifier() ?? "none")")
print("server socket: \(path) (\(path.utf8.count) bytes)")
print("server requires connector: \(LocalAPIConnectorLink.connectorBundleIdentifier())")
let identifier = LocalAPIConnectorLink.connectorBundleIdentifier()
let requirement = LocalAPIConnectorLink.connectorRequirement(team: LocalAPIConnectorLink.ownTeamIdentifier() ?? "", identifier: identifier)
func timedCheck(_ fd: Int32) -> Bool {
    let start = Date()
    var token = audit_token_t()
    var size = socklen_t(MemoryLayout<audit_token_t>.size)
    _ = getsockopt(fd, SOL_LOCAL, LOCAL_PEERTOKEN, &token, &size)
    var code: SecCode?
    let lookup = SecCodeCopyGuestWithAttributes(nil, [kSecGuestAttributeAudit: withUnsafeBytes(of: token) { Data($0) }] as CFDictionary, [], &code)
    print("server peer check start, guest lookup \(lookup)")
    let trusted = LocalAPIConnectorLink.verifyConnector(fd, requirement: requirement)
    print("server peer check \(trusted) after \(Int(Date().timeIntervalSince(start) * 1000)) ms")
    return trusted
}
let link = LocalAPIConnectorLink(verifyPeer: timedCheck, isRemembered: { true }, remember: {}, ask: { .deny }, credentials: { (47999, "fixture-token") })
do { try link.start(); print("server listening") } catch { print("server start failed: \(error)"); exit(1) }
Thread.sleep(forTimeInterval: 60)
link.stop()
print("server stopped")
