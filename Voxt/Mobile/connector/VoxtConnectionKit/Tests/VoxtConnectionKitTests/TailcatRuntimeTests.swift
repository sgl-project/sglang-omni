import Foundation
import Testing
@testable import VoxtConnectionKit

private enum PersistenceFailure: Error { case denied }

@MainActor @Test(arguments: [false, true])
func runtimePersistsIdentityBeforeReady(refusePersistence: Bool) async throws {
    let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let script = directory.appending(path: "fixture.sh")
    try #"""
    #!/bin/sh
    read config
    printf '%s\n' '{"version":1,"address":"tc-fixture","port":48931,"state":"synthetic-state"}'
    cat >/dev/null
    """#.write(to: script, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: script.path)
    var persisted: String?
    let runtime = TailcatRuntime(executableURL: script, localPort: 47920,
        allowedClients: ["nodekey:" + String(repeating: "a", count: 64)], state: nil,
        persistState: {
            persisted = $0
            if refusePersistence { throw PersistenceFailure.denied }
        })
    defer { runtime.stop() }
    if refusePersistence {
        await #expect(throws: PersistenceFailure.denied) { try await runtime.start() }
    } else {
        let address = try await runtime.start()
        #expect(address == "tc-fixture")
    }
    #expect(persisted == "synthetic-state")
}
