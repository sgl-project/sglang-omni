import Foundation
import Testing
@testable import VoxtConnectionKit

/// The Connector asks Voxt for its port and Access token over a Unix socket; each side
/// checks the other's code signature first.
@Suite struct ConnectorLinkTests {
    /// A connected socket pair: the Connector side and a fake Voxt side.
    private func pair() -> (Int32, Int32) {
        var fds: [Int32] = [0, 0]
        #expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0)
        return (fds[0], fds[1])
    }

    private func fakeVoxt(_ fd: Int32, replies: [String], holdOpenSeconds: UInt32 = 0) -> Task<String, Never> {
        Task.detached {
            // A missing request must end the test, not block it.
            var timeout = timeval(tv_sec: 2, tv_usec: 0)
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            var request = Data()
            while !request.contains(UInt8(ascii: "\n")), let byte = try? handle.read(upToCount: 1), !byte.isEmpty { request.append(byte) }
            for reply in replies { handle.write(Data((reply + "\n").utf8)) }
            sleep(holdOpenSeconds)   // like Voxt waiting for the user's click
            return String(decoding: request, as: UTF8.self)
        }
    }

    @Test func approvedConnectionReturnsPortAndToken() async throws {
        let (connector, voxt) = pair()
        let server = fakeVoxt(voxt, replies: [#"{"v":1,"status":"pending"}"#, #"{"v":1,"status":"approved","port":47920,"token":"fixture-token"}"#])
        let link = ConnectorLink(connect: { connector }, verifyPeer: { _ in true })
        let result = try await link.authorize()
        #expect(result.port == 47920)
        #expect(result.token == "fixture-token")
        #expect(await server.value.contains(#""op":"connect""#))
    }

    @Test func approvalSaysWhetherVoxtWillRememberIt() async throws {
        for (remembered, expected) in [("true", true), ("false", false)] {
            let (connector, voxt) = pair()
            _ = fakeVoxt(voxt, replies: [#"{"v":1,"status":"approved","port":47920,"token":"t","alwaysAllowed":\#(remembered)}"#])
            let grant = try await ConnectorLink(connect: { connector }, verifyPeer: { _ in true }).authorize()
            #expect(grant.alwaysAllowed == expected)
            #expect(grant.port == 47920)
        }
        let (connector, voxt) = pair()
        _ = fakeVoxt(voxt, replies: [#"{"v":1,"status":"approved","port":47920,"token":"t"}"#])
        #expect(try await ConnectorLink(connect: { connector }, verifyPeer: { _ in true }).authorize().alwaysAllowed == false,
                "an older Voxt without the field counts as not remembered")
    }

    @Test func cancellingClosesTheSocketPromptly() async throws {
        let (connector, voxt) = pair()
        let server = fakeVoxt(voxt, replies: [#"{"v":1,"status":"pending"}"#], holdOpenSeconds: 4)
        let link = ConnectorLink(connect: { connector }, verifyPeer: { _ in true })
        let started = Date()
        let task = Task { try await link.authorize() }
        try await Task.sleep(for: .milliseconds(300))
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(Date().timeIntervalSince(started) < 2, "Cancel must not wait for Voxt's reply or the 130 s timeout")
        _ = await server.value
    }

    @Test func deniedOrRejectedConnectionThrows() async throws {
        for (reply, expected) in [(#"{"v":1,"status":"denied"}"#, ConnectorLink.Failure.denied),
                                  (#"{"v":1,"status":"rejected","reason":"signature"}"#, .rejectedBySignature)] {
            let (connector, voxt) = pair()
            _ = fakeVoxt(voxt, replies: [reply])
            let link = ConnectorLink(connect: { connector }, verifyPeer: { _ in true })
            await #expect(throws: expected) { try await link.authorize() }
        }
    }

    @Test func untrustedVoxtGetsNothingSent() async throws {
        let (connector, voxt) = pair()
        let server = fakeVoxt(voxt, replies: [])
        let link = ConnectorLink(connect: { connector }, verifyPeer: { _ in false })
        await #expect(throws: ConnectorLink.Failure.untrustedVoxt) { try await link.authorize() }
        #expect(await server.value.isEmpty, "the request is sent only after Voxt's signature checks out")
    }

    @Test func voxtNotRunningIsReported() async throws {
        let link = ConnectorLink(connect: { throw ConnectorLink.Failure.voxtNotRunning }, verifyPeer: { _ in true })
        await #expect(throws: ConnectorLink.Failure.voxtNotRunning) { try await link.authorize() }
    }

    @Test func requirementUsesTheConnectorsOwnTeamAndVoxtBundle() {
        #expect(ConnectorLink.voxtRequirement(team: "ABCDE12345") == #"anchor apple generic and certificate leaf[subject.OU] = "ABCDE12345" and (identifier "com.voxt.Voxt" or identifier "com.voxt.Voxt.omni.dev")"#)
    }

    @Test func socketIsLookedForInEachTrustedVoxtsContainerThenOutsideTheSandbox() {
        let home = URL(fileURLWithPath: "/Users/someone")
        #expect(ConnectorLink.socketPaths(identifiers: ["com.voxt.Voxt", "com.voxt.Voxt.omni.dev"], home: home) == [
            "/Users/someone/Library/Containers/com.voxt.Voxt/Data/connector.sock",
            "/Users/someone/Library/Containers/com.voxt.Voxt.omni.dev/Data/connector.sock",
            "/Users/someone/Library/Application Support/Voxt/connector.sock",
        ])
    }

    @Test func connectUsesTheFirstSocketThatAnswers() throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "cl-\(UUID().uuidString.prefix(6))")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let stale = folder.appending(path: "stale.sock").path
        let live = folder.appending(path: "live.sock").path
        let staleListener = try Self.listen(at: stale)
        close(staleListener)  // the file stays, nobody answers: what a quit Voxt leaves behind
        let listener = try Self.listen(at: live)
        defer { close(listener) }
        let fd = try ConnectorLink.connectToVoxt(paths: [folder.appending(path: "missing.sock").path, stale, live])
        close(fd)
        #expect(throws: ConnectorLink.Failure.voxtNotRunning) { try ConnectorLink.connectToVoxt(paths: [stale]) }
    }

    private static func listen(at path: String) throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutableBytes(of: &address.sun_path) { $0.copyBytes(from: Array(path.utf8) + [0]) }
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard bound == 0, Darwin.listen(fd, 1) == 0 else { close(fd); throw POSIXError(.EADDRINUSE) }
        return fd
    }

    @Test func unsignedConnectorFallsBackToManualEntryWithoutConnecting() async {
        final class Probe: @unchecked Sendable { var connected = false }
        let probe = Probe()
        let link = ConnectorLink(team: nil, connect: { probe.connected = true; return -1 })
        // Same outcome as a Voxt that refused this build: the Connector shows manual entry.
        await #expect(throws: ConnectorLink.Failure.rejectedBySignature) { _ = try await link.authorize() }
        #expect(probe.connected == false)
    }
}
