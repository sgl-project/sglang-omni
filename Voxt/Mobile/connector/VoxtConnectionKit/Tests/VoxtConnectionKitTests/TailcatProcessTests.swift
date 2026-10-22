import Darwin
import Foundation
import XCTest
@testable import VoxtConnectionKit

@MainActor final class TailcatProcessTests: XCTestCase {
    private var directory: URL!
    private let configuration = TailcatConfiguration(state: nil, allowedClients: ["nodekey:" + String(repeating: "a", count: 64)], localPort: 47920)
    private let success = #"{"version":1,"address":"tc-fixture","port":48931,"state":"opaque"}"#
    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-process-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    override func tearDown() async throws { try FileManager.default.removeItem(at: directory) }
    private func script(_ body: String) throws -> URL {
        let url = directory.appendingPathComponent(UUID().uuidString + ".sh")
        try ("#!/bin/sh\n" + body).write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
        return url
    }
    private func waitForFile(_ name: String) async throws -> URL {
        let url = directory.appendingPathComponent(name)
        for _ in 0..<150 {
            if FileManager.default.fileExists(atPath: url.path) { return url }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Child did not create expected file")
        throw CocoaError(.fileNoSuchFile)
    }
    private func waitForExit(_ pid: pid_t) async throws {
        for _ in 0..<150 {
            if kill(pid, 0) == -1 && errno == ESRCH { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Owned child stayed alive")
    }
    func testReadyArrivesBeforeEOFAndStopClosesChild() async throws {
        let pidFile = directory.appendingPathComponent("pid").path
        let url = try script("echo $$ > '\(pidFile)'\nread config\nprintf '%s\\n' '\(success)'\ncat >/dev/null\n")
        let process = TailcatProcess(executableURL: url)
        defer { process.stop() }
        let ready = try await process.start(configuration: configuration)
        XCTAssertEqual(ready.address, "tc-fixture")
        let path = try await waitForFile("pid")
        let pid = try XCTUnwrap(Int32(String(contentsOf: path, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
        XCTAssertEqual(kill(pid, 0), 0)
        process.stop()
        try await waitForExit(pid)
    }
    func testMalformedOversizedAndEarlyExitAreRejected() async throws {
        for body in ["exit 1", "read config\nprintf 'garbage\\n'", "read config\nhead -c 70000 /dev/zero\nprintf '\\n'"] {
            let process = TailcatProcess(executableURL: try script(body), startupTimeoutSeconds: 0.5)
            defer { process.stop() }
            do { _ = try await process.start(configuration: configuration); XCTFail("Invalid child reply was accepted") }
            catch { }
        }
    }
    func testStartupDeadlineKillsSilentChild() async throws {
        let pidFile = directory.appendingPathComponent("pid").path
        let process = TailcatProcess(executableURL: try script("echo $$ > '\(pidFile)'\nread config\nexec sleep 30"), startupTimeoutSeconds: 1)
        defer { process.stop() }
        let start = ContinuousClock.now
        do { _ = try await process.start(configuration: configuration); XCTFail("Silent helper was accepted") }
        catch { }
        XCTAssertLessThan(start.duration(to: .now), .seconds(4))
        let path = try await waitForFile("pid")
        let pid = try XCTUnwrap(Int32(String(contentsOf: path, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
        try await waitForExit(pid)
    }
    func testCancellationAndStopDuringLaunchRejectLateReply() async throws {
        for cancel in [true, false] {
            let name = cancel ? "cancelpid" : "stoppid"
            let pidFile = directory.appendingPathComponent(name).path
            let process = TailcatProcess(executableURL: try script("echo $$ > '\(pidFile)'\nread config\nexec sleep 30"))
            defer { process.stop() }
            let task = Task { try await process.start(configuration: configuration) }
            _ = try? await waitForFile(name)
            if cancel { task.cancel() } else { process.stop() }
            do { _ = try await task.value; XCTFail("Cancelled/stopped launch was accepted") }
            catch { }
        }
    }
    func testBrokenStdinCannotKillParent() async throws {
        let large = TailcatConfiguration(state: String(repeating: "x", count: 48000), allowedClients: configuration.allowedClients, localPort: 47920)
        let process = TailcatProcess(executableURL: try script("exec 0<&-\nsleep 0.15\nexit 1"), startupTimeoutSeconds: 0.5)
        defer { process.stop() }
        do { _ = try await process.start(configuration: large); XCTFail("Child with closed stdin was accepted") }
        catch { }
    }
    func testUnexpectedExitNotifiesButExplicitStopDoesNot() async throws {
        let url = try script("read config\nprintf '%s\\n' '\(success)'\nsleep 0.2\nexit 0")
        let process = TailcatProcess(executableURL: url)
        defer { process.stop() }
        var exits = 0
        process.onUnexpectedExit = { exits += 1 }
        _ = try await process.start(configuration: configuration)
        try await Task.sleep(for: .milliseconds(450))
        XCTAssertEqual(exits, 1)
        _ = try await process.start(configuration: configuration)
        process.stop()
        try await Task.sleep(for: .milliseconds(450))
        XCTAssertEqual(exits, 1)
    }
    func testClosedReadEndReturnsEPIPEInsteadOfTerminatingParent() throws {
        let pipe = Pipe()
        defer { try? pipe.fileHandleForWriting.close() }
        try TailcatProcess.configureInputPipe(pipe.fileHandleForWriting.fileDescriptor)
        try pipe.fileHandleForReading.close()
        var byte: UInt8 = 1
        let count = Darwin.write(pipe.fileHandleForWriting.fileDescriptor, &byte, 1)
        XCTAssertEqual(count, -1)
        XCTAssertEqual(errno, EPIPE)
    }

    func testReplacingPendingLaunchKeepsNewHelperAlive() async throws {
        let firstPIDFile = directory.appendingPathComponent("firstpid").path
        let nextPIDFile = directory.appendingPathComponent("nextpid").path
        let url = try script("read config\ncase \"$config\" in *first*) echo $$ > '\(firstPIDFile)'; exec sleep 30;; esac\necho $$ > '\(nextPIDFile)'\nprintf '%s\\n' '\(success)'\ncat >/dev/null\n")
        let process = TailcatProcess(executableURL: url)
        defer { process.stop() }
        var exits = 0
        process.onUnexpectedExit = { exits += 1 }
        let firstConfiguration = TailcatConfiguration(state: "first", allowedClients: configuration.allowedClients, localPort: 47920)
        let first = Task { try await process.start(configuration: firstConfiguration) }
        _ = try await waitForFile("firstpid")
        _ = try await process.start(configuration: configuration)
        do { _ = try await first.value; XCTFail("Replaced launch returned ready") }
        catch is CancellationError { }
        let nextPath = try await waitForFile("nextpid")
        let nextPID = try XCTUnwrap(Int32(String(contentsOf: nextPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
        try await Task.sleep(for: .milliseconds(150))
        XCTAssertEqual(kill(nextPID, 0), 0)
        XCTAssertEqual(exits, 0)
        process.stop()
        try await waitForExit(nextPID)
    }

    func testReleasingOwnerClosesReadyHelper() async throws {
        let pidFile = directory.appendingPathComponent("pid").path
        let url = try script("echo $$ > '\(pidFile)'\nread config\nprintf '%s\\n' '\(success)'\ncat >/dev/null\n")
        var process: TailcatProcess? = TailcatProcess(executableURL: url)
        weak var owner = process
        _ = try await process!.start(configuration: configuration)
        let path = try await waitForFile("pid")
        let pid = try XCTUnwrap(Int32(String(contentsOf: path, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
        process = nil
        XCTAssertNil(owner)
        try await waitForExit(pid)
    }

    func testTerminationIgnoringChildIsReaped() async throws {
        let pidFile = directory.appendingPathComponent("pid").path
        let url = try script("trap '' TERM\necho $$ > '\(pidFile)'\nread config\nprintf '%s\\n' '\(success)'\nexec sleep 30\n")
        let process = TailcatProcess(executableURL: url)
        defer { process.stop() }
        _ = try await process.start(configuration: configuration)
        let path = try await waitForFile("pid")
        let pid = try XCTUnwrap(Int32(String(contentsOf: path, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
        process.stop()
        try await waitForExit(pid)
    }

}
