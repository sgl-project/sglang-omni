import Foundation
import OmniModelArtifacts
import XCTest
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

func modelStorageTemporaryDirectory() throws -> URL {
    // note (Codex): Foundation preserves /var on macOS while the filesystem resolves it to /private/var.
    guard let resolved = realpath(FileManager.default.temporaryDirectory.path, nil) else {
        throw OmniModelStorageError.filesystem(operation: "resolve temporary test directory", code: errno)
    }
    defer { free(resolved) }
    return URL(fileURLWithPath: String(cString: resolved), isDirectory: true)
}

final class OmniModelDirectoryLeaseTests: XCTestCase {
    private var root: URL!
    private var directory: URL!

    override func setUpWithError() throws {
        root = try modelStorageTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        directory = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    func testSharedReadersExclusiveBusyAndLastReferenceClosesExactlyOnce() throws {
        var first: OmniModelDirectoryLease? = try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead)
        var second: OmniModelDirectoryLease? = try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead)
        var retained = first
        first = nil
        for _ in 0..<200 {
            XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .busy(path: self.directory.path))
            }
            let unrelated = try OmniModelDirectoryLease.acquire(for: root.appendingPathComponent("other"), mode: .exclusiveMutation)
            withExtendedLifetime(unrelated) {}
        }
        withExtendedLifetime(second) {}
        second = nil
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation))
        withExtendedLifetime(retained) {}
        retained = nil
        let writer = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation))
        withExtendedLifetime(writer) {}
    }

    func testCanonicalSymlinkAliasAndAbsentPayloadUseSameLease() throws {
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: directory)
        let lease = try OmniModelDirectoryLease.acquire(for: alias, mode: .sharedRead)
        XCTAssertEqual(lease.directory, directory)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation))
        withExtendedLifetime(lease) {}
        let absent = root.appendingPathComponent("absent")
        let mutation = try OmniModelDirectoryLease.acquire(for: absent, mode: .exclusiveMutation)
        try FileManager.default.createDirectory(at: absent, withIntermediateDirectories: true)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: absent, mode: .sharedRead))
        withExtendedLifetime(mutation) {}
    }

    func testParentAliasForAbsentPayloadHasStableCanonicalIdentity() throws {
        let parent = root.appendingPathComponent("parent")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: parent)
        let lease = try OmniModelDirectoryLease.acquire(for: alias.appendingPathComponent("new"), mode: .exclusiveMutation)
        XCTAssertEqual(lease.directory, parent.appendingPathComponent("new"))
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: parent.appendingPathComponent("new"), mode: .sharedRead))
        withExtendedLifetime(lease) {}
    }

    func testRejectsSymlinkLockDirectory() throws {
        let outside = root.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent(".omni-model-locks"), withDestinationURL: outside)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)
    }

    func testRejectsSymlinkHardlinkAndNonRegularLockFilesWithoutTruncating() throws {
        let locks = root.appendingPathComponent(".omni-model-locks")
        try FileManager.default.createDirectory(at: locks, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        let lockFile = locks.appendingPathComponent("model")
        let outside = root.appendingPathComponent("outside")
        let original = Data("must not truncate".utf8)
        try original.write(to: outside)
        try FileManager.default.createSymbolicLink(at: lockFile, withDestinationURL: outside)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
        XCTAssertEqual(try Data(contentsOf: outside), original)
        try FileManager.default.removeItem(at: lockFile)
        try FileManager.default.linkItem(at: outside, to: lockFile)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
        XCTAssertEqual(try Data(contentsOf: outside), original)
        try FileManager.default.removeItem(at: lockFile)
        XCTAssertEqual(mkfifo(lockFile.path, 0o600), 0)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
    }

    func testRejectsWorldAccessibleLocksAndReservedPaths() throws {
        let locks = root.appendingPathComponent(".omni-model-locks")
        try FileManager.default.createDirectory(at: locks, withIntermediateDirectories: false)
        XCTAssertEqual(chmod(locks.path, 0o777), 0)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
        XCTAssertEqual(chmod(locks.path, 0o700), 0)
        let lockFile = locks.appendingPathComponent("model")
        try Data().write(to: lockFile)
        XCTAssertEqual(chmod(lockFile.path, 0o666), 0)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
        for invalid in [URL(fileURLWithPath: "/"), locks, locks.appendingPathComponent("nested"), root.appendingPathComponent(".OMNI-MODEL-LOCKS"), URL(string: "https://example.invalid/model")!] {
            XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: invalid, mode: .sharedRead))
        }
    }

    func testTwoProcessSharedReadersAndExclusiveMutationRejectWithoutWaiting() throws {
        let (child, input) = try startProbe(mode: "shared")
        defer { stopProbe(child, input: input) }
        let reader = try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead)
        let started = Date()
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .busy(path: self.directory.path))
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 1)
        withExtendedLifetime(reader) {}
    }

    func testTwoProcessExclusiveRejectsReadersAndCrashReleasesLease() throws {
        let (child, input) = try startProbe(mode: "exclusive")
        defer { stopProbe(child, input: input) }
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead))
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation))
        XCTAssertEqual(kill(child.processIdentifier, SIGKILL), 0)
        try waitForExit(child)
        XCTAssertEqual(child.terminationReason, .uncaughtSignal)
        XCTAssertNoThrow(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation))
    }

    func testChildCannotAcquireWhileParentReaderLivesThenNormalExitReleases() throws {
        do {
            let lease = try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead)
            let child = makeProbe(mode: "exclusive", readyFile: root.appendingPathComponent("blocked"))
            try child.run()
            try waitForExit(child)
            XCTAssertEqual(child.terminationStatus, 2)
            withExtendedLifetime(lease) {}
        }
        let (child, input) = try startProbe(mode: "exclusive")
        defer { stopProbe(child, input: input) }
        input.fileHandleForWriting.write(Data("release\n".utf8))
        try waitForExit(child)
        XCTAssertEqual(child.terminationStatus, 0)
        XCTAssertNoThrow(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation))
    }

    func testOtherProcessParentReaderRejectsNestedChildMutation() throws {
        let nested = directory.appendingPathComponent("nested/child")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        let (child, input) = try startProbe(mode: "shared")
        defer { stopProbe(child, input: input) }
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: nested, mode: .exclusiveMutation)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: nested.path))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: nested.path))
    }

    func testOtherProcessChildReaderRejectsParentMutationAndPublication() throws {
        let nested = directory.appendingPathComponent("nested/child")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        let (child, input) = try startProbe(mode: "shared", payloadDirectory: nested)
        defer { stopProbe(child, input: input) }
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: self.directory.path))
        }
        let staging = root.appendingPathComponent("staging")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        let mutation = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        var validated = false
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: mutation, to: directory, validate: { _ in validated = true })) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: self.directory.path))
        }
        XCTAssertFalse(validated)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: nested, mode: .exclusiveMutation)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .busy(path: nested.path))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: nested.path))
    }

    private func makeProbe(mode: String, readyFile: URL, payloadDirectory: URL? = nil) -> Process {
        let child = Process()
        #if os(macOS)
        let binaryDirectory = Bundle(for: Self.self).bundleURL.deletingLastPathComponent()
        #else
        let binaryDirectory = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        #endif
        child.executableURL = binaryDirectory.appendingPathComponent("StorageLeaseProbe")
        child.arguments = [(payloadDirectory ?? directory).path, mode, readyFile.path]
        child.standardError = Pipe()
        return child
    }

    private func startProbe(mode: String, payloadDirectory: URL? = nil) throws -> (Process, Pipe) {
        let readyFile = root.appendingPathComponent(UUID().uuidString)
        let child = makeProbe(mode: mode, readyFile: readyFile, payloadDirectory: payloadDirectory)
        let input = Pipe()
        child.standardInput = input
        try child.run()
        let deadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: readyFile.path), child.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        guard FileManager.default.fileExists(atPath: readyFile.path) else {
            stopProbe(child, input: input)
            throw CocoaError(.executableRuntimeMismatch)
        }
        return (child, input)
    }

    private func waitForExit(_ child: Process) throws {
        let deadline = Date().addingTimeInterval(5)
        while child.isRunning, Date() < deadline { Thread.sleep(forTimeInterval: 0.01) }
        guard !child.isRunning else {
            _ = kill(child.processIdentifier, SIGKILL)
            child.waitUntilExit()
            XCTFail("Lease probe did not exit within five seconds.")
            throw CocoaError(.executableRuntimeMismatch)
        }
        child.waitUntilExit()
    }

    private func stopProbe(_ child: Process, input: Pipe) {
        if child.isRunning { _ = kill(child.processIdentifier, SIGKILL) }
        child.waitUntilExit()
        try? input.fileHandleForWriting.close()
    }
}
