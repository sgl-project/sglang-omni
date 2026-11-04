import Foundation
@testable import OmniModelArtifacts
import OmniModelStoragePOSIX
import XCTest
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

final class OmniModelPublicationTests: XCTestCase {
    private var root: URL!
    private var destination: URL!
    private var staging: URL!

    override func setUpWithError() throws {
        root = try modelStorageTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        destination = root.appendingPathComponent("model")
        staging = root.appendingPathComponent("download")
        try writeModel(at: destination, value: 1)
        try writeModel(at: staging, value: 2)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func writeModel(at directory: URL, value: UInt8) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        let header = Data(#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        var weights = withUnsafeBytes(of: &length) { Data($0) }
        weights.append(header)
        weights.append(value)
        try weights.write(to: directory.appendingPathComponent("model.safetensors"))
    }

    private func value(at directory: URL) throws -> UInt8? {
        try Data(contentsOf: directory.appendingPathComponent("model.safetensors")).last
    }

    private func exchangeOrSkip(_ lease: OmniModelDirectoryLease) throws -> OmniModelPublication.Outcome {
        do {
            return try OmniModelPublication.publish(stagingLease: lease, to: destination)
        } catch OmniModelStorageError.unsupportedAtomicExchange {
            throw XCTSkip("Filesystem does not implement atomic directory exchange; rejection preserved old payload.")
        }
    }

    func testActualExchangeAndStableDestinationLockIdentity() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        let lockFile = root.appendingPathComponent(".omni-model-locks/model")
        let oldInode: NSNumber?
        do {
            let initialReader = try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead)
            oldInode = try FileManager.default.attributesOfItem(atPath: lockFile.path)[.systemFileNumber] as? NSNumber
            XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination)) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .busy(path: self.destination.path))
            }
            withExtendedLifetime(initialReader) {}
        }
        _ = try exchangeOrSkip(lease)
        let inode = try FileManager.default.attributesOfItem(atPath: lockFile.path)[.systemFileNumber] as? NSNumber
        XCTAssertEqual(inode, oldInode)
        XCTAssertEqual(try value(at: destination), 2)
    }

    func testActualExchangePublishesVerifiedNewDirectoryAndConsumesStagingLease() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        let outcome = try exchangeOrSkip(lease)
        XCTAssertEqual(outcome.directory, destination)
        XCTAssertNil(outcome.retainedPreviousDirectory)
        XCTAssertNil(outcome.cleanupErrorDescription)
        XCTAssertEqual(try value(at: destination), 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path))
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .stagingPathReuse(path: self.staging.path))
        }
    }

    func testPublicationAllowsBlobSymlinksAndCleanupDoesNotDeleteBlobTargets() throws {
        let previousBlob = root.appendingPathComponent("previous-blob")
        let newBlob = root.appendingPathComponent("new-blob")
        for (directory, blob) in [(destination!, previousBlob), (staging!, newBlob)] {
            let weights = directory.appendingPathComponent("model.safetensors")
            try FileManager.default.moveItem(at: weights, to: blob)
            try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: blob)
        }
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        _ = try exchangeOrSkip(lease)
        XCTAssertEqual(try value(at: destination), 2)
        XCTAssertEqual(try Data(contentsOf: previousBlob).last, 1)
        XCTAssertEqual(try Data(contentsOf: newBlob).last, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path))
    }

    func testFirstPublicationRenameAndLockSurvivesRemoveRecreate() throws {
        try FileManager.default.removeItem(at: destination)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        let outcome = try OmniModelPublication.publish(stagingLease: lease, to: destination)
        XCTAssertEqual(outcome.directory, destination)
        XCTAssertEqual(try value(at: destination), 2)
        let lockFile = root.appendingPathComponent(".omni-model-locks/model")
        let inode = try FileManager.default.attributesOfItem(atPath: lockFile.path)[.systemFileNumber] as? NSNumber
        let mutation = try OmniModelDirectoryLease.acquire(for: destination, mode: .exclusiveMutation)
        try FileManager.default.removeItem(at: destination)
        try writeModel(at: destination, value: 3)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead))
        withExtendedLifetime(mutation) {}
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: lockFile.path)[.systemFileNumber] as? NSNumber, inode)
    }

    func testInvalidPayloadAndModelSpecificValidationPreserveOld() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        try Data("truncated".utf8).write(to: staging.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination))
        XCTAssertEqual(try value(at: destination), 1)
        try writeModel(at: staging, value: 2)
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination, validate: { directory in
            try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory)
            throw CocoaError(.fileReadCorruptFile)
        }))
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: staging), 2)
    }

    func testPrecommitCancellationPreservesBothTreesAndAllowsRetry() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        var checks = 0
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination, checkCancellation: {
            checks += 1
            if checks == 2 { throw CancellationError() }
        })) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(checks, 2)
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: staging), 2)
        _ = try exchangeOrSkip(lease)
        XCTAssertEqual(try value(at: destination), 2)
    }

    func testActualRenameFailurePreservesOldAndDoesNotFallback() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        let relocated = root.appendingPathComponent("relocated")
        var checks = 0
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination, checkCancellation: {
            checks += 1
            if checks == 2 { try FileManager.default.moveItem(at: self.staging, to: relocated) }
        })) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .filesystem(operation: "publish directory", code: ENOENT))
        }
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: relocated), 2)
    }

    func testUnsupportedExchangeAndCrossDeviceErrorsNeverFallback() throws {
        for errorCode in [ENOSYS, EOPNOTSUPP, EINVAL, EXDEV, EACCES, EIO] {
            let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
            var cleanupCalled = false
            XCTAssertThrowsError(try OmniModelPublication.publish(
                stagingLease: lease, to: destination,
                validate: OmniModelArtifactValidation.validateSafetensorsDirectory(at:), checkCancellation: {},
                exchange: { _, _ in errorCode }, cleanup: { _ in cleanupCalled = true }
            )) { error in
                if errorCode == EXDEV {
                    XCTAssertEqual(error as? OmniModelStorageError, .crossFilesystem)
                } else if [ENOSYS, EOPNOTSUPP, EINVAL].contains(errorCode) {
                    XCTAssertEqual(error as? OmniModelStorageError, .unsupportedAtomicExchange)
                } else {
                    XCTAssertEqual(error as? OmniModelStorageError, .filesystem(operation: "publish directory", code: errorCode))
                }
            }
            XCTAssertFalse(cleanupCalled)
            XCTAssertEqual(try value(at: destination), 1)
            XCTAssertEqual(try value(at: staging), 2)
        }
    }

    func testActualCleanupFailureCommitsAndQuarantinesOldStaging() throws {
        guard geteuid() != 0 else { throw XCTSkip("Permission failure requires a non-root user.") }
        XCTAssertEqual(chmod(destination.path, 0o500), 0)
        defer { _ = chmod(staging.path, 0o700); _ = chmod(destination.path, 0o700) }
        let outcome: OmniModelPublication.Outcome
        do {
            let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
            outcome = try exchangeOrSkip(lease)
        }
        XCTAssertEqual(outcome.directory, destination)
        XCTAssertEqual(outcome.retainedPreviousDirectory, staging)
        XCTAssertNotNil(outcome.cleanupErrorDescription)
        XCTAssertEqual(try value(at: destination), 2)
        XCTAssertEqual(try value(at: staging), 1)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .stagingPathReuse(path: self.staging.path))
        }
        XCTAssertNoThrow(try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead))
    }

    func testNoCancellationCheckAfterCommitEvenWhenCleanupFails() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        var committed = false
        var checks = 0
        let outcome: OmniModelPublication.Outcome
        do {
            outcome = try OmniModelPublication.publish(
                stagingLease: lease, to: destination,
                validate: OmniModelArtifactValidation.validateSafetensorsDirectory(at:),
                checkCancellation: {
                    checks += 1
                    if committed { throw CancellationError() }
                },
                exchange: { first, second in
                    let result = omni_model_storage_exchange(first.path, second.path)
                    committed = result == 0
                    return result
                }, cleanup: { _ in throw CancellationError() }
            )
        } catch OmniModelStorageError.unsupportedAtomicExchange {
            throw XCTSkip("Filesystem does not implement atomic directory exchange.")
        }
        XCTAssertEqual(checks, 2)
        XCTAssertEqual(try value(at: destination), 2)
        XCTAssertEqual(outcome.retainedPreviousDirectory, staging)
        XCTAssertNotNil(outcome.cleanupErrorDescription)
    }

    func testRejectsOverlappingDirectoriesAndSharedStagingLease() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        for invalid in [staging!, root!, staging.appendingPathComponent("child")] {
            XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: invalid)) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .stagingPathReuse(path: self.staging.path))
            }
        }
        let reader = try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead)
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: reader, to: staging)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .exclusiveLeaseRequired)
        }
    }

    func testCrashBeforeCommitQuarantinesStagingButReleasesDestinationLease() throws {
        let child = Process()
        #if os(macOS)
        let binaryDirectory = Bundle(for: Self.self).bundleURL.deletingLastPathComponent()
        #else
        let binaryDirectory = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        #endif
        child.executableURL = binaryDirectory.appendingPathComponent("StorageLeaseProbe")
        let readyFile = root.appendingPathComponent("ready")
        child.arguments = [staging.path, "pending", readyFile.path, destination.path]
        let input = Pipe()
        child.standardInput = input
        try child.run()
        defer {
            if child.isRunning { _ = kill(child.processIdentifier, SIGKILL) }
            child.waitUntilExit()
            try? input.fileHandleForWriting.close()
        }
        let deadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: readyFile.path), child.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        guard FileManager.default.fileExists(atPath: readyFile.path) else { return XCTFail("Publication probe did not reach precommit gate") }
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead))
        XCTAssertEqual(kill(child.processIdentifier, SIGKILL), 0)
        child.waitUntilExit()
        XCTAssertNoThrow(try OmniModelDirectoryLease.acquire(for: destination, mode: .exclusiveMutation))
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .stagingPathReuse(path: self.staging.path))
        }
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: staging), 2)
    }

    func testActualCrossFilesystemRejectsBeforeValidationOrMutation() throws {
        #if os(Linux)
        let otherRoot = URL(fileURLWithPath: "/dev/shm").appendingPathComponent(UUID().uuidString).appendingPathComponent("model")
        try writeModel(at: otherRoot, value: 3)
        defer { try? FileManager.default.removeItem(at: otherRoot.deletingLastPathComponent()) }
        var localStatus = stat()
        var remoteStatus = stat()
        guard stat(root.path, &localStatus) == 0, stat(otherRoot.path, &remoteStatus) == 0,
              localStatus.st_dev != remoteStatus.st_dev else { throw XCTSkip("No distinct test filesystem.") }
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: otherRoot)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .crossFilesystem)
        }
        XCTAssertEqual(try value(at: otherRoot), 3)
        XCTAssertEqual(try value(at: staging), 2)
        #else
        throw XCTSkip("Cross-filesystem fixture requires Linux /dev/shm; real macOS volume regression remains a platform gate.")
        #endif
    }
}
