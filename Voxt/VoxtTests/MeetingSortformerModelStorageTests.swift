import Foundation
import OmniModelArtifacts
import XCTest
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@testable import Voxt

@MainActor
final class MeetingSortformerModelStorageTests: XCTestCase {
    private var root: URL!
    private var directory: URL!

    override func setUp() async throws {
        root = try TestDoubles.canonicalTemporaryDirectory().appendingPathComponent("voxt-sortformer-readers-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(root)
        directory = try XCTUnwrap(MeetingSortformerModelStorage.writeModelDirectory())
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        try Data([1]).write(to: directory.appendingPathComponent("model.safetensors"))
    }

    override func tearDown() async throws {
        ModelStorageDirectoryManager.resetForTesting()
        try FileManager.default.removeItem(at: root)
    }

    func testValidatedReaderExcludesDeleteAndReleasesWithoutMovingModel() async throws {
        var reader = try await MeetingSortformerModelStorage.validatedModelDirectoryLease()
        XCTAssertEqual(reader?.directory.path, try TestDoubles.canonicalFileURL(for: directory).path)
        do {
            try await ModelDiskOperations.removeModelDirectories([directory])
            XCTFail("A validated reader did not exclude deletion")
        } catch OmniModelStorageError.busy {}
        reader = nil
        try await ModelDiskOperations.removeModelDirectories([directory])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    func testDeletionDeduplicatesParentAliasesAndStillDeletesDistinctDirectories() async throws {
        let alias = try parentAlias()
        let distinct = root.appendingPathComponent("distinct")
        try FileManager.default.createDirectory(at: distinct, withIntermediateDirectories: true)
        try await ModelDiskOperations.removeModelDirectories([directory, alias, directory, distinct])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: distinct.path))
    }

    func testReaderThroughEitherParentAliasBlocksAllDeletions() async throws {
        let alias = try parentAlias()
        let distinct = root.appendingPathComponent("distinct")
        try FileManager.default.createDirectory(at: distinct, withIntermediateDirectories: true)
        for readerDirectory in [directory!, alias] {
            let reader = try OmniModelDirectoryLease.acquire(for: readerDirectory, mode: .sharedRead)
            do {
                try await ModelDiskOperations.removeModelDirectories([distinct, directory, alias])
                XCTFail("An alias bypassed the active reader")
            } catch OmniModelStorageError.busy {}
            XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
            XCTAssertTrue(FileManager.default.fileExists(atPath: distinct.path))
            withExtendedLifetime(reader) {}
        }
        try await ModelDiskOperations.removeModelDirectories([distinct, alias, directory])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: distinct.path))
    }

    func testLeafSymlinkDeletionRemainsBlockedByATargetReader() async throws {
        let link = root.appendingPathComponent("leaf-link")
        let secondLink = root.appendingPathComponent("second-leaf-link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: directory)
        try FileManager.default.createSymbolicLink(at: secondLink, withDestinationURL: directory)
        let reader = try OmniModelDirectoryLease.acquire(for: directory, mode: .sharedRead)
        do {
            try await ModelDiskOperations.removeModelDirectories([link, secondLink])
            XCTFail("Unlink bypassed the target reader")
        } catch OmniModelStorageError.busy {}
        XCTAssertTrue(FileManager.default.fileExists(atPath: link.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: secondLink.path))
        withExtendedLifetime(reader) {}
    }

    func testLeafSymlinkAliasesUnlinkOnceWithoutRemovingTheirTarget() async throws {
        let link = root.appendingPathComponent("leaf-link")
        let secondLink = root.appendingPathComponent("second-leaf-link")
        let rootAlias = root.appendingPathComponent("root-alias")
        try FileManager.default.createSymbolicLink(at: rootAlias, withDestinationURL: root)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: directory)
        try FileManager.default.createSymbolicLink(at: secondLink, withDestinationURL: directory)
        try await ModelDiskOperations.removeModelDirectories([
            link, rootAlias.appendingPathComponent("leaf-link"), secondLink
        ])
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertThrowsError(try FileManager.default.destinationOfSymbolicLink(atPath: link.path))
        XCTAssertThrowsError(try FileManager.default.destinationOfSymbolicLink(atPath: secondLink.path))
    }

    func testMultiplyLinkedLeafSymlinksRejectBeforeAnyDeletion() async throws {
        let firstLink = root.appendingPathComponent("first-linked-symlink")
        let secondLink = root.appendingPathComponent("second-linked-symlink")
        try FileManager.default.createSymbolicLink(at: firstLink, withDestinationURL: directory)
        guard linkat(AT_FDCWD, firstLink.path, AT_FDCWD, secondLink.path, 0) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: secondLink.path), directory.path)
        var entryStatus = stat()
        XCTAssertEqual(lstat(firstLink.path, &entryStatus), 0)
        XCTAssertGreaterThan(entryStatus.st_nlink, 1)
        do {
            try await ModelDiskOperations.removeModelDirectories([directory, firstLink, secondLink])
            XCTFail("Multiply linked symlinks reported partial deletion as success")
        } catch OmniModelStorageError.unsafePath {}
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: firstLink.path), directory.path)
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: secondLink.path), directory.path)
    }

    func testLeafSymlinkAndPhysicalTargetCanBothBeDeletedInEitherInputOrder() async throws {
        let link = root.appendingPathComponent("leaf-link")
        for linkFirst in [false, true] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: directory)
            try await ModelDiskOperations.removeModelDirectories(linkFirst ? [link, directory] : [directory, link])
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
            XCTAssertThrowsError(try FileManager.default.destinationOfSymbolicLink(atPath: link.path))
        }
    }

    func testCaseInsensitiveFilesystemAliasesShareOneDeletionAndReaderProtection() async throws {
        let alias = directory.deletingLastPathComponent().appendingPathComponent(directory.lastPathComponent.uppercased())
        guard alias.path != directory.path, FileManager.default.fileExists(atPath: alias.path) else {
            throw XCTSkip("Case-alias regression requires a case-insensitive filesystem")
        }
        var reader: OmniModelDirectoryLease? = try OmniModelDirectoryLease.acquire(for: alias, mode: .sharedRead)
        do {
            try await ModelDiskOperations.removeModelDirectories([directory, alias])
            XCTFail("Case alias bypassed the active reader")
        } catch OmniModelStorageError.busy {}
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        withExtendedLifetime(reader) {}
        reader = nil
        try await ModelDiskOperations.removeModelDirectories([directory, alias])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    private func parentAlias() throws -> URL {
        let parentAlias = root.appendingPathComponent("parent-alias")
        try FileManager.default.createSymbolicLink(at: parentAlias, withDestinationURL: directory.deletingLastPathComponent())
        return parentAlias.appendingPathComponent(directory.lastPathComponent)
    }

    func testMutationRejectsModelAcquisitionBeforeValidationOrLoading() async throws {
        let mutation = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
        do {
            _ = try await MeetingSortformerModelStorage.validatedModelDirectoryLease()
            XCTFail("A mutating model was selected")
        } catch OmniModelStorageError.busy {}
        withExtendedLifetime(mutation) {}
    }

    func testInvalidPresenceAndMissingDirectoryReleaseReaders() async throws {
        try Data("broken".utf8).write(to: directory.appendingPathComponent("config.json"))
        let invalid = try await MeetingSortformerModelStorage.validatedModelDirectoryLease()
        XCTAssertNil(invalid)
        _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
        try FileManager.default.removeItem(at: directory)
        let missing = try await MeetingSortformerModelStorage.validatedModelDirectoryLease()
        XCTAssertNil(missing)
    }

    func testCancelledSelectionDoesNotLeaveAReader() async throws {
        let selection = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await MeetingSortformerModelStorage.validatedModelDirectoryLease()
        }
        do { _ = try await selection.value; XCTFail("Cancelled selection succeeded") }
        catch is CancellationError {}
        _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
    }

}
