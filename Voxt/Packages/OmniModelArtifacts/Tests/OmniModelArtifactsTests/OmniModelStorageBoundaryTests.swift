import Foundation
import OmniModelArtifacts
import XCTest

final class OmniModelStorageBoundaryTests: XCTestCase {
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

    private func publishOrSkip(_ lease: OmniModelDirectoryLease, to directory: URL) throws {
        do {
            _ = try OmniModelPublication.publish(stagingLease: lease, to: directory)
        } catch OmniModelStorageError.unsupportedAtomicExchange {
            throw XCTSkip("Filesystem does not implement atomic directory exchange.")
        }
    }

    func testDifferentParentRelativeBlobRejectsBeforeDeletingPreviousPayload() throws {
        let otherParent = root.appendingPathComponent("other")
        let otherDestination = otherParent.appendingPathComponent("model")
        try writeModel(at: otherDestination, value: 3)
        let weights = staging.appendingPathComponent("model.safetensors")
        try FileManager.default.moveItem(at: weights, to: root.appendingPathComponent("blob"))
        try FileManager.default.createSymbolicLink(atPath: weights.path, withDestinationPath: "../blob")
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: staging)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: otherDestination)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .publicationRequiresSiblingDirectories)
        }
        XCTAssertEqual(try value(at: otherDestination), 3)
        XCTAssertEqual(try value(at: staging), 2)
    }

    func testAbsoluteStagingInternalWeightRejectsReplacementAndFirstPublication() throws {
        let weights = staging.appendingPathComponent("model.safetensors")
        let internalBlob = staging.appendingPathComponent("blob")
        try FileManager.default.moveItem(at: weights, to: internalBlob)
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: internalBlob)
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: staging)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        for target in [destination!, root.appendingPathComponent("first-publication")] {
            XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: target)) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .relocationSensitiveLink(path: weights.path))
            }
            XCTAssertEqual(try value(at: staging), 2)
            XCTAssertEqual(try value(at: destination), 1)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("first-publication").path))
        try FileManager.default.removeItem(at: weights)
        try FileManager.default.createSymbolicLink(atPath: weights.path, withDestinationPath: "blob")
        try publishOrSkip(lease, to: destination)
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: destination)
        XCTAssertEqual(try value(at: destination), 2)
    }

    func testExternalAliasBackIntoStagingRejectsBeforeCommit() throws {
        let weights = staging.appendingPathComponent("model.safetensors")
        try FileManager.default.moveItem(at: weights, to: staging.appendingPathComponent("blob"))
        let alias = root.appendingPathComponent("external-alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: staging)
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: alias.appendingPathComponent("blob"))
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: staging)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .relocationSensitiveLink(path: weights.path))
        }
        XCTAssertEqual(try value(at: staging), 2)
        XCTAssertEqual(try value(at: destination), 1)
    }

    func testLinkIntoPreviousDestinationRejectsEvenWhenFutureTargetWouldBeValid() throws {
        let previousBlob = destination.appendingPathComponent("blob")
        try FileManager.default.copyItem(at: destination.appendingPathComponent("model.safetensors"), to: previousBlob)
        let weights = staging.appendingPathComponent("model.safetensors")
        try FileManager.default.moveItem(at: weights, to: staging.appendingPathComponent("blob"))
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: previousBlob)
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: staging)
        XCTAssertEqual(try value(at: staging), 1)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .relocationSensitiveLink(path: weights.path))
        }
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: staging), 1)
    }

    func testSiblingRelativeExternalBlobSurvivesPublication() throws {
        let weights = staging.appendingPathComponent("model.safetensors")
        try FileManager.default.moveItem(at: weights, to: root.appendingPathComponent("blob"))
        try FileManager.default.createSymbolicLink(atPath: weights.path, withDestinationPath: "../blob")
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        try publishOrSkip(lease, to: destination)
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: destination)
        XCTAssertEqual(try value(at: destination), 2)
    }

    func testRelativeInternalLinksSurvivePublicationThroughCanonicalParentAlias() throws {
        let weights = staging.appendingPathComponent("model.safetensors")
        let nested = staging.appendingPathComponent("nested")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: false)
        try FileManager.default.moveItem(at: weights, to: nested.appendingPathComponent("blob"))
        try FileManager.default.createSymbolicLink(atPath: weights.path, withDestinationPath: "nested/link")
        try FileManager.default.createSymbolicLink(atPath: nested.appendingPathComponent("link").path, withDestinationPath: "../nested/blob")
        let parentAlias = root.appendingPathComponent("parent-alias")
        try FileManager.default.createSymbolicLink(at: parentAlias, withDestinationURL: root)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        try publishOrSkip(lease, to: parentAlias.appendingPathComponent("model"))
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: destination)
        XCTAssertEqual(try value(at: destination), 2)
    }

    func testStableAbsoluteExternalBlobThroughSymlinkChainSurvivesPublication() throws {
        let weights = staging.appendingPathComponent("model.safetensors")
        let blob = root.appendingPathComponent("blob")
        let blobAlias = root.appendingPathComponent("blob-alias")
        try FileManager.default.moveItem(at: weights, to: blob)
        try FileManager.default.createSymbolicLink(atPath: blobAlias.path, withDestinationPath: "blob")
        try FileManager.default.createSymbolicLink(at: weights, withDestinationURL: blobAlias)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        try publishOrSkip(lease, to: destination)
        try OmniModelArtifactValidation.validateSafetensorsDirectory(at: destination)
        XCTAssertEqual(try value(at: destination), 2)
        XCTAssertEqual(try Data(contentsOf: blob).last, 2)
    }

    func testPublicationRejectsImportedLockNamespaceAfterStagingLeaseAcquisition() throws {
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        let namespace = staging.appendingPathComponent("nested/.omni-model-locks")
        try FileManager.default.createDirectory(at: namespace, withIntermediateDirectories: true)
        let importedLock = namespace.appendingPathComponent("child")
        try Data().write(to: importedLock)
        let inode = try FileManager.default.attributesOfItem(atPath: importedLock.path)[.systemFileNumber] as? NSNumber
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: self.staging.path))
        }
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: staging), 2)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: importedLock.path)[.systemFileNumber] as? NSNumber, inode)
    }

    func testPublicationRejectsDirectorySymlinksRatherThanClaimingTheirContentsAreStable() throws {
        let linkedDirectory = staging.appendingPathComponent("linked-directory")
        try FileManager.default.createSymbolicLink(at: linkedDirectory, withDestinationURL: root)
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .relocationSensitiveLink(path: linkedDirectory.path))
        }
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: staging), 2)
    }

    func testPublicationChecksNestedAuxiliaryLinksEvenWithCustomModelValidation() throws {
        let nested = staging.appendingPathComponent("auxiliary")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: false)
        let linkedFile = nested.appendingPathComponent("config.json")
        try FileManager.default.createSymbolicLink(at: linkedFile, withDestinationURL: staging.appendingPathComponent("config.json"))
        let lease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
        var validated = false
        XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: lease, to: destination, validate: { _ in validated = true })) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .relocationSensitiveLink(path: linkedFile.path))
        }
        XCTAssertTrue(validated)
        XCTAssertEqual(try value(at: destination), 1)
        XCTAssertEqual(try value(at: staging), 2)
    }

    func testParentReaderRejectsChildMutationAndPersistentParentClaimSurvivesRelease() throws {
        let nested = destination.appendingPathComponent("nested/child")
        try writeModel(at: nested, value: 3)
        let parentLock = root.appendingPathComponent(".omni-model-locks/model")
        let parentInode: NSNumber?
        do {
            let reader = try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead)
            parentInode = try FileManager.default.attributesOfItem(atPath: parentLock.path)[.systemFileNumber] as? NSNumber
            XCTAssertThrowsError(try {
                let mutation = try OmniModelDirectoryLease.acquire(for: nested, mode: .exclusiveMutation)
                defer { withExtendedLifetime(mutation) {} }
                try FileManager.default.removeItem(at: nested)
            }()) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: nested.path))
            }
            XCTAssertEqual(try value(at: nested), 3)
            withExtendedLifetime(reader) {}
        }
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: nested, mode: .sharedRead)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: nested.path))
        }
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: parentLock.path)[.systemFileNumber] as? NSNumber, parentInode)
        XCTAssertNoThrow(try OmniModelDirectoryLease.acquire(for: destination, mode: .exclusiveMutation))
        XCTAssertFalse(FileManager.default.fileExists(atPath: nested.deletingLastPathComponent().appendingPathComponent(".omni-model-locks").path))
    }

    func testChildReaderRejectsParentPublicationAndRemovalWithoutUnlinkingLockNamespace() throws {
        let nested = destination.appendingPathComponent("nested/child")
        try writeModel(at: nested, value: 3)
        let childLock = nested.deletingLastPathComponent().appendingPathComponent(".omni-model-locks/child")
        let childInode: NSNumber?
        do {
            let reader = try OmniModelDirectoryLease.acquire(for: nested, mode: .sharedRead)
            childInode = try FileManager.default.attributesOfItem(atPath: childLock.path)[.systemFileNumber] as? NSNumber
            let stagingLease = try OmniModelDirectoryLease.acquire(for: staging, mode: .exclusiveMutation)
            XCTAssertThrowsError(try OmniModelPublication.publish(stagingLease: stagingLease, to: destination)) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: self.destination.path))
            }
            XCTAssertThrowsError(try {
                let mutation = try OmniModelDirectoryLease.acquire(for: destination, mode: .exclusiveMutation)
                defer { withExtendedLifetime(mutation) {} }
                try FileManager.default.removeItem(at: destination)
            }()) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: self.destination.path))
            }
            XCTAssertEqual(try value(at: destination), 1)
            XCTAssertEqual(try value(at: staging), 2)
            XCTAssertEqual(try value(at: nested), 3)
            XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: nested, mode: .exclusiveMutation)) { error in
                XCTAssertEqual(error as? OmniModelStorageError, .busy(path: nested.path))
            }
            withExtendedLifetime(reader) {}
        }
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: childLock.path)[.systemFileNumber] as? NSNumber, childInode)
        XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead)) { error in
            XCTAssertEqual(error as? OmniModelStorageError, .unsupportedNestedRoots(path: self.destination.path))
        }
        XCTAssertNoThrow(try OmniModelDirectoryLease.acquire(for: nested, mode: .exclusiveMutation))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(".omni-model-locks/model").path))
    }
}
