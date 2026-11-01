import Foundation
import OmniModelArtifacts
import XCTest
@testable import Voxt

@MainActor
final class ModelInstallationCacheTests: XCTestCase {
    func testUnknownStateIsCheckingAndOneRequestIsCoalesced() async throws {
        let cache = ModelInstallationCache()
        let started = expectation(description: "scan started")
        let release = DispatchSemaphore(value: 0)
        let directory = URL(fileURLWithPath: "/tmp/installed-model")
        XCTAssertTrue(cache.isChecking("repo"))
        cache.request("repo") {
            XCTAssertFalse(Thread.isMainThread)
            started.fulfill()
            release.wait()
            return ModelInstallationSnapshot(directory: directory)
        }
        cache.request("repo") { XCTFail("Duplicate scan"); return .init() }
        await fulfillment(of: [started], timeout: 1)
        XCTAssertNil(cache.peek("repo"))
        release.signal()
        let value = try await cache.value("repo") { XCTFail("Duplicate scan"); return .init() }
        XCTAssertEqual(value.directory, directory)
        XCTAssertFalse(cache.isChecking("repo"))
    }

    func testInvalidatedScanCannotRestoreDeletedModelOrOldStorageRoot() async throws {
        let cache = ModelInstallationCache()
        let started = expectation(description: "old scan started")
        let release = DispatchSemaphore(value: 0)
        let old = Task {
            try await cache.value("repo") {
                started.fulfill()
                release.wait()
                return .init(directory: URL(fileURLWithPath: "/old-root/model"))
            }
        }
        await fulfillment(of: [started], timeout: 1)
        cache.invalidateAll()
        let current = try await cache.value("repo") { .init() }
        XCTAssertFalse(current.isInstalled)
        release.signal()
        do {
            _ = try await old.value
            XCTFail("Old generation should be invalidated")
        } catch is CancellationError {} catch { XCTFail("Unexpected error: \(error)") }
        XCTAssertEqual(cache.peek("repo"), current)
    }

    func testCancelledWaiterDoesNotWaitForSlowDiskOrCancelSharedScan() async throws {
        let cache = ModelInstallationCache()
        let started = expectation(description: "scan started")
        let cancelled = expectation(description: "waiter cancelled before disk completes")
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let waiter = Task {
            do {
                _ = try await cache.value("repo") {
                    started.fulfill()
                    release.wait()
                    return .init()
                }
                XCTFail("Cancelled waiter returned a value")
            } catch is CancellationError { cancelled.fulfill() }
            catch { XCTFail("Unexpected error: \(error)") }
        }
        await fulfillment(of: [started], timeout: 1)
        waiter.cancel()
        await fulfillment(of: [cancelled], timeout: 1)
        XCTAssertTrue(cache.hasPendingRequest("repo"))
    }

    func testPartialAndIncompleteShardsAreNotInstalled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        try Data("weights".utf8).write(to: root.appendingPathComponent("part-1.safetensors"))
        let index = #"{"weight_map":{"a":"part-1.safetensors","b":"part-2.safetensors"}}"#
        try Data(index.utf8).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertFalse(CustomLLMModelStorageSupport.isModelDirectoryValid(root))
        XCTAssertFalse(MLXModelDownloadSupport.isModelDirectoryValid(root, fileManager: .default))
        try Data("weights".utf8).write(to: root.appendingPathComponent("part-2.safetensors"))
        XCTAssertTrue(CustomLLMModelStorageSupport.isModelDirectoryValid(root))
        XCTAssertTrue(MLXModelDownloadSupport.isModelDirectoryValid(root, fileManager: .default))
        try Data(#"{"weight_map":{"a":"../outside.safetensors"}}"#.utf8)
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertFalse(ModelWeightFileValidation.hasCompleteIndex(in: root))
    }

    func testPublicationForwardsCancellationBeforeCommitWithoutLosingOldWeights() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("model")
        let staging = root.appendingPathComponent("staging")
        try seedPublicationFixture(at: destination, value: 1)
        try seedPublicationFixture(at: staging, value: 2)
        let lease = try await ModelDiskOperations.prepareStagingDirectory(at: staging)
        let validating = expectation(description: "validation on disk queue")
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let publication = Task {
            try await ModelDiskOperations.publish(stagingLease: lease, to: destination) { directory in
                try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory)
                validating.fulfill()
                release.wait()
            }
        }
        await fulfillment(of: [validating], timeout: 2)
        publication.cancel()
        release.signal()
        do {
            _ = try await publication.value
            XCTFail("Cancelled publication committed")
        } catch is CancellationError {} catch { XCTFail("Unexpected error: \(error)") }
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("weights.safetensors")).last, 1)
        XCTAssertEqual(try Data(contentsOf: staging.appendingPathComponent("weights.safetensors")).last, 2)
    }

    func testPublicationResultRemainsInstalledAfterLateParentCancellation() async throws {
        let root = try TestDoubles.canonicalTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("model")
        let staging = root.appendingPathComponent("staging")
        try seedPublicationFixture(at: staging, value: 2)
        let lease = try await ModelDiskOperations.prepareStagingDirectory(at: staging)
        let publication = Task {
            let outcome = try await ModelDiskOperations.publish(stagingLease: lease, to: destination) { directory in
                try OmniModelArtifactValidation.validateSafetensorsDirectory(at: directory)
            }
            withUnsafeCurrentTask { $0?.cancel() }
            return outcome
        }
        let outcome = try await publication.value
        XCTAssertTrue(publication.isCancelled)
        XCTAssertEqual(outcome.directory.path, destination.path)
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("weights.safetensors")).last, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path))
    }

    private func seedPublicationFixture(at directory: URL, value: UInt8) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        let header = Data(#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#.utf8)
        var headerByteCount = UInt64(header.count).littleEndian
        var weights = withUnsafeBytes(of: &headerByteCount) { Data($0) }
        weights.append(header)
        weights.append(value)
        try weights.write(to: directory.appendingPathComponent("weights.safetensors"))
    }

    func testCheckingSnapshotCannotOfferInstallAction() {
        let snapshot = LocalModelInstallSnapshot(
            target: .mlx("repo"), state: .checking, isInstalled: false,
            isCurrentSelection: false, statusText: "", badgeText: nil, downloadStatus: nil,
            canOpenLocation: false, canConfigure: false, configureActionTitle: nil
        )
        let action = ModelSettingsInstallActionResolver.catalogPrimaryAction(for: snapshot) { _, _ in
            XCTFail("Checking must not start a download")
        }
        XCTAssertFalse(action?.isEnabled ?? true)
    }
}
