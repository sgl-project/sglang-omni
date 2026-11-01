import Foundation
@testable import OmniModelArtifacts
import XCTest
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

final class OmniModelPreparationTests: XCTestCase {
    private var root: URL!
    private var source: URL!
    private var destination: URL!

    override func setUpWithError() throws {
        root = try modelStorageTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        source = root.appendingPathComponent("installed")
        destination = root.appendingPathComponent("prepared")
        try writeModel(at: source, value: 1)
    }

    override func tearDownWithError() throws { try FileManager.default.removeItem(at: root) }

    func testCopiesWithoutMutatingInstalledWeightsOrLeavingLinks() throws {
        let blob = root.appendingPathComponent("tokenizer-blob")
        try Data("tokenizer".utf8).write(to: blob)
        try FileManager.default.createSymbolicLink(at: source.appendingPathComponent("tokenizer.model"), withDestinationURL: blob)
        let reader = try OmniModelDirectoryLease.acquire(for: source, mode: .sharedRead)
        let outcome = try OmniModelPreparation.prepare(from: source, to: destination)
        XCTAssertEqual(outcome.directory.path, destination.path)
        XCTAssertNil(outcome.retainedPreviousDirectory)
        XCTAssertEqual(try payload(at: destination), 1)
        XCTAssertEqual(try payload(at: source), 1)
        XCTAssertThrowsError(try FileManager.default.destinationOfSymbolicLink(atPath: destination.appendingPathComponent("tokenizer.model").path))
        try Data("changed blob".utf8).write(to: blob)
        XCTAssertEqual(try String(contentsOf: destination.appendingPathComponent("tokenizer.model"), encoding: .utf8), "tokenizer")
        withExtendedLifetime(reader) {}
    }

    func testExistingDestinationRequiresExplicitReplacement() throws {
        try writeModel(at: destination, value: 2)
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination)) {
            XCTAssertEqual($0 as? OmniModelPreparationError, .destinationExists)
        }
        XCTAssertEqual(try payload(at: destination), 2)
        let outcome = try OmniModelPreparation.prepare(from: source, to: destination, replaceExisting: true)
        XCTAssertEqual(try payload(at: outcome.directory), 1)
        XCTAssertEqual(try payload(at: source), 1)
    }

    func testReadersExcludeReplacementAndSourceWritersExcludeCopy() throws {
        try writeModel(at: destination, value: 2)
        var reader: OmniModelDirectoryLease? = try OmniModelDirectoryLease.acquire(for: destination, mode: .sharedRead)
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination, replaceExisting: true)) {
            guard case OmniModelStorageError.busy = $0 else { return XCTFail("Expected reader exclusion: \($0)") }
        }
        XCTAssertEqual(try payload(at: destination), 2)
        reader = nil
        withExtendedLifetime(reader) {}
        let writer = try OmniModelDirectoryLease.acquire(for: source, mode: .exclusiveMutation)
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination, replaceExisting: true)) {
            guard case OmniModelStorageError.busy = $0 else { return XCTFail("Expected writer exclusion: \($0)") }
        }
        withExtendedLifetime(writer) {}
    }

    func testCancellationAtValidationPreservesOldPayloadAndRetainsStage() throws {
        try writeModel(at: destination, value: 2)
        var cancelled = false
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination, replaceExisting: true,
            validate: { _ in cancelled = true }, checkCancellation: { if cancelled { throw CancellationError() } })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(try payload(at: destination), 2)
        let stages = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix(".omni-prepare-") }
        XCTAssertEqual(stages.count, 1)
        XCTAssertEqual(try payload(at: source), 1)
    }

    func testValidationRejectsBeforeCommitAndSourceReaderSpansIt() throws {
        try writeModel(at: destination, value: 2)
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination, replaceExisting: true,
            validate: { _ in
                XCTAssertThrowsError(try OmniModelDirectoryLease.acquire(for: self.source, mode: .exclusiveMutation))
                throw CancellationError()
            })) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(try payload(at: destination), 2)
        _ = try OmniModelDirectoryLease.acquire(for: source, mode: .exclusiveMutation)
    }

    func testInvalidPayloadAndDirectoryLinksRejectWithoutPublishing() throws {
        try Data("bad".utf8).write(to: source.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
        try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
        try FileManager.default.createSymbolicLink(at: source.appendingPathComponent("external"), withDestinationURL: root)
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    func testFIFOAndDanglingLinksRejectWithoutBlockingCopy() throws {
        let fifo = source.appendingPathComponent("pipe")
        XCTAssertEqual(mkfifo(fifo.path, 0o600), 0)
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination))
        try FileManager.default.removeItem(at: fifo)
        try FileManager.default.createSymbolicLink(at: source.appendingPathComponent("dangling"), withDestinationURL: root.appendingPathComponent("missing"))
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination))
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    func testByteBudgetRejectsSparseExtraFileBeforeCopyingIt() throws {
        let oversized = source.appendingPathComponent("oversized.bin")
        XCTAssertTrue(FileManager.default.createFile(atPath: oversized.path, contents: Data()))
        XCTAssertEqual(truncate(oversized.path, off_t(OmniModelPreparation.maximumTotalFileByteCount + 1)), 0)
        XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: destination)) {
            XCTAssertEqual($0 as? OmniModelPreparationError, .byteLimitExceeded)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))
    }

    func testSameAndNestedSourceDestinationsReject() throws {
        for target in [source!, source.appendingPathComponent("nested"), root!] {
            XCTAssertThrowsError(try OmniModelPreparation.prepare(from: source, to: target))
        }
        XCTAssertEqual(try payload(at: source), 1)
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

    private func payload(at directory: URL) throws -> UInt8? {
        try Data(contentsOf: directory.appendingPathComponent("model.safetensors")).last
    }
}
