import Foundation
import OmniModelArtifacts
import XCTest

@testable import Voxt

@MainActor
final class SileroVADModelReaderTests: XCTestCase {
    private var root: URL!
    private var directory: URL!

    override func setUp() async throws {
        root = try TestDoubles.canonicalTemporaryDirectory().appendingPathComponent("voxt-silero-readers-\(UUID())")
        directory = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: directory.appendingPathComponent("config.json"))
        let header = Data("{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}".utf8)
        var headerByteCount = UInt64(header.count).littleEndian
        var weights = withUnsafeBytes(of: &headerByteCount) { Data($0) }
        weights.append(header)
        weights.append(Data(repeating: 0, count: 4))
        try weights.write(to: directory.appendingPathComponent("model.safetensors"))
    }

    override func tearDown() async throws {
        try FileManager.default.removeItem(at: root)
    }

    func testValidatedReaderRejectsDeleteUntilReleased() async throws {
        var reader: OmniModelDirectoryLease? = try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
        XCTAssertEqual(reader?.directory.path, try TestDoubles.canonicalFileURL(for: directory).path)
        do {
            try await ModelDiskOperations.removeModelDirectories([directory])
            XCTFail("A live Silero reader allowed deletion")
        } catch OmniModelStorageError.busy {}
        withExtendedLifetime(reader) {}
        reader = nil
        try await ModelDiskOperations.removeModelDirectories([directory])
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }

    func testMutationIsRejectedBeforeInvalidArtifactValidation() async throws {
        try Data("invalid".utf8).write(to: directory.appendingPathComponent("config.json"))
        let mutation = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
        defer { withExtendedLifetime(mutation) {} }
        do {
            _ = try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
            XCTFail("A mutating directory was validated")
        } catch OmniModelStorageError.busy {}
    }

    func testInvalidAndMissingArtifactsDoNotStrandReaders() async throws {
        try Data([0]).write(to: directory.appendingPathComponent("model.safetensors"))
        do {
            _ = try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
            XCTFail("A truncated safetensors file was accepted")
        } catch is OmniModelArtifactValidationError {}
        _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
        try FileManager.default.removeItem(at: directory)
        do {
            _ = try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
            XCTFail("A missing directory was accepted")
        } catch is OmniModelArtifactValidationError {}
        _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
    }

    func testCanonicalAliasAndConcurrentReadersExcludeMutation() async throws {
        let alias = root.appendingPathComponent("alias")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: directory)
        var first: OmniModelDirectoryLease? = try await SileroVADModelSupport.validatedModelDirectoryLease(from: alias)
        var second: OmniModelDirectoryLease? = try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
        XCTAssertEqual(first?.directory.path, second?.directory.path)
        first = nil
        do {
            _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
            XCTFail("One released reader invalidated a sibling reader")
        } catch OmniModelStorageError.busy {}
        withExtendedLifetime(second) {}
        second = nil
        _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
    }

    func testReaderRejectsAtomicReplacementUntilReleased() async throws {
        let staging = root.appendingPathComponent("staging")
        let stagingLease = try await ModelDiskOperations.prepareStagingDirectory(at: staging)
        try Data("replacement".utf8).write(to: staging.appendingPathComponent("config.json"))
        var reader: OmniModelDirectoryLease? = try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
        do {
            _ = try await ModelDiskOperations.publish(stagingLease: stagingLease, to: directory) { _ in }
            XCTFail("A live reader allowed model replacement")
        } catch OmniModelStorageError.busy {}
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("config.json")), Data("{}".utf8))
        withExtendedLifetime(reader) {}
        reader = nil
        _ = try await ModelDiskOperations.publish(stagingLease: stagingLease, to: directory) { _ in }
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("config.json")), Data("replacement".utf8))
    }

    func testCancelledAcquisitionDoesNotStrandAReader() async throws {
        let directory = try XCTUnwrap(directory)
        let operation = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await SileroVADModelSupport.validatedModelDirectoryLease(from: directory)
        }
        do { _ = try await operation.value; XCTFail("Cancelled acquisition succeeded") }
        catch is CancellationError {}
        _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
    }
}
