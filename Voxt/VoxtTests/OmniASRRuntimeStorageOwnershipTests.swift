import Foundation
import OmniModelArtifacts
import XCTest

@testable import Voxt

final class OmniASRRuntimeStorageOwnershipTests: XCTestCase {
    func testEveryNativeHTTPFamilyHoldsAReaderUntilRetirement() async throws {
        for kind in [OmniASRModelKind.qwen3ASR, .whisper, .cohereTranscribe, .mossTranscribeDiarize,
                     .nemotronASR, .parakeet, .senseVoice, .sileroVAD, .sortformer] {
            try await withFixture(kind: kind) { fixture in
                _ = try await fixture.runtime.prepare()
                try Self.assertBusy(fixture.directory)
                _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .sharedRead)
                await fixture.runtime.retire()
                _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
            }
        }
    }

    func testDeleteAndAtomicReplacementRejectALiveNativeReader() async throws {
        try await withFixture { fixture in
            let oldConfig = fixture.directory.appendingPathComponent("config.json")
            try Data("old".utf8).write(to: oldConfig)
            _ = try await fixture.runtime.prepare()
            do {
                try await ModelDiskOperations.removeModelDirectories([fixture.directory])
                XCTFail("An active server's model was deleted")
            } catch OmniModelStorageError.busy {}
            let staging = fixture.root.appendingPathComponent("staging")
            let stagingLease = try await ModelDiskOperations.prepareStagingDirectory(at: staging)
            try Data("new".utf8).write(to: staging.appendingPathComponent("config.json"))
            do {
                _ = try await ModelDiskOperations.publish(stagingLease: stagingLease, to: fixture.directory) { _ in }
                XCTFail("An active server's model was replaced")
            } catch OmniModelStorageError.busy {}
            XCTAssertEqual(try Data(contentsOf: oldConfig), Data("old".utf8))
            await fixture.runtime.retire()
            _ = try await ModelDiskOperations.publish(stagingLease: stagingLease, to: fixture.directory) { _ in }
            XCTAssertEqual(try Data(contentsOf: oldConfig), Data("new".utf8))
            try await ModelDiskOperations.removeModelDirectories([fixture.directory])
            XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.directory.path))
        }
    }

    func testMutationPreventsSpawnAndDoesNotSubstituteAnotherDirectory() async throws {
        try await withFixture { fixture in
            let mutation = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
            do {
                _ = try await fixture.runtime.prepare()
                XCTFail("A mutating model directory was used")
            } catch OmniModelStorageError.busy {}
            XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.started.path))
            withExtendedLifetime(mutation) {}
            await fixture.runtime.retire()
        }
    }

    func testRetirementKeepsTheReaderWhileWaitingForTheActualChildExit() async throws {
        try await withFixture(blockShutdown: true) { fixture in
            _ = try await fixture.runtime.prepare()
            let completion = Completion()
            let retirement = Task { await fixture.runtime.retire(); await completion.mark() }
            try await Self.waitUntil { FileManager.default.fileExists(atPath: fixture.shutdownRequested.path) }
            try Self.assertBusy(fixture.directory)
            let isComplete = await completion.finished
            XCTAssertFalse(isComplete)
            let state = await fixture.runtime.state
            XCTAssertEqual(state, .retiring)
            try Data().write(to: fixture.exitGate)
            await retirement.value
            _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
            let finalState = await fixture.runtime.state
            XCTAssertEqual(finalState, .stopped)
        }
    }

    func testIgnoringShutdownAndTerminationRetainsTheReaderUntilKillAndExit() async throws {
        try await withFixture(blockShutdown: true) { fixture in
            let script = try String(contentsOf: fixture.executable, encoding: .utf8)
                .replacingOccurrences(of: "#!/bin/sh\n", with: "#!/bin/sh\ntrap '' TERM\n")
            try script.write(to: fixture.executable, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fixture.executable.path)
            _ = try await fixture.runtime.prepare()
            let retirement = Task { await fixture.runtime.retire() }
            try await Self.waitUntil { FileManager.default.fileExists(atPath: fixture.shutdownRequested.path) }
            try Self.assertBusy(fixture.directory)
            await retirement.value
            _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
            XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.exitGate.path))
            let state = await fixture.runtime.state
            XCTAssertEqual(state, .stopped)
        }
    }

    func testAnActiveUseKeepsRetirementAndMutationBlocked() async throws {
        try await withFixture { fixture in
            _ = try await fixture.runtime.prepare()
            _ = try await fixture.runtime.beginUse()
            let retirement = Task { await fixture.runtime.retire() }
            try await Self.waitUntil { await fixture.runtime.state == .retiring }
            try Self.assertBusy(fixture.directory)
            XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.shutdownRequested.path))
            await fixture.runtime.endUse()
            await retirement.value
            _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
        }
    }

    func testSharedRuntimeDoesNotReplaceHealthyReadersForEquivalentDirectoryURLs() async throws {
        try await withFixture { fixture in
            let shared = OmniSharedModelRuntime(kind: .sortformer, configuration: {
                OmniBackendConfiguration(runtimeExecutable: fixture.executable)
            })
            let alias = fixture.root.appendingPathComponent("model-alias")
            try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.directory)
            do {
                let first = try await shared.acquire(modelDirectory: URL(fileURLWithPath: fixture.directory.path, isDirectory: true))
                let second = try await shared.acquire(modelDirectory: URL(fileURLWithPath: fixture.directory.path, isDirectory: false))
                let third = try await shared.acquire(modelDirectory: alias)
                XCTAssertEqual(first, second)
                XCTAssertEqual(second, third)
                await shared.release()
                await shared.release()
                try Self.assertBusy(fixture.directory)
                await shared.release()
                _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
            } catch {
                await shared.shutdownForApplicationTermination()
                throw error
            }
        }
    }

    func testSharedPreparationAndRepeatedRetirementDoNotStrandAReader() async throws {
        try await withFixture { fixture in
            async let first = fixture.runtime.prepare()
            async let second = fixture.runtime.prepare()
            let endpoints = try await (first, second)
            XCTAssertEqual(endpoints.0, endpoints.1)
            try Self.assertBusy(fixture.directory)
            async let firstRetirement: Void = fixture.runtime.retire()
            async let secondRetirement: Void = fixture.runtime.retire()
            _ = await (firstRetirement, secondRetirement)
            _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
        }
    }

    func testRetirementBeforePreparationDoesNotSpawnOrAcquireAReader() async throws {
        try await withFixture { fixture in
            await fixture.runtime.retire()
            do {
                _ = try await fixture.runtime.prepare()
                XCTFail("A retired runtime was launched")
            } catch OmniASRRuntimeError.retired {}
            XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.started.path))
            _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
        }
    }

    func testSpawnFailureReleasesItsReader() async throws {
        try await withFixture { fixture in
            try FileManager.default.removeItem(at: fixture.executable)
            do {
                _ = try await fixture.runtime.prepare()
                XCTFail("A missing executable was launched")
            } catch OmniASRRuntimeError.launchFailed {}
            _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
        }
    }

    func testFailedStartupAndStartupDeadlineReleaseReadersOnlyAfterExit() async throws {
        for script in ["#!/bin/sh\necho '{\"event\":\"failed\",\"reason\":\"fixture\"}'\n",
                       "#!/bin/sh\nexec sleep 60\n"] {
            let startupTimeoutSeconds = script.contains("exec sleep") ? 0.2 : 5
            try await withFixture(startupTimeoutSeconds: startupTimeoutSeconds) { fixture in
                try script.write(to: fixture.executable, atomically: true, encoding: .utf8)
                try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fixture.executable.path)
                do {
                    _ = try await fixture.runtime.prepare()
                    XCTFail("An unavailable fixture reported ready")
                } catch OmniASRRuntimeError.launchFailed(let reason) {
                    XCTAssertEqual(reason, script.contains("exec sleep")
                        ? "the server did not report ready within 0.2 s" : "fixture")
                }
                _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
            }
        }
    }

    func testCanonicalSymlinkReaderBlocksMutationOfTheResolvedModel() async throws {
        try await withFixture { fixture in
            let alias = fixture.root.appendingPathComponent("model-alias")
            try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: fixture.directory)
            let aliasedRuntime = OmniASRRuntime(kind: .sortformer, modelDirectory: alias,
                configuration: OmniBackendConfiguration(runtimeExecutable: fixture.executable))
            _ = try await aliasedRuntime.prepare()
            try Self.assertBusy(fixture.directory)
            let arguments = try String(contentsOf: fixture.arguments, encoding: .utf8)
            XCTAssertTrue(arguments.contains(fixture.directory.resolvingSymlinksInPath().path))
            await aliasedRuntime.retire()
            _ = try OmniModelDirectoryLease.acquire(for: fixture.directory, mode: .exclusiveMutation)
        }
    }

    func testAChildRetainsItsReaderAfterTheCallerDropsTheRuntime() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-reader-drop-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let directory = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let gate = root.appendingPathComponent("exit")
        let executable = root.appendingPathComponent("server")
        let script = """
        #!/bin/sh
        printf '{"event":"ready","host":"127.0.0.1","port":43210,"model_name":"fixture","server_pid":%s}\\n' "$$"
        while [ ! -f '\(gate.path)' ]; do sleep 0.02; done
        """
        try script.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        var runtime: OmniASRRuntime? = OmniASRRuntime(kind: .sortformer, modelDirectory: directory,
            configuration: OmniBackendConfiguration(runtimeExecutable: executable))
        _ = try await runtime?.prepare()
        runtime = nil
        do {
            try Self.assertBusy(directory)
            try Data().write(to: gate)
            try await Self.waitUntil {
                do { _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation); return true }
                catch OmniModelStorageError.busy { return false }
            }
            try FileManager.default.removeItem(at: root)
        } catch {
            try Data().write(to: gate)
            throw error
        }
    }

    private struct Fixture: Sendable {
        let root: URL
        let directory: URL
        let executable: URL
        let arguments: URL
        let started: URL
        let shutdownRequested: URL
        let exitGate: URL
        let runtime: OmniASRRuntime
    }

    private func withFixture(
        kind: OmniASRModelKind = .sortformer, blockShutdown: Bool = false,
        startupTimeoutSeconds: Double = 5,
        operation: @Sendable (Fixture) async throws -> Void
    ) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("voxt-reader-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let directory = root.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let executable = root.appendingPathComponent("server")
        let arguments = root.appendingPathComponent("arguments")
        let started = root.appendingPathComponent("started")
        let shutdownRequested = root.appendingPathComponent("shutdown-requested")
        let exitGate = root.appendingPathComponent("exit")
        let script = """
        #!/bin/sh
        printf '%s\\n' "$@" > '\(arguments.path)'
        : > '\(started.path)'
        printf '{"event":"ready","host":"127.0.0.1","port":43210,"model_name":"fixture","server_pid":%s}\\n' "$$"
        IFS= read -r command
        : > '\(shutdownRequested.path)'
        \(blockShutdown ? "while [ ! -f '\(exitGate.path)' ]; do sleep 0.02; done" : "exit 0")
        """
        try script.write(to: executable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let fixture = Fixture(root: root, directory: directory, executable: executable, arguments: arguments,
            started: started, shutdownRequested: shutdownRequested, exitGate: exitGate,
            runtime: OmniASRRuntime(kind: kind, modelDirectory: directory,
                configuration: OmniBackendConfiguration(runtimeExecutable: executable,
                    startupTimeoutSeconds: startupTimeoutSeconds)))
        do {
            try await operation(fixture)
        } catch {
            try Data().write(to: exitGate)
            await fixture.runtime.retire()
            try FileManager.default.removeItem(at: root)
            throw error
        }
        try Data().write(to: exitGate)
        await fixture.runtime.retire()
        try FileManager.default.removeItem(at: root)
    }

    private static func assertBusy(_ directory: URL) throws {
        do {
            _ = try OmniModelDirectoryLease.acquire(for: directory, mode: .exclusiveMutation)
            XCTFail("An active Native process lost its model reader")
        } catch OmniModelStorageError.busy {}
    }

    private static func waitUntil(_ predicate: @Sendable () async throws -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while try await !predicate(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        let succeeded = try await predicate()
        XCTAssertTrue(succeeded, "The expected lifecycle transition did not complete")
    }
}

private actor Completion {
    var finished = false
    func mark() { finished = true }
}
