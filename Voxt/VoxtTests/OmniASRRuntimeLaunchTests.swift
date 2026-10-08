// OmniASRRuntimeLaunchTests.swift
// Covers how the Omni runtime starts its supervisor process.

import XCTest
@testable import Voxt

final class OmniASRRuntimeLaunchTests: XCTestCase {
    func testSupervisorEnvironmentDropsInjectedDynamicLoaderVariables() {
        let inherited = [
            "PATH": "/usr/bin",
            "HOME": "/Users/someone",
            "DYLD_INSERT_LIBRARIES": "/Xcode/libXCTestBundleInject.dylib",
            "DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
            "DYLD_FRAMEWORK_PATH": "/Xcode/Frameworks",
            "__XPC_DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
            "PYTHONPATH": "/somewhere/else",
        ]

        let environment = OmniASRRuntime.supervisorEnvironment(
            inheriting: inherited,
            backendDirectory: URL(fileURLWithPath: "/backend", isDirectory: true)
        )

        XCTAssertEqual(environment["PATH"], "/usr/bin")
        XCTAssertEqual(environment["HOME"], "/Users/someone")
        XCTAssertEqual(environment["PYTHONPATH"], "/backend")
        XCTAssertEqual(environment["PYTHONUNBUFFERED"], "1")
        XCTAssertEqual(
            environment.keys.filter { $0.hasPrefix("DYLD_") || $0.hasPrefix("__XPC_DYLD_") },
            []
        )
    }

    func testDiagnosticTailKeepsOnlyTheEndOfLongOutput() {
        let tail = OmniDiagnosticTail(limit: 8)
        tail.append(Data("0123456789".utf8))
        tail.append(Data("ab".utf8))

        XCTAssertEqual(tail.text, "456789ab")
    }
}

@MainActor
final class OmniModelManagerRecoveryTests: XCTestCase {
    private actor LoadCounter {
        private(set) var value = 0

        func increment() {
            value += 1
        }
    }

    /// A server that crashed or was killed must not keep failing every request
    /// until the idle unload: the next load replaces the runtime.
    func testALoadedOmniRuntimeThatStoppedServingIsReplacedOnTheNextLoad() async throws {
        let scratch = FileManager.default.temporaryDirectory
        let configuration = OmniBackendConfiguration(
            pythonExecutable: URL(fileURLWithPath: "/usr/bin/false"),
            backendDirectory: scratch,
            derivedRoot: scratch
        )
        let loads = LoadCounter()
        let manager = MLXModelManager(modelRepo: "mlx-community/Qwen3-ASR-0.6B-4bit") { _ in
            await loads.increment()
            // Never launched, so never serving: the same answer a dead server gives.
            let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
            return MLXLoadedModelBox(loaded: .omni(runtime))
        }

        let first = try await manager.loadModel()
        let second = try await manager.loadModel()

        XCTAssertFalse(first.omniRuntime === second.omniRuntime)
        let loadCount = await loads.value
        XCTAssertEqual(loadCount, 2)
        await manager.shutdownForApplicationTermination()
    }
}
