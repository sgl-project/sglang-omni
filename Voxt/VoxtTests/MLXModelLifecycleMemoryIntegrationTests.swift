// MLXModelLifecycleMemoryIntegrationTests.swift
// Runs opt-in, same-process lifecycle stress tests against an installed MLX ASR model.

import Darwin
import MLX
import XCTest
@testable import Voxt

@MainActor
final class MLXModelLifecycleMemoryIntegrationTests: XCTestCase {
    private struct MemorySample {
        let cycle: Int
        let physicalFootprint: UInt64
        let mlxActive: Int
        let mlxCache: Int
        let mallocPressureRelief: Int
        let elapsedMilliseconds: Int
    }

    func testRepeatedInferenceAndReleaseReportsStableIdleMemory() async throws {
        try ModelTestGate.requireEnabled("MLX model lifecycle memory integration test")

        let environment = ProcessInfo.processInfo.environment
        let modelRoot = environment["VOXT_MODEL_STORAGE_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let modelRoot, !modelRoot.isEmpty else {
            throw XCTSkip("Set VOXT_MODEL_STORAGE_ROOT to an installed Voxt model directory.")
        }

        let repo = environment["VOXT_MEMORY_STRESS_MODEL_REPO"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty
            ?? "mlx-community/Qwen3-ASR-1.7B-4bit"
        let cycleCount = max(Int(environment["VOXT_MEMORY_STRESS_CYCLES"] ?? "") ?? 3, 2)
        let loadOnly = environment["VOXT_MEMORY_STRESS_LOAD_ONLY"] == "1"
        let settleSeconds = max(Double(environment["VOXT_MEMORY_STRESS_SETTLE_SECONDS"] ?? "") ?? 0, 0)
        let holdSeconds = max(Double(environment["VOXT_MEMORY_STRESS_HOLD_SECONDS"] ?? "") ?? 0, 0)
        let fixtureURL = try qwenFixtureURL(named: "qwen_audio_short_zh_chongqing.wav")

        configureModelStorageRoot(modelRoot)
        let probeManager = MLXModelManager(modelRepo: repo)
        _ = try await probeManager.refreshInstallation(repo: repo)
        guard probeManager.isModelDownloaded(repo: repo) else {
            throw XCTSkip("The requested memory-stress model is not installed: \(repo)")
        }
        await probeManager.shutdownForApplicationTermination()

        let baseline = await settledMemorySample(cycle: 0)
        report(baseline, repo: repo)

        var allSamples = [baseline]
        var samples: [MemorySample] = []
        for cycle in 1...cycleCount {
            let cycleStartedAt = Date()
            let transcript: String
            do {
                transcript = try await runAndReleaseCycle(
                    repo: repo,
                    fixtureURL: fixtureURL,
                    loadOnly: loadOnly
                )
            } catch {
                throw NSError(
                    domain: "VoxtTests.MLXModelLifecycle",
                    code: cycle,
                    userInfo: [
                        NSLocalizedDescriptionKey: "ASR lifecycle cycle \(cycle) failed: \(error)"
                    ]
                )
            }
            if !loadOnly {
                XCTAssertFalse(
                    transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                    "The real-model cycle should complete inference before measuring cleanup."
                )
            }
            if settleSeconds > 0 {
                try await Task.sleep(for: .seconds(settleSeconds))
            }

            let sample = await settledMemorySample(
                cycle: cycle,
                elapsedMilliseconds: Int(Date().timeIntervalSince(cycleStartedAt) * 1_000)
            )
            samples.append(sample)
            allSamples.append(sample)
            report(sample, repo: repo)
            try writeProgressReportIfRequested(samples: allSamples, repo: repo)
        }

        let first = try XCTUnwrap(samples.first)
        let last = try XCTUnwrap(samples.last)
        let footprintGrowth = Int64(last.physicalFootprint) - Int64(first.physicalFootprint)
        let activeGrowth = last.mlxActive - first.mlxActive
        print(
            "MLX_LIFECYCLE_STRESS summary repo=\(repo) cycles=\(cycleCount) "
                + "idleFootprintGrowthBytes=\(footprintGrowth) mlxActiveGrowthBytes=\(activeGrowth)"
        )
        try writeReportIfRequested(
            samples: allSamples,
            repo: repo,
            footprintGrowth: footprintGrowth,
            activeGrowth: activeGrowth
        )
        if holdSeconds > 0 {
            print(
                "MLX_LIFECYCLE_STRESS holding pid=\(ProcessInfo.processInfo.processIdentifier) "
                    + "seconds=\(holdSeconds)"
            )
            try await Task.sleep(for: .seconds(holdSeconds))
        }

        XCTAssertLessThanOrEqual(
            activeGrowth,
            1024,
            "MLX active allocations should not accumulate across completed model lifecycles."
        )
        assertConfiguredMemoryLimits(samples: samples, footprintGrowth: footprintGrowth)
    }

    func testRepeatedLLMLoadAndReleaseReportsStableIdleMemory() async throws {
        try ModelTestGate.requireEnabled("MLX LLM lifecycle memory integration test")

        let environment = ProcessInfo.processInfo.environment
        let modelRoot = environment["VOXT_MODEL_STORAGE_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let modelRoot, !modelRoot.isEmpty else {
            throw XCTSkip("Set VOXT_MODEL_STORAGE_ROOT to an installed Voxt model directory.")
        }

        let repo = environment["VOXT_MEMORY_STRESS_LLM_REPO"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty
            ?? "mlx-community/gemma-4-e2b-it-4bit"
        let cycleCount = max(Int(environment["VOXT_MEMORY_STRESS_CYCLES"] ?? "") ?? 3, 2)
        let holdSeconds = max(Double(environment["VOXT_MEMORY_STRESS_HOLD_SECONDS"] ?? "") ?? 0, 0)

        configureModelStorageRoot(modelRoot)
        let probeManager = CustomLLMModelManager(modelRepo: repo)
        _ = try await probeManager.refreshInstallation(repo: repo)
        guard probeManager.isModelDownloaded(repo: repo) else {
            throw XCTSkip("The requested memory-stress LLM is not installed: \(repo)")
        }
        await probeManager.shutdownForApplicationTermination()

        let baseline = await settledMemorySample(cycle: 0)
        report(baseline, repo: repo)
        var allSamples = [baseline]
        var samples: [MemorySample] = []
        for cycle in 1...cycleCount {
            let cycleStartedAt = Date()
            let manager = CustomLLMModelManager(modelRepo: repo)
            try await manager.prewarmModel(repo: repo)
            XCTAssertTrue(manager.hasLoadedInferenceModel)
            await manager.shutdownForApplicationTermination()
            XCTAssertFalse(manager.hasLoadedInferenceModel)

            let sample = await settledMemorySample(
                cycle: cycle,
                elapsedMilliseconds: Int(Date().timeIntervalSince(cycleStartedAt) * 1_000)
            )
            samples.append(sample)
            allSamples.append(sample)
            report(sample, repo: repo)
            try writeProgressReportIfRequested(samples: allSamples, repo: repo)
        }

        let first = try XCTUnwrap(samples.first)
        let last = try XCTUnwrap(samples.last)
        let footprintGrowth = Int64(last.physicalFootprint) - Int64(first.physicalFootprint)
        let activeGrowth = last.mlxActive - first.mlxActive
        print(
            "MLX_LLM_LIFECYCLE_STRESS summary repo=\(repo) cycles=\(cycleCount) "
                + "idleFootprintGrowthBytes=\(footprintGrowth) mlxActiveGrowthBytes=\(activeGrowth)"
        )
        try writeReportIfRequested(
            samples: allSamples,
            repo: repo,
            footprintGrowth: footprintGrowth,
            activeGrowth: activeGrowth
        )
        if holdSeconds > 0 {
            print(
                "MLX_LLM_LIFECYCLE_STRESS holding pid=\(ProcessInfo.processInfo.processIdentifier) "
                    + "seconds=\(holdSeconds)"
            )
            try await Task.sleep(for: .seconds(holdSeconds))
        }
        XCTAssertLessThanOrEqual(
            activeGrowth,
            1024,
            "MLX active allocations should not accumulate across completed LLM lifecycles."
        )
        assertConfiguredMemoryLimits(samples: samples, footprintGrowth: footprintGrowth)
    }

    func testMemoryEfficientLLMLoaderCompletesRealGeneration() async throws {
        try ModelTestGate.requireEnabled("MLX memory-efficient LLM generation integration test")

        let environment = ProcessInfo.processInfo.environment
        let modelRoot = environment["VOXT_MODEL_STORAGE_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let modelRoot, !modelRoot.isEmpty else {
            throw XCTSkip("Set VOXT_MODEL_STORAGE_ROOT to an installed Voxt model directory.")
        }
        let repo = environment["VOXT_MEMORY_STRESS_LLM_REPO"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty
            ?? "mlx-community/gemma-4-e2b-it-4bit"

        configureModelStorageRoot(modelRoot)
        let manager = CustomLLMModelManager(modelRepo: repo)
        _ = try await manager.refreshInstallation(repo: repo)
        guard manager.isModelDownloaded(repo: repo) else {
            throw XCTSkip("The requested generation-test LLM is not installed: \(repo)")
        }
        do {
            let output = try await manager.enhance(
                userPrompt: "请只回复：测试通过",
                repo: repo
            )
            XCTAssertFalse(output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            await manager.shutdownForApplicationTermination()
        } catch {
            await manager.shutdownForApplicationTermination()
            throw error
        }
    }

    private func runAndReleaseCycle(
        repo: String,
        fixtureURL: URL,
        loadOnly: Bool
    ) async throws -> String {
        let manager = MLXModelManager(modelRepo: repo)

        if loadOnly {
            do {
                print("MLX_LIFECYCLE_STRESS loadOnly phase=begin repo=\(repo)")
                manager.beginActiveUse()
                _ = try await manager.loadModel()
                print("MLX_LIFECYCLE_STRESS loadOnly phase=loaded repo=\(repo)")
                manager.endActiveUse()
                await manager.shutdownForApplicationTermination()
                print("MLX_LIFECYCLE_STRESS loadOnly phase=released repo=\(repo)")
                XCTAssertFalse(manager.hasLoadedModel)
                XCTAssertFalse(manager.hasPendingModelLoad)
                return ""
            } catch {
                print("MLX_LIFECYCLE_STRESS loadOnly phase=failed repo=\(repo) error=\(error)")
                manager.endActiveUse()
                await manager.shutdownForApplicationTermination()
                throw error
            }
        }

        let transcriber = MLXTranscriber(modelManager: manager)
        do {
            let transcript = try await transcriber.transcribeAudioFile(fixtureURL)
            await transcriber.shutdownForApplicationTermination()
            _ = transcriber.releaseIdleResources()
            await manager.shutdownForApplicationTermination()
            XCTAssertFalse(manager.hasLoadedModel)
            XCTAssertFalse(manager.hasPendingModelLoad)
            return transcript
        } catch {
            await transcriber.shutdownForApplicationTermination()
            _ = transcriber.releaseIdleResources()
            await manager.shutdownForApplicationTermination()
            throw error
        }
    }

    private func settledMemorySample(
        cycle: Int,
        elapsedMilliseconds: Int = 0
    ) async -> MemorySample {
        // This fixture has explicitly stopped its model work. Settle GPU queues
        // off MainActor for comparable measurements; production idle cleanup must
        // not insert these global fences into interactive sessions.
        await Task.detached(priority: .utility) {
            Stream.gpu.synchronize()
            Stream.cpu.synchronize()
        }.value
        let reclaimedBytes = await IdleMemoryReclamationSupport.releaseAllocatorCaches()
        let mlx = Memory.snapshot()
        return MemorySample(
            cycle: cycle,
            physicalFootprint: physicalFootprintBytes(),
            mlxActive: mlx.activeMemory,
            mlxCache: mlx.cacheMemory,
            mallocPressureRelief: reclaimedBytes,
            elapsedMilliseconds: elapsedMilliseconds
        )
    }

    private func physicalFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) { infoPointer in
            infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    reboundPointer,
                    &count
                )
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }

    private func assertConfiguredMemoryLimits(
        samples: [MemorySample],
        footprintGrowth: Int64
    ) {
        let environment = ProcessInfo.processInfo.environment
        if let maximumIdleFootprint = UInt64(
            environment["VOXT_MEMORY_STRESS_MAX_IDLE_FOOTPRINT_BYTES"] ?? ""
        ) {
            let measuredMaximum = samples.map(\.physicalFootprint).max() ?? 0
            XCTAssertLessThanOrEqual(
                measuredMaximum,
                maximumIdleFootprint,
                "An idle physical footprint sample exceeded the configured memory target."
            )
        }
        if let maximumGrowth = Int64(
            environment["VOXT_MEMORY_STRESS_MAX_GROWTH_BYTES"] ?? ""
        ) {
            XCTAssertLessThanOrEqual(
                footprintGrowth,
                maximumGrowth,
                "Idle physical footprint growth exceeded the configured memory target."
            )
        }
    }

    private func report(_ sample: MemorySample, repo: String) {
        print(reportLine(sample, repo: repo))
    }

    private func reportLine(_ sample: MemorySample, repo: String) -> String {
        "MLX_LIFECYCLE_STRESS cycle=\(sample.cycle) repo=\(repo) "
            + "physicalFootprintBytes=\(sample.physicalFootprint) "
            + "mlxActiveBytes=\(sample.mlxActive) mlxCacheBytes=\(sample.mlxCache) "
            + "mallocPressureReliefBytes=\(sample.mallocPressureRelief) "
            + "elapsedMilliseconds=\(sample.elapsedMilliseconds)"
    }

    private func writeReportIfRequested(
        samples: [MemorySample],
        repo: String,
        footprintGrowth: Int64,
        activeGrowth: Int
    ) throws {
        guard let reportPath = ProcessInfo.processInfo.environment["VOXT_MEMORY_STRESS_REPORT_PATH"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty
        else { return }

        let summary = "MLX_LIFECYCLE_STRESS summary pid=\(ProcessInfo.processInfo.processIdentifier) "
            + "repo=\(repo) cycles=\(samples.count - 1) "
            + "idleFootprintGrowthBytes=\(footprintGrowth) mlxActiveGrowthBytes=\(activeGrowth)"
        let contents = (samples.map { reportLine($0, repo: repo) } + [summary])
            .joined(separator: "\n") + "\n"
        try contents.write(
            to: URL(fileURLWithPath: reportPath),
            atomically: true,
            encoding: .utf8
        )
    }

    private func writeProgressReportIfRequested(
        samples: [MemorySample],
        repo: String
    ) throws {
        guard let first = samples.dropFirst().first,
              let last = samples.last
        else { return }

        try writeReportIfRequested(
            samples: samples,
            repo: repo,
            footprintGrowth: Int64(last.physicalFootprint) - Int64(first.physicalFootprint),
            activeGrowth: last.mlxActive - first.mlxActive
        )
    }

    private func configureModelStorageRoot(_ modelRoot: String) {
        let defaults = UserDefaults.standard
        let previousPath = defaults.string(forKey: AppPreferenceKey.modelStorageRootPath)
        let previousBookmark = defaults.data(forKey: AppPreferenceKey.modelStorageRootBookmark)

        defaults.set(modelRoot, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
        ModelStorageDirectoryManager.resetForTesting()
        ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(
            URL(fileURLWithPath: modelRoot, isDirectory: true)
        )

        addTeardownBlock {
            if let previousPath {
                defaults.set(previousPath, forKey: AppPreferenceKey.modelStorageRootPath)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
            }
            if let previousBookmark {
                defaults.set(previousBookmark, forKey: AppPreferenceKey.modelStorageRootBookmark)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
            }
            ModelStorageDirectoryManager.resetForTesting()
        }
    }

    private func qwenFixtureURL(named fileName: String) throws -> URL {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Audio/qwen-official", isDirectory: true)
            .appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Missing official audio fixture: \(fileName)")
        }
        return url
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
