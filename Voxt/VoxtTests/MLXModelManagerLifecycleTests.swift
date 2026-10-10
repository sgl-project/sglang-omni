import XCTest
@testable import Voxt
import HuggingFace
import MLX
import MLXAudioSTT

@MainActor
final class MLXModelManagerLifecycleTests: MLXModelManagerTestCase {
    func testSharedModelLoadCoordinatorDeduplicatesConcurrentWaiters() async throws {
        let loader = ControlledSharedValueLoader()
        let coordinator = SharedModelLoadCoordinator<Int>()
        let firstTask = Task<Int, Error> { @MainActor in
            try await coordinator.value(for: "same-model") {
                await loader.load()
            }
        }
        await loader.waitUntilStarted()
        let secondTask = Task<Int, Error> { @MainActor in
            try await coordinator.value(for: "same-model") {
                await loader.load()
            }
        }
        await Task.yield()

        let invocationCount = await loader.count()
        XCTAssertEqual(invocationCount, 1)
        await loader.finish(with: 42)
        let firstValue = try await firstTask.value
        let secondValue = try await secondTask.value

        XCTAssertEqual(firstValue, 42)
        XCTAssertEqual(secondValue, 42)
        XCTAssertFalse(coordinator.hasPendingLoad)
    }

    func testSharedModelLoadCoordinatorDoesNotCancelRemainingWaiter() async throws {
        let loader = ControlledSharedValueLoader()
        let coordinator = SharedModelLoadCoordinator<Int>()
        let firstTask = Task<Int, Error> { @MainActor in
            try await coordinator.value(for: "same-model") {
                await loader.load()
            }
        }
        await loader.waitUntilStarted()
        let secondTask = Task<Int, Error> { @MainActor in
            try await coordinator.value(for: "same-model") {
                await loader.load()
            }
        }
        await Task.yield()
        firstTask.cancel()
        await Task.yield()

        XCTAssertTrue(coordinator.hasPendingLoad)
        let invocationCount = await loader.count()
        XCTAssertEqual(invocationCount, 1)

        await loader.finish(with: 7)
        let secondValue = try await secondTask.value
        XCTAssertEqual(secondValue, 7)
        do {
            _ = try await firstTask.value
            XCTFail("The cancelled waiter must not receive the shared result.")
        } catch is CancellationError {
            // Expected.
        }
    }

    func testSharedModelLoadCoordinatorRejectsResultAfterExplicitInvalidation() async {
        let loader = ControlledSharedValueLoader()
        let coordinator = SharedModelLoadCoordinator<Int>()
        let loadTask = Task<Int, Error> { @MainActor in
            try await coordinator.value(for: "same-model") {
                await loader.load()
            }
        }

        await loader.waitUntilStarted()
        let invalidatedTasks = coordinator.cancelAll()
        XCTAssertEqual(invalidatedTasks.count, 1)
        XCTAssertFalse(coordinator.hasPendingLoad)

        await loader.finish(with: 99)
        do {
            _ = try await loadTask.value
            XCTFail("An invalidated generation must never return a stale model result.")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("Expected CancellationError, got \(error).")
        }
    }

    func testCancellingOnlyModelLoadWaiterInvalidatesStaleResult() async {
        let loader = ControlledMLXModelLoader()
        let manager = MLXModelManager(
            modelRepo: MLXModelManager.defaultModelRepo,
            modelLoadingOverride: { _ in await loader.load() }
        )
        manager.beginActiveUse()
        let loadTask = Task<Void, Error> { @MainActor in
            defer { manager.endActiveUse() }
            _ = try await manager.loadModel()
        }

        await loader.waitUntilStarted()
        loadTask.cancel()
        await waitUntilPendingModelLoadClears(manager)

        XCTAssertFalse(manager.hasPendingModelLoad)
        XCTAssertFalse(manager.hasLoadedModel)

        await loader.finish()
        do {
            _ = try await loadTask.value
            XCTFail("A cancelled model load waiter must not receive or install a stale model.")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("Expected CancellationError, got \(error).")
        }

        XCTAssertFalse(manager.hasLoadedModel)
        await manager.shutdownForApplicationTermination()
    }

    func testApplicationTerminationShutdownWaitsForModelLoadCancelledBeforeShutdown() async {
        let loader = ControlledMLXModelLoader()
        let manager = MLXModelManager(
            modelRepo: MLXModelManager.defaultModelRepo,
            modelLoadingOverride: { _ in await loader.load() }
        )
        let loadTask = Task<Void, Error> { @MainActor in
            _ = try await manager.loadModel()
        }

        await loader.waitUntilStarted()
        manager.cancelPendingModelLoadForApplicationTermination()

        let completionProbe = AsyncCompletionProbe()
        let shutdownTask = Task { @MainActor in
            await manager.shutdownForApplicationTermination()
            await completionProbe.markCompleted()
        }
        try? await Task.sleep(for: .milliseconds(25))

        let completedBeforeLoaderStopped = await completionProbe.isCompleted()
        XCTAssertFalse(
            completedBeforeLoaderStopped,
            "Shutdown must retain and await a model load cancelled by the earlier termination phase."
        )

        await loader.finish()
        await shutdownTask.value
        let completedAfterLoaderStopped = await completionProbe.isCompleted()
        XCTAssertTrue(completedAfterLoaderStopped)

        do {
            _ = try await loadTask.value
            XCTFail("The invalidated model load must not install its stale result.")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("Expected CancellationError, got \(error).")
        }
        XCTAssertFalse(manager.hasLoadedModel)
    }

    func testCancellingOneOfMultipleModelLoadWaitersKeepsSharedLoadAlive() async throws {
        let loader = ControlledMLXModelLoader()
        let manager = MLXModelManager(
            modelRepo: MLXModelManager.defaultModelRepo,
            modelLoadingOverride: { _ in await loader.load() }
        )
        manager.beginActiveUse()
        manager.beginActiveUse()
        let firstTask = Task<Void, Error> { @MainActor in
            defer { manager.endActiveUse() }
            _ = try await manager.loadModel()
        }
        let secondTask = Task<Void, Error> { @MainActor in
            defer { manager.endActiveUse() }
            _ = try await manager.loadModel()
        }

        await loader.waitUntilStarted()
        firstTask.cancel()
        await Task.yield()

        XCTAssertTrue(manager.hasPendingModelLoad)
        XCTAssertFalse(manager.hasLoadedModel)

        await loader.finish()
        _ = try await secondTask.value
        XCTAssertTrue(manager.hasLoadedModel)

        do {
            _ = try await firstTask.value
            XCTFail("The cancelled waiter must remain cancelled when another waiter completes the shared load.")
        } catch is CancellationError {
            // Expected.
        }

        await manager.shutdownForApplicationTermination()
    }

    func testApplicationTerminationShutdownRejectsNewMLXModelLoads() async {
        let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
        await manager.shutdownForApplicationTermination()

        do {
            _ = try await manager.loadModel()
            XCTFail("MLX model loading should not start after application termination shutdown.")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("Expected CancellationError, got \(error).")
        }
    }

    func testApplicationTerminationShutdownRejectsNewCustomLLMInference() async {
        let manager = CustomLLMModelManager(modelRepo: CustomLLMModelManager.defaultModelRepo)
        await manager.shutdownForApplicationTermination()

        do {
            try await manager.prewarmModel(repo: manager.currentModelRepo)
            XCTFail("Custom LLM inference should not start after application termination shutdown.")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("Expected CancellationError, got \(error).")
        }
    }

    func testModelSwitchPreservesOutstandingActiveUseLease() async {
        await withIsolatedModelStorageRoot { _ in
            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo)
            manager.beginActiveUse()
            manager.updateModel(repo: "mlx-community/parakeet-tdt-0.6b-v3")
            XCTAssertTrue(manager.hasActiveUse)
            manager.endActiveUse()
            XCTAssertFalse(manager.hasActiveUse)
        }
    }

    func testStorageRootChangeRejectsOldModelLoad() async throws {
        try await withIsolatedModelStorageRoot { root in
            let loader = ControlledMLXModelLoader()
            let manager = MLXModelManager(modelRepo: MLXModelManager.defaultModelRepo, modelLoadingOverride: { _ in
                await loader.load()
            })
            let loading = Task { try await manager.loadModel() }
            await loader.waitUntilStarted()
            ModelStorageDirectoryManager.setAuthorizedRootURLForTesting(root.appendingPathComponent("new-root"))
            manager.refreshStorageRoot()
            await loader.finish()
            do {
                _ = try await loading.value
                XCTFail("Old-root model must not become the new-root cache")
            } catch is CancellationError {} catch { XCTFail("Unexpected error: \(error)") }
            XCTAssertFalse(manager.hasLoadedModel)
        }
    }

    private func waitUntilPendingModelLoadClears(_ manager: MLXModelManager) async {
        for _ in 0..<100 where manager.hasPendingModelLoad {
            try? await Task.sleep(for: .milliseconds(5))
        }
    }
}

private final class MLXModelManagerTestModel: STTGenerationModel {
    let defaultGenerationParameters = STTGenerateParameters()

    func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        fatalError("Inference is not used by model lifecycle tests.")
    }

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        fatalError("Inference is not used by model lifecycle tests.")
    }
}

private actor ControlledMLXModelLoader {
    private var didStart = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var loadContinuation: CheckedContinuation<MLXLoadedModelBox, Never>?

    func load() async -> MLXLoadedModelBox {
        didStart = true
        let waiters = startWaiters
        startWaiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
        return await withCheckedContinuation { continuation in
            loadContinuation = continuation
        }
    }

    func waitUntilStarted() async {
        guard !didStart else { return }
        await withCheckedContinuation { continuation in
            startWaiters.append(continuation)
        }
    }

    func finish() {
        loadContinuation?.resume(returning: MLXLoadedModelBox(model: MLXModelManagerTestModel()))
        loadContinuation = nil
    }
}

private actor ControlledSharedValueLoader {
    private var invocationCount = 0
    private var didStart = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var loadContinuation: CheckedContinuation<Int, Never>?

    func load() async -> Int {
        invocationCount += 1
        didStart = true
        let waiters = startWaiters
        startWaiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
        return await withCheckedContinuation { continuation in
            loadContinuation = continuation
        }
    }

    func waitUntilStarted() async {
        guard !didStart else { return }
        await withCheckedContinuation { continuation in
            startWaiters.append(continuation)
        }
    }

    func count() -> Int { invocationCount }

    func finish(with value: Int) {
        loadContinuation?.resume(returning: value)
        loadContinuation = nil
    }
}

private actor AsyncCompletionProbe {
    private var didComplete = false

    func markCompleted() {
        didComplete = true
    }

    func isCompleted() -> Bool {
        didComplete
    }
}
