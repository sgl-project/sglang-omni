import XCTest
@testable import Voxt

@MainActor
final class SharedModelLoadCoordinatorTests: XCTestCase {
    func testCancelledGenerationRemainsAvailableToLaterShutdownBarrier() async {
        let coordinator = SharedModelLoadCoordinator<Int>()
        let gate = ManualTaskBarrier()
        let loading = Task { try await coordinator.value(for: "model") { await gate.wait(); return 42 } }
        await gate.waitUntilEntered()
        _ = coordinator.cancelAll()
        XCTAssertFalse(coordinator.hasPendingLoad)
        XCTAssertTrue(coordinator.hasOutstandingLoad)
        let shutdownLoads = coordinator.cancelAll()
        XCTAssertEqual(shutdownLoads.count, 1)
        gate.release()
        _ = await loading.result
        for load in shutdownLoads { await load.waitForCompletion() }
        XCTAssertFalse(coordinator.hasOutstandingLoad)
    }

    func testLastWaiterCancellationDoesNotLoseTheUnderlyingTask() async {
        let coordinator = SharedModelLoadCoordinator<String>()
        let gate = ManualTaskBarrier()
        let loading = Task { try await coordinator.value(for: "model") { await gate.wait(); return "stale" } }
        await gate.waitUntilEntered()
        loading.cancel()
        for _ in 0..<100 where coordinator.hasPendingLoad { await Task.yield() }
        XCTAssertFalse(coordinator.hasPendingLoad)
        XCTAssertTrue(coordinator.hasOutstandingLoad)
        XCTAssertEqual(coordinator.cancelAll().count, 1)
        gate.release()
        do {
            _ = try await loading.value
            XCTFail("Cancelled waiter returned a model")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(coordinator.hasOutstandingLoad)
    }

    func testOldGenerationCompletionDoesNotEraseReplacement() async throws {
        let coordinator = SharedModelLoadCoordinator<Int>()
        let oldGate = ManualTaskBarrier()
        let nextGate = ManualTaskBarrier()
        let old = Task { try await coordinator.value(for: "same-key") { await oldGate.wait(); return 1 } }
        await oldGate.waitUntilEntered()
        coordinator.cancelAll()
        let next = Task { try await coordinator.value(for: "same-key") { await nextGate.wait(); return 2 } }
        await nextGate.waitUntilEntered()
        oldGate.release()
        _ = await old.result
        XCTAssertTrue(coordinator.hasPendingLoad)
        XCTAssertTrue(coordinator.hasOutstandingLoad)
        nextGate.release()
        let result = try await next.value
        XCTAssertEqual(result, 2)
        XCTAssertFalse(coordinator.hasOutstandingLoad)
    }

    func testInvalidatedFailureCannotBecomeReplacementModelError() async throws {
        let coordinator = SharedModelLoadCoordinator<Int>()
        let oldGate = ManualTaskBarrier()
        let nextGate = ManualTaskBarrier()
        let old = Task {
            try await coordinator.value(for: "same-key") {
                await oldGate.wait()
                throw URLError(.timedOut)
            }
        }
        await oldGate.waitUntilEntered()
        coordinator.cancelAll()
        let next = Task { try await coordinator.value(for: "same-key") { await nextGate.wait(); return 2 } }
        await nextGate.waitUntilEntered()
        oldGate.release()
        do { _ = try await old.value; XCTFail("Expected cancelled generation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertTrue(coordinator.hasPendingLoad)
        nextGate.release()
        let result = try await next.value
        XCTAssertEqual(result, 2)
    }

    func testFailedLoadCanBeRetriedWithTypedValue() async throws {
        let coordinator = SharedModelLoadCoordinator<[String]>()
        do {
            _ = try await coordinator.value(for: "model") { throw URLError(.cannotOpenFile) }
            XCTFail("Expected load failure")
        } catch { XCTAssertEqual((error as NSError).code, NSURLErrorCannotOpenFile) }
        XCTAssertFalse(coordinator.hasOutstandingLoad)
        let result = try await coordinator.value(for: "model") { ["loaded"] }
        XCTAssertEqual(result, ["loaded"])
    }

    func testCancelledCallerDoesNotStartNewLoader() async {
        let coordinator = SharedModelLoadCoordinator<Int>()
        let loading = Task {
            try await coordinator.value(for: "model") {
                XCTFail("Cancelled caller started loading")
                return 0
            }
        }
        loading.cancel()
        _ = await loading.result
        XCTAssertFalse(coordinator.hasPendingLoad)
        XCTAssertFalse(coordinator.hasOutstandingLoad)
    }
}
