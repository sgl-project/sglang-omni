import XCTest
@testable import Voxt

@MainActor
final class MLXCorrectionPassCoordinatorTests: XCTestCase {
    func testIntermediatePassIsSkippedWhileAnotherPassRuns() async {
        let coordinator = MLXCorrectionPassCoordinator()
        let barrier = ManualTaskBarrier()
        let first = Task {
            await coordinator.run(kind: .intermediate, isCurrent: { true }) {
                await barrier.wait()
                return .success("first")
            }
        }
        await barrier.waitUntilEntered()
        let skipped = await coordinator.run(kind: .intermediate, isCurrent: { true }) {
            XCTFail("Overlapping intermediate inference")
            return .success("unexpected")
        }
        XCTAssertNil(skipped.text)
        barrier.release()
        _ = await first.value
        XCTAssertFalse(coordinator.hasPendingWork)
    }

    func testFinalWaitsForCancelledIntermediateToReleaseResources() async {
        let coordinator = MLXCorrectionPassCoordinator()
        let barrier = ManualTaskBarrier()
        var events: [String] = []
        let first = Task {
            await coordinator.run(kind: .intermediate, isCurrent: { true }) {
                events.append("acquire")
                await barrier.wait()
                events.append("release")
                return .success("partial")
            }
        }
        await barrier.waitUntilEntered()
        let final = Task {
            await coordinator.run(kind: .postStopFinal, isCurrent: { true }) {
                events.append("final")
                return .success("final result")
            }
        }
        // Cancellation does not discard the slot while its native work is unwinding.
        coordinator.cancel()
        XCTAssertTrue(coordinator.hasPendingWork)
        barrier.release()
        _ = await first.value
        let result = await final.value
        XCTAssertEqual(events, ["acquire", "release", "final"])
        XCTAssertEqual(result.text, "final result")
    }

    func testCancelledCallerCancelsInnerPassButRemainsOwnedUntilExit() async {
        let coordinator = MLXCorrectionPassCoordinator()
        let barrier = ManualTaskBarrier()
        var sawCancellation = false
        let caller = Task {
            await coordinator.run(kind: .postStopFinal, isCurrent: { true }) {
                await barrier.wait()
                sawCancellation = Task.isCancelled
                return .success("cancelled output")
            }
        }
        await barrier.waitUntilEntered()
        caller.cancel()
        XCTAssertTrue(coordinator.hasPendingWork)
        barrier.release()
        let result = await caller.value
        XCTAssertTrue(sawCancellation)
        XCTAssertNil(result.text)
        XCTAssertFalse(coordinator.hasPendingWork)
    }

    func testStaleRevisionCannotPublishOrStartAnotherPass() async {
        let coordinator = MLXCorrectionPassCoordinator()
        let barrier = ManualTaskBarrier()
        var current = true
        let caller = Task {
            await coordinator.run(kind: .postStopFinal, isCurrent: { current }) {
                await barrier.wait()
                return .success("stale")
            }
        }
        await barrier.waitUntilEntered()
        current = false
        barrier.release()
        let result = await caller.value
        XCTAssertNil(result.text)
        _ = await coordinator.run(kind: .postStopFinal, isCurrent: { current }) {
            XCTFail("Stale revision acquired model")
            return .success(nil)
        }
    }

    func testCancelledWaiterDoesNotStartAfterActivePassFinishes() async {
        let coordinator = MLXCorrectionPassCoordinator()
        let barrier = ManualTaskBarrier()
        let first = Task {
            await coordinator.run(kind: .postStopQuick, isCurrent: { true }) {
                await barrier.wait()
                return .success("quick")
            }
        }
        await barrier.waitUntilEntered()
        let waiter = Task {
            await coordinator.run(kind: .postStopFinal, isCurrent: { true }) {
                XCTFail("Cancelled waiter started inference")
                return .success(nil)
            }
        }
        waiter.cancel()
        barrier.release()
        _ = await first.value
        let result = await waiter.value
        XCTAssertNil(result.text)
    }
}
