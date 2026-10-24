import XCTest
@testable import Voxt

@MainActor
final class LLMRequestLifecycleTests: XCTestCase {
    func testRequestInvalidationAndOutstandingWorkHaveDifferentLifetimes() async {
        let lifecycle = LLMRequestLifecycle()
        let barrier = ManualTaskBarrier()
        let first = lifecycle.begin()
        lifecycle.run(first) { await barrier.wait() }
        await barrier.waitUntilEntered()
        let second = lifecycle.begin()
        XCTAssertFalse(lifecycle.isCurrent(first))
        XCTAssertTrue(lifecycle.isCurrent(second))
        XCTAssertTrue(lifecycle.hasPendingWork)
        let cancelled = lifecycle.cancel()
        XCTAssertFalse(lifecycle.isCurrent(second))
        barrier.release()
        for task in cancelled { await task.value }
        XCTAssertFalse(lifecycle.hasPendingWork)
    }

    func testMultipleOperationsForSameRequestRemainTrackedIndependently() async {
        let lifecycle = LLMRequestLifecycle()
        let first = ManualTaskBarrier()
        let second = ManualTaskBarrier()
        let request = lifecycle.begin()
        lifecycle.run(request) { await first.wait() }
        lifecycle.run(request) { await second.wait() }
        await first.waitUntilEntered()
        await second.waitUntilEntered()
        let tasks = lifecycle.cancel()
        XCTAssertEqual(tasks.count, 2)
        first.release()
        second.release()
        for task in tasks { await task.value }
        XCTAssertFalse(lifecycle.hasPendingWork)
    }

    func testStaleRequestCannotEnqueueWork() {
        let lifecycle = LLMRequestLifecycle()
        let old = lifecycle.begin()
        _ = lifecycle.begin()
        lifecycle.run(old) { XCTFail("Stale request executed") }
        XCTAssertFalse(lifecycle.hasPendingWork)
    }

    func testInvalidationBeforeExecutionSkipsQueuedRequest() async {
        let lifecycle = LLMRequestLifecycle()
        let request = lifecycle.begin()
        lifecycle.run(request) { XCTFail("Invalidated request executed") }
        for task in lifecycle.cancel() { await task.value }
        XCTAssertFalse(lifecycle.hasPendingWork)
    }
}
