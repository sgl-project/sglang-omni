import XCTest
@testable import Voxt

@MainActor
final class TrackedTaskStoreTests: XCTestCase {
    func testCancellationRetainsWorkUntilItsCleanupCompletes() async {
        let store = TrackedTaskStore()
        let barrier = ManualTaskBarrier()
        var cleanedUp = false
        let task = store.start {
            await barrier.wait()
            cleanedUp = true
        }
        await barrier.waitUntilEntered()
        let cancelled = store.cancelAll()
        XCTAssertEqual(cancelled.count, 1)
        XCTAssertFalse(store.isEmpty)
        XCTAssertTrue(task.isCancelled)
        XCTAssertFalse(cleanedUp)
        barrier.release()
        await store.waitForAll()
        XCTAssertTrue(cleanedUp)
        XCTAssertTrue(store.isEmpty)
    }

    func testCaptureReplacementWaitsForRetiringStartBeforeTouchingEngine() async {
        let store = TrackedTaskStore()
        let barrier = ManualTaskBarrier()
        var events: [String] = []
        store.start {
            events.append("old-start")
            await barrier.wait()
            events.append("old-cleanup")
        }
        await barrier.waitUntilEntered()
        store.start(after: store.cancelAll()) { events.append("new-start") }
        XCTAssertEqual(store.count, 2)
        barrier.release()
        await store.waitForAll()
        XCTAssertEqual(events, ["old-start", "old-cleanup", "new-start"])
    }

    func testCancellingQueuedReplacementSkipsItsOperationButPreservesBarrier() async {
        let store = TrackedTaskStore()
        let barrier = ManualTaskBarrier()
        store.start { await barrier.wait() }
        await barrier.waitUntilEntered()
        let replacement = store.start(after: store.cancelAll()) { XCTFail("Queued capture was cancelled") }
        replacement.cancel()
        XCTAssertEqual(store.count, 2)
        barrier.release()
        await store.waitForAll()
        XCTAssertTrue(store.isEmpty)
    }

    func testImmediateCompletionCannotLeaveRegistrationBehind() async {
        let store = TrackedTaskStore()
        let task = store.start {}
        await task.value
        XCTAssertTrue(store.isEmpty)
    }

    func testCancelBeforeTaskStartsDoesNotRunOperation() async {
        let store = TrackedTaskStore()
        let task = store.start { XCTFail("Cancelled before start") }
        store.cancelAll()
        await task.value
        XCTAssertTrue(store.isEmpty)
    }
}
