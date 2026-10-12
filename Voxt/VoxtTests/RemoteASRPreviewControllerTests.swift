import XCTest
@testable import Voxt

@MainActor
final class RemoteASRPreviewControllerTests: XCTestCase {
    func testLateCancelledPreviewCannotPublishIntoReplacement() async {
        let controller = RemoteASRPreviewController()
        let gate = ManualTaskBarrier()
        var published: [String] = []
        let first = controller.start(
            shouldRun: { true }, transcribe: { await gate.wait(); return "old" },
            publish: { published.append($0) }, waitForNext: {}
        )
        await gate.waitUntilEntered()
        let second = controller.start(
            shouldRun: { true }, transcribe: { "new" },
            publish: { published.append($0); controller.cancel() }, waitForNext: {}
        )
        await second.value
        gate.release()
        await first.value
        XCTAssertEqual(published, ["new"])
    }

    func testRecordingStoppedDuringRequestSuppressesReply() async {
        let controller = RemoteASRPreviewController()
        let gate = ManualTaskBarrier()
        var recording = true
        let task = controller.start(
            shouldRun: { recording }, transcribe: { await gate.wait(); return "late" },
            publish: { _ in XCTFail("Stopped recording must not publish") }, waitForNext: {}
        )
        await gate.waitUntilEntered()
        recording = false
        gate.release()
        await task.value
    }

    func testDuplicateVisibleTextIsPublishedOncePerInvocation() async {
        let controller = RemoteASRPreviewController()
        var values = ["same", "same", "changed"]
        var published: [String] = []
        let task = controller.start(
            shouldRun: { true }, transcribe: { values.removeFirst() },
            publish: { published.append($0) },
            waitForNext: { if values.isEmpty { throw CancellationError() } }
        )
        await task.value
        XCTAssertEqual(published, ["same", "changed"])
    }

    func testWaitForIdleIncludesCancelledInFlightRequest() async {
        let controller = RemoteASRPreviewController()
        let gate = ManualTaskBarrier()
        controller.start(
            shouldRun: { true }, transcribe: { await gate.wait(); return nil },
            publish: { _ in XCTFail("Cancelled work must not publish") }, waitForNext: {}
        )
        await gate.waitUntilEntered()
        controller.cancel()
        let entered = ManualTaskBarrier()
        var idle = false
        let waiter = Task { entered.release(); await controller.waitForIdle(); idle = true }
        await entered.wait()
        XCTAssertFalse(idle)
        gate.release()
        await waiter.value
        XCTAssertTrue(idle)
    }
}
