import XCTest
@testable import Voxt

@MainActor
final class RemoteProviderSheetOperationsTests: XCTestCase {
    func testReplacedModelLoadCannotOverwriteNewerOptions() async {
        let owner = RemoteProviderSheetOperations()
        let old = ManualTaskBarrier()
        let first = owner.loadModels { await old.wait(); return [.init(id: "old", title: "Old")] }
        await old.waitUntilEntered()
        let second = owner.loadModels { [.init(id: "new", title: "New")] }
        await second?.value
        old.release()
        await first?.value
        XCTAssertEqual(owner.modelOptions?.map(\.id), ["new"])
    }

    func testCloseRejectsLateSuccessAndBlocksNewWorkUntilReopened() async {
        let owner = RemoteProviderSheetOperations()
        let gate = ManualTaskBarrier()
        let request = owner.testConnection { await gate.wait(); return "old success" }
        await gate.waitUntilEntered()
        owner.cancel()
        XCTAssertFalse(owner.isTestingConnection)
        XCTAssertNil(owner.testConnection { XCTFail("Closed sheet must not send requests"); return "" })
        gate.release()
        await request?.value
        XCTAssertNil(owner.connectionResult)
        owner.activate()
        await owner.testConnection { "fresh" }?.value
        XCTAssertEqual(owner.connectionResult?.message, "fresh")
    }

    func testReplacedFailureCannotStopOrOverwriteCurrentRequest() async {
        let owner = RemoteProviderSheetOperations()
        let firstGate = ManualTaskBarrier()
        let secondGate = ManualTaskBarrier()
        let first = owner.testConnection { await firstGate.wait(); throw URLError(.timedOut) }
        await firstGate.waitUntilEntered()
        let second = owner.testConnection { await secondGate.wait(); return "new" }
        await secondGate.waitUntilEntered()
        firstGate.release()
        await first?.value
        XCTAssertTrue(owner.isTestingConnection)
        XCTAssertNil(owner.connectionResult)
        secondGate.release()
        await second?.value
        XCTAssertEqual(owner.connectionResult?.message, "new")
        XCTAssertEqual(owner.connectionResult?.succeeded, true)
    }

    func testValidationFailureInvalidatesPendingNetworkResult() async {
        let owner = RemoteProviderSheetOperations()
        let gate = ManualTaskBarrier()
        let task = owner.testConnection { await gate.wait(); return "success" }
        await gate.waitUntilEntered()
        owner.showFailure("invalid configuration")
        gate.release()
        await task?.value
        XCTAssertEqual(owner.connectionResult?.message, "invalid configuration")
        XCTAssertEqual(owner.connectionResult?.succeeded, false)
        XCTAssertFalse(owner.isTestingConnection)
    }

    func testCloseDrainsCancelledModelLoadBeforeIdle() async {
        let owner = RemoteProviderSheetOperations()
        let gate = ManualTaskBarrier()
        owner.loadModels { await gate.wait(); return [] }
        await gate.waitUntilEntered()
        owner.cancel()
        var finished = false
        let entered = ManualTaskBarrier()
        let wait = Task { entered.release(); await owner.waitForIdle(); finished = true }
        await entered.wait()
        XCTAssertFalse(finished)
        gate.release()
        await wait.value
        XCTAssertTrue(finished)
        XCTAssertNil(owner.modelOptions)
    }

    func testClosingClearsPublishedResultsBeforeDeferredViewCallbacks() async {
        let owner = RemoteProviderSheetOperations()
        await owner.loadModels { [.init(id: "model", title: "Model")] }?.value
        await owner.testConnection { "done" }?.value
        XCTAssertNotNil(owner.modelOptions)
        XCTAssertNotNil(owner.connectionResult)
        owner.cancel()
        XCTAssertNil(owner.modelOptions)
        XCTAssertNil(owner.connectionResult)
        XCTAssertFalse(owner.isTestingConnection)
    }

    func testCurrentFailurePublishesOnceAndClearsBusyState() async {
        let owner = RemoteProviderSheetOperations()
        await owner.testConnection { throw NSError(domain: "fixture", code: 1, userInfo: [NSLocalizedDescriptionKey: "failed"]) }?.value
        XCTAssertFalse(owner.isTestingConnection)
        XCTAssertEqual(owner.connectionResult?.succeeded, false)
        XCTAssertEqual(owner.connectionResult?.message, "failed")
    }
}
