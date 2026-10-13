import XCTest
@testable import Voxt

@MainActor
final class MeetingImportedFileAnalyzerTests: XCTestCase {
    private let url = URL(fileURLWithPath: "/unused-test-input.wav")

    func testCancelWhileWaitingForPreviousCleanupNeverStartsPipeline() async throws {
        let analyzer = MeetingImportedFileAnalyzer()
        let cleanupGate = ManualTaskBarrier()
        let cleanup = Task { await cleanupGate.wait() }
        await cleanupGate.waitUntilEntered()
        let pipeline = ControlledImportPipeline()
        let task = try analyzer.start(at: url, after: cleanup, using: pipeline, progress: { _ in })
        XCTAssertTrue(analyzer.isRunning)
        let cancellation = analyzer.cancel()
        XCTAssertNotNil(cancellation)
        XCTAssertTrue(analyzer.isRunning)
        cleanupGate.release()
        await cancellation?.value
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(pipeline.analysisCount, 0)
        XCTAssertEqual(pipeline.finishValues, [false])
        XCTAssertFalse(analyzer.isRunning)
    }

    func testAbandonedOwnerCancelsRegisteredImportBeforeCleanupFinishes() async throws {
        var analyzer: MeetingImportedFileAnalyzer? = MeetingImportedFileAnalyzer()
        weak var weakAnalyzer = analyzer
        let gate = ManualTaskBarrier()
        let cleanup = Task { await gate.wait() }
        await gate.waitUntilEntered()
        let pipeline = ControlledImportPipeline()
        let task = try analyzer!.start(at: url, after: cleanup, using: pipeline, progress: { _ in })
        analyzer = nil
        XCTAssertNil(weakAnalyzer)
        gate.release()
        do { _ = try await task.value; XCTFail("Abandoned import started") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(pipeline.analysisCount, 0)
        XCTAssertEqual(pipeline.finishValues, [false])
    }

    func testConcurrentImportIsRejectedUntilCleanupCompletes() async throws {
        let analyzer = MeetingImportedFileAnalyzer()
        let finishGate = ManualTaskBarrier()
        let pipeline = ControlledImportPipeline(finishGate: finishGate)
        let task = try analyzer.start(at: url, after: nil, using: pipeline, progress: { _ in })
        await finishGate.waitUntilEntered()
        XCTAssertTrue(analyzer.isRunning)
        XCTAssertThrowsError(try analyzer.start(at: url, after: nil, using: ControlledImportPipeline(), progress: { _ in }))
        finishGate.release()
        _ = try await task.value
        XCTAssertFalse(analyzer.isRunning)
        let next = try analyzer.start(at: url, after: nil, using: ControlledImportPipeline(), progress: { _ in })
        _ = try await next.value
    }

    func testCancellationDuringSuccessfulCleanupDiscardsPreparedResult() async throws {
        let analyzer = MeetingImportedFileAnalyzer()
        let finishGate = ManualTaskBarrier()
        let pipeline = ControlledImportPipeline(finishGate: finishGate)
        let task = try analyzer.start(at: url, after: nil, using: pipeline, progress: { _ in })
        await finishGate.waitUntilEntered()
        let cancellation = analyzer.cancel()
        finishGate.release()
        await cancellation?.value
        do { _ = try await task.value; XCTFail("Cancelled result escaped") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(pipeline.finishValues, [true, false])
        XCTAssertFalse(analyzer.isRunning)
    }

    func testFailureAlwaysCleansUpAndAllowsNextImport() async throws {
        let analyzer = MeetingImportedFileAnalyzer()
        let pipeline = ControlledImportPipeline()
        pipeline.shouldFail = true
        let task = try analyzer.start(at: url, after: nil, using: pipeline, progress: { _ in })
        do { _ = try await task.value; XCTFail("Expected pipeline error") }
        catch { XCTAssertEqual((error as NSError).code, NSURLErrorCannotOpenFile) }
        XCTAssertEqual(pipeline.finishValues, [false])
        XCTAssertFalse(analyzer.isRunning)
        let next = try analyzer.start(at: url, after: nil, using: ControlledImportPipeline(), progress: { _ in })
        _ = try await next.value
    }

    func testParentCancellationPropagatesToOwnedAnalysis() async {
        let analyzer = MeetingImportedFileAnalyzer()
        let analysisGate = ManualTaskBarrier()
        let pipeline = ControlledImportPipeline(analysisGate: analysisGate)
        let caller = Task { try await analyzer.analyze(at: url, after: nil, using: pipeline, progress: { _ in }) }
        await analysisGate.waitUntilEntered()
        caller.cancel()
        analysisGate.release()
        do { _ = try await caller.value; XCTFail("Expected parent cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(pipeline.finishValues, [false])
    }

    func testDelayedCancelOnlyTouchesCapturedPipeline() async throws {
        let analyzer = MeetingImportedFileAnalyzer()
        let cancelGate = ManualTaskBarrier()
        let analysisGate = ManualTaskBarrier()
        let old = ControlledImportPipeline(analysisGate: analysisGate)
        old.cancelGate = cancelGate
        let oldTask = try analyzer.start(at: url, after: nil, using: old, progress: { _ in })
        await analysisGate.waitUntilEntered()
        let cancelled = analyzer.cancel()
        await cancelGate.waitUntilEntered()
        analysisGate.release()
        _ = await oldTask.result
        let nextGate = ManualTaskBarrier()
        let next = ControlledImportPipeline(analysisGate: nextGate)
        let nextTask = try analyzer.start(at: url, after: nil, using: next, progress: { _ in })
        await nextGate.waitUntilEntered()
        cancelGate.release()
        await cancelled?.value
        XCTAssertEqual(next.cancelCount, 0)
        XCTAssertTrue(analyzer.isRunning)
        nextGate.release()
        _ = try await nextTask.value
    }
}

@MainActor
private final class ControlledImportPipeline: MeetingImportedFileAnalyzing {
    let analysisGate: ManualTaskBarrier?
    let finishGate: ManualTaskBarrier?
    var cancelGate: ManualTaskBarrier?
    var shouldFail = false
    var analysisCount = 0
    var cancelCount = 0
    var finishValues: [Bool] = []

    init(analysisGate: ManualTaskBarrier? = nil, finishGate: ManualTaskBarrier? = nil) {
        self.analysisGate = analysisGate
        self.finishGate = finishGate
    }

    func analyze(at url: URL, progress: @escaping @MainActor @Sendable (MeetingFileAnalysisProgress) -> Void) async throws -> MeetingSessionResult {
        analysisCount += 1
        await analysisGate?.wait()
        if shouldFail { throw URLError(.cannotOpenFile) }
        return MeetingSessionResult(
            transcriptionEngine: .remote, transcriptionModelDescription: "test",
            segments: [], visibleSnapshotSegments: [], audioDurationSeconds: 1, archivedAudioURL: nil
        )
    }

    func cancel() async { cancelCount += 1; await cancelGate?.wait() }
    func finish(keepingResult: Bool) async {
        finishValues.append(keepingResult)
        await finishGate?.wait()
    }
}
