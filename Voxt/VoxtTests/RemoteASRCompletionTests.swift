import XCTest
@testable import Voxt

@MainActor
final class RemoteASRCompletionTests: XCTestCase {
    func testProviderFailureRecoversPartialWithoutReportingSuccessAsFailure() async {
        let transcriber = RemoteASRTranscriber()
        var failures = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        let text = await transcriber.resolveStreamingResult(
            warningMessage: "test",
            waitForFinal: { throw URLError(.networkConnectionLost) },
            fallback: { "partial result" }
        )
        XCTAssertEqual(text, "partial result")
        XCTAssertEqual(failures, 0)
    }

    func testFailureBeforeAnyTextReportsRuntimeFailure() async {
        let transcriber = RemoteASRTranscriber()
        var failures = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        let text = await transcriber.resolveStreamingResult(
            warningMessage: "test",
            waitForFinal: { throw URLError(.timedOut) },
            fallback: { "" }
        )
        XCTAssertTrue(text.isEmpty)
        XCTAssertEqual(failures, 1)
    }

    func testCancellationDoesNotRecoverOrDeliverPartialText() async {
        let transcriber = RemoteASRTranscriber()
        var failures = 0
        var delivered = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        transcriber.onTranscriptionFinished = { _ in delivered += 1 }
        let generation = transcriber.recordingGenerationID
        let task = Task {
            let text = await transcriber.resolveStreamingResult(
                warningMessage: "test",
                waitForFinal: { try Task.checkCancellation(); return "unexpected" },
                fallback: { "must not be delivered" }
            )
            transcriber.finish(with: text, generationID: generation)
            return text
        }
        task.cancel()
        let result = await task.value
        XCTAssertTrue(result.isEmpty)
        XCTAssertEqual(failures, 0)
        XCTAssertEqual(delivered, 0)
    }

    func testLateFailureCannotReportIntoNewGeneration() {
        let transcriber = RemoteASRTranscriber()
        let oldGeneration = transcriber.recordingGenerationID
        transcriber.discardPendingSessionOutput()
        var failures = 0
        transcriber.onRuntimeFailure = { _ in failures += 1 }
        transcriber.notifyRuntimeFailure(URLError(.timedOut), generationID: oldGeneration)
        XCTAssertEqual(failures, 0)
        transcriber.notifyRuntimeFailure(URLError(.timedOut), generationID: transcriber.recordingGenerationID)
        XCTAssertEqual(failures, 1)
    }

    func testLateCompletionCannotDeliverIntoNewGeneration() {
        // No capture is prepared here. Completing a cold instance must not
        // create an input node or initialize microphone hardware during cleanup.
        let transcriber = RemoteASRTranscriber()
        let oldGeneration = transcriber.recordingGenerationID
        transcriber.discardPendingSessionOutput()
        var delivered: [String] = []
        transcriber.onTranscriptionFinished = { delivered.append($0) }
        transcriber.finish(with: "stale", generationID: oldGeneration)
        XCTAssertTrue(delivered.isEmpty)
        transcriber.finish(with: "current", generationID: transcriber.recordingGenerationID)
        XCTAssertEqual(delivered, ["current"])
    }
}
