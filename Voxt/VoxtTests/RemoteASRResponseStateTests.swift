import XCTest
@testable import Voxt

final class RemoteASRResponseStateTests: XCTestCase {
    func testDoubaoFinalSurvivesLateTextAndSocketError() async throws {
        let state = DoubaoResponseState()
        _ = await state.replace(text: "final answer", isFinal: true)
        _ = await state.replace(text: "late partial", isFinal: false)
        await state.markCompletedWithError(URLError(.networkConnectionLost))
        let result = try await state.waitForFinalResult(timeoutSeconds: 0)
        XCTAssertEqual(result, "final answer")
    }

    func testAliyunFunRetainsLastPartialWhenDrainTimesOut() async throws {
        let state = AliyunFunResponseState()
        _ = await state.updateWithSentence("first", isSentenceEnd: true)
        _ = await state.updateWithSentence("last partial", isSentenceEnd: false)
        await state.markFinishRequested()
        let result = try await state.waitForFinalResult(timeoutSeconds: 0)
        XCTAssertEqual(result, "first last partial")
        _ = await state.updateWithSentence("too late", isSentenceEnd: true)
        await state.markCompletedWithError(URLError(.networkConnectionLost))
        let repeated = try await state.waitForFinalResult(timeoutSeconds: 0)
        XCTAssertEqual(repeated, result)
    }

    func testAliyunQwenFinishedSessionIgnoresLateEvents() async throws {
        let state = AliyunQwenResponseState()
        _ = await state.commit("confirmed")
        await state.markSessionFinished()
        _ = await state.setPartial("late")
        _ = await state.commit("also late")
        await state.markCompletedWithError(URLError(.cancelled))
        let result = try await state.waitForFinalResult(timeoutSeconds: 0)
        XCTAssertEqual(result, "confirmed")
    }

    func testStepFunCommitsFinalItemAfterStopAndRejectsLateDelta() async throws {
        let state = StepFunResponseState()
        _ = await state.appendDelta("partial", itemID: "one")
        await state.markFinishRequested()
        _ = await state.commit("complete", itemID: "one")
        _ = await state.appendDelta("late", itemID: "one")
        let result = try await state.waitForFinalResult(timeoutSeconds: 0)
        XCTAssertEqual(result, "complete")
    }

    func testGeminiTimeoutKeepsInterimWithoutDuplicatingIt() async throws {
        let state = GeminiLiveResponseState()
        _ = await state.setInterim("remaining speech")
        await state.markFinishRequested()
        let first = try await state.waitForFinalResult(timeoutSeconds: 0)
        let second = try await state.waitForFinalResult(timeoutSeconds: 0)
        XCTAssertEqual(first, "remaining speech")
        XCTAssertEqual(second, first)
    }

    func testProviderFailureIsNotConvertedToSuccessfulEmptyResult() async {
        let state = DoubaoResponseState()
        await state.markCompletedWithError(URLError(.timedOut))
        do {
            _ = try await state.waitForFinalResult(timeoutSeconds: 0)
            XCTFail("Expected provider failure")
        } catch {
            XCTAssertEqual((error as NSError).code, NSURLErrorTimedOut)
        }
    }

    func testAllProviderWaitsPropagateCancellation() async {
        let doubao = DoubaoResponseState()
        let fun = AliyunFunResponseState()
        let qwen = AliyunQwenResponseState()
        let step = StepFunResponseState()
        let gemini = GeminiLiveResponseState()
        let waits: [@Sendable () async throws -> String] = [
            { try await doubao.waitForFinalResult(timeoutSeconds: 20) },
            { try await fun.waitForFinalResult(timeoutSeconds: 20) },
            { try await qwen.waitForFinalResult(timeoutSeconds: 20) },
            { try await step.waitForFinalResult(timeoutSeconds: 20) },
            { try await gemini.waitForFinalResult(timeoutSeconds: 20) }
        ]
        for wait in waits {
            let task = Task { try await wait() }
            task.cancel()
            do {
                _ = try await task.value
                XCTFail("Cancellation must not yield a transcript")
            } catch {
                XCTAssertTrue(error is CancellationError)
            }
        }
    }

    func testCancelledWaitFreezesStateBeforeLateSocketError() async {
        let state = DoubaoResponseState(onError: { _ in XCTFail("Cancelled state must ignore late errors") })
        let task = Task { try await state.waitForFinalResult(timeoutSeconds: 20) }
        task.cancel()
        _ = try? await task.value
        await state.markCompletedWithError(URLError(.networkConnectionLost))
        _ = await state.replace(text: "late", isFinal: true)
        let text = await state.currentText()
        XCTAssertTrue(text.isEmpty)
    }

    func testHandshakeGateRemembersEarlySuccessAndFailure() async throws {
        let successful = AsyncGate()
        await successful.open()
        try await successful.wait(timeoutSeconds: 0)
        let failed = AsyncGate()
        await failed.fail(URLError(.cannotConnectToHost))
        await failed.open()
        do {
            try await failed.wait(timeoutSeconds: 0)
            XCTFail("Handshake failure must not release audio upload")
        } catch {
            XCTAssertEqual((error as NSError).code, NSURLErrorCannotConnectToHost)
        }
    }

    func testHandshakeGateTimesOutAndCancelsWithoutPollingForever() async {
        let gate = AsyncGate()
        do {
            try await gate.wait(timeoutSeconds: 0)
            XCTFail("Expected handshake timeout")
        } catch {
            XCTAssertEqual((error as NSError).code, NSURLErrorTimedOut)
        }
        let task = Task { try await gate.wait(timeoutSeconds: 20) }
        task.cancel()
        do {
            try await task.value
            XCTFail("Expected handshake cancellation")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }
}
