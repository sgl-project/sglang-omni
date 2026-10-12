import XCTest
@testable import Voxt

@MainActor
final class TextInjectionTransactionTests: XCTestCase {
    func testValidAttemptPostsAndCompletesOnce() {
        var posts = 0
        var results: [Bool] = []
        let transaction = TextInjectionTransaction(
            isValid: { true },
            inject: { done in posts += 1; done(true) },
            completion: { results.append($0) }
        )
        transaction.perform()
        transaction.perform()
        XCTAssertEqual(posts, 1)
        XCTAssertEqual(results, [true])
    }

    func testAttemptWithoutCompletionStillInjects() {
        var posts = 0
        let transaction = TextInjectionTransaction(
            isValid: { true },
            inject: { done in posts += 1; done(true) }
        )
        transaction.perform()
        XCTAssertEqual(posts, 1)
    }

    func testCancellationBeforeQueuedAttemptSkipsPaste() {
        var lifecycle = RecordingSessionLifecycle()
        let sessionID = lifecycle.id
        var results: [Bool] = []
        let transaction = TextInjectionTransaction(
            isValid: { lifecycle.accepts(sessionID) },
            inject: { _ in XCTFail("Cancelled work must not touch pasteboard or keys") },
            completion: { results.append($0) }
        )
        lifecycle.cancel()
        transaction.perform()
        transaction.perform()
        XCTAssertEqual(results, [false])
    }

    func testNewSessionBeforeQueuedAttemptSkipsOldPaste() {
        var lifecycle = RecordingSessionLifecycle()
        let sessionID = lifecycle.id
        var results: [Bool] = []
        let transaction = TextInjectionTransaction(
            isValid: { lifecycle.accepts(sessionID) },
            inject: { _ in XCTFail("Old paste must not enter the new session") },
            completion: { results.append($0) }
        )
        lifecycle.begin()
        transaction.perform()
        XCTAssertEqual(results, [false])
        XCTAssertFalse(lifecycle.hasCommittedOutput)
    }

    func testFailedInjectionIsReportedWithoutRetry() {
        var posts = 0
        var results: [Bool] = []
        let transaction = TextInjectionTransaction(
            isValid: { true },
            inject: { done in posts += 1; done(false) },
            completion: { results.append($0) }
        )
        transaction.perform()
        transaction.perform()
        XCTAssertEqual(posts, 1)
        XCTAssertEqual(results, [false])
    }

    func testDuplicateNativeCompletionKeepsFirstOutcome() {
        var results: [Bool] = []
        let transaction = TextInjectionTransaction(
            isValid: { true },
            inject: { done in done(false); done(true) },
            completion: { results.append($0) }
        )
        transaction.perform()
        XCTAssertEqual(results, [false])
    }

    func testCompletionReentryCannotPostAgain() {
        var posts = 0
        var completions = 0
        var transaction: TextInjectionTransaction?
        transaction = TextInjectionTransaction(
            isValid: { true },
            inject: { done in posts += 1; done(true) },
            completion: { _ in
                completions += 1
                transaction?.perform()
            }
        )
        transaction?.perform()
        transaction = nil
        XCTAssertEqual(posts, 1)
        XCTAssertEqual(completions, 1)
    }

    func testLateCallbackReportsOutcomeWithoutPretendingToUndoPostedKeys() {
        var valid = true
        var reply: TextInjectionTransaction.Completion?
        var results: [Bool] = []
        let transaction = TextInjectionTransaction(
            isValid: { valid },
            inject: { reply = $0 },
            completion: { results.append($0) }
        )
        transaction.perform()
        XCTAssertTrue(results.isEmpty)
        valid = false
        reply?(true)
        reply?(false)
        reply = nil
        XCTAssertEqual(results, [true])
    }
}
