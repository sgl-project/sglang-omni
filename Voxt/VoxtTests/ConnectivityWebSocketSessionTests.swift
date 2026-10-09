import Foundation
import XCTest
@testable import Voxt

@MainActor
final class ConnectivityWebSocketSessionTests: XCTestCase {
    func testMessageCancelsDeadlineWithoutClosingHealthyConnection() async throws {
        let receiver = Receiver()
        let connection = makeConnection(receiver)
        let task = Task { try await connection.receive(timeoutSeconds: 60) }
        await receiver.entered.wait()
        receiver.reply(.success(.string("ready")))
        guard case .string(let message) = try await task.value else { return XCTFail("Expected text") }
        XCTAssertEqual(message, "ready")
        XCTAssertEqual(receiver.closeCount, 0)
        connection.close()
        await receiver.closed.wait()
        XCTAssertEqual(receiver.closeCount, 1)
    }

    func testDeadlineClosesBlockedReceiveBeforeWaitingForGroupExit() async {
        let receiver = Receiver()
        let deadline = ManualTaskBarrier()
        let connection = makeConnection(receiver)
        let task = Task {
            try await connection.receive(timeoutSeconds: 3, wait: { _ in await deadline.wait() })
        }
        await receiver.entered.wait()
        await deadline.waitUntilEntered()
        deadline.release()
        do { _ = try await task.value; XCTFail("Expected deadline") }
        catch { XCTAssertEqual((error as NSError).code, -121) }
        XCTAssertEqual(receiver.closeCount, 1)
    }

    func testParentCancellationClosesAndDrainsReceive() async {
        let receiver = Receiver()
        let connection = makeConnection(receiver)
        let task = Task { try await connection.receive(timeoutSeconds: 60) }
        await receiver.entered.wait()
        task.cancel()
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(receiver.closeCount, 1)
    }

    func testParentCancellationAlsoUnblocksSend() async {
        let receiver = Receiver()
        let connection = ConnectivityWebSocketSession(
            send: { _ in _ = try await receiver.receive() },
            receive: { .string("unused") },
            close: { Task { @MainActor in receiver.close() } }
        )
        let task = Task { try await connection.send(.string("request")) }
        await receiver.entered.wait()
        task.cancel()
        do { try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(receiver.closeCount, 1)
    }

    func testAlreadyCancelledRequestDoesNotStartTransportWork() async {
        let receiver = Receiver()
        let admission = ManualTaskBarrier()
        let connection = makeConnection(receiver)
        let task = Task {
            await admission.wait()
            return try await connection.receive(timeoutSeconds: 60)
        }
        await admission.waitUntilEntered()
        task.cancel()
        admission.release()
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        await receiver.closed.wait()
        XCTAssertEqual(receiver.receiveCount, 0)
    }

    func testCloseIsIdempotentAndRejectsFurtherSends() async {
        let receiver = Receiver()
        let connection = makeConnection(receiver)
        connection.close()
        connection.close()
        do { try await connection.send(.string("late")); XCTFail("Closed socket must reject send") }
        catch { XCTAssertTrue(error is CancellationError) }
        await receiver.closed.wait()
        XCTAssertEqual(receiver.closeCount, 1)
    }

    func testReceiveFailureReleasesTransport() async {
        let receiver = Receiver()
        let connection = makeConnection(receiver)
        let task = Task { try await connection.receive(timeoutSeconds: 60) }
        await receiver.entered.wait()
        receiver.reply(.failure(URLError(.badServerResponse)))
        do { _ = try await task.value; XCTFail("Expected receive failure") }
        catch { XCTAssertEqual((error as NSError).code, NSURLErrorBadServerResponse) }
        await receiver.closed.wait()
        XCTAssertEqual(receiver.closeCount, 1)
    }

    private func makeConnection(_ receiver: Receiver) -> ConnectivityWebSocketSession {
        ConnectivityWebSocketSession(
            send: { _ in }, receive: { try await receiver.receive() },
            close: { Task { @MainActor in receiver.close() } }
        )
    }
}

@MainActor
private final class Receiver {
    let entered = ManualTaskBarrier()
    let closed = ManualTaskBarrier()
    private var continuation: CheckedContinuation<URLSessionWebSocketTask.Message, Error>?
    private var isClosed = false
    private(set) var closeCount = 0
    private(set) var receiveCount = 0

    func receive() async throws -> URLSessionWebSocketTask.Message {
        receiveCount += 1
        entered.release()
        guard !isClosed else { throw CancellationError() }
        return try await withCheckedThrowingContinuation { continuation = $0 }
    }

    func reply(_ result: Result<URLSessionWebSocketTask.Message, Error>) {
        let pending = continuation
        continuation = nil
        pending?.resume(with: result)
    }

    func close() {
        closeCount += 1
        isClosed = true
        reply(.failure(CancellationError()))
        closed.release()
    }
}
