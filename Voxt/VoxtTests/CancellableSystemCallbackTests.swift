// CancellableSystemCallbackTests.swift
// Provides cancellation coverage for system callbacks that may outlive application tasks.

import XCTest
@testable import Voxt

@MainActor
final class CancellableSystemCallbackTests: XCTestCase {
    func testCancellationReturnsBeforeSystemCallbackAndIgnoresLateResult() async {
        let callbackBox = CallbackBox<Int>()
        let callbackInstalled = expectation(description: "callback installed")
        let waitFinished = expectation(description: "wait finished")
        let cancellationResultBox = ValueBox<Bool>()

        let waitTask = Task {
            let result = await CancellableSystemCallback.wait { callback in
                callbackBox.store(callback)
                callbackInstalled.fulfill()
            }
            cancellationResultBox.store(result == nil)
            waitFinished.fulfill()
        }

        await fulfillment(of: [callbackInstalled], timeout: 1)
        waitTask.cancel()
        await fulfillment(of: [waitFinished], timeout: 1)

        XCTAssertEqual(cancellationResultBox.value, true)
        callbackBox.invoke(42)
        _ = await waitTask.value
        XCTAssertEqual(cancellationResultBox.value, true)
    }

    func testCallbackResultIsDeliveredWhenTaskRemainsActive() async {
        let result = await CancellableSystemCallback.wait { callback in
            callback("authorized")
        }

        XCTAssertEqual(result, "authorized")
    }
}

private final class CallbackBox<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var callback: (@Sendable (Value) -> Void)?

    func store(_ callback: @escaping @Sendable (Value) -> Void) {
        lock.lock()
        self.callback = callback
        lock.unlock()
    }

    func invoke(_ value: Value) {
        lock.lock()
        let callback = callback
        self.callback = nil
        lock.unlock()
        callback?(value)
    }
}

private final class ValueBox<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var storedValue: Value?

    var value: Value? {
        lock.lock()
        defer { lock.unlock() }
        return storedValue
    }

    func store(_ value: Value) {
        lock.lock()
        storedValue = value
        lock.unlock()
    }
}
