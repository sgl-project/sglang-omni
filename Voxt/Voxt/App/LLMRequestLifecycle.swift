import Foundation

/// Owns both request validity and the work still unwinding after invalidation.
@MainActor
final class LLMRequestLifecycle {
    private var currentRequestID = UUID()
    private let tasks = TrackedTaskStore()

    var hasPendingWork: Bool { !tasks.isEmpty }

    func begin() -> UUID {
        tasks.cancelAll()
        currentRequestID = UUID()
        return currentRequestID
    }

    func isCurrent(_ requestID: UUID) -> Bool {
        currentRequestID == requestID
    }

    func run(_ requestID: UUID, operation: @escaping @MainActor () async -> Void) {
        guard isCurrent(requestID) else { return }
        tasks.start { [weak self] in
            guard self?.isCurrent(requestID) == true else { return }
            await operation()
        }
    }

    func cancel() -> [Task<Void, Never>] {
        currentRequestID = UUID()
        return tasks.cancelAll()
    }
}

extension AppDelegate {
    @discardableResult
    func beginLLMRequest() -> UUID {
        llmRequests.begin()
    }

    func isCurrentLLMRequest(_ requestID: UUID) -> Bool {
        llmRequests.isCurrent(requestID) && !isSessionCancellationRequested
    }

    func invalidateActiveLLMRequest() {
        _ = cancelActiveLLMRequest()
    }

    func runTrackedLLMRequest(
        _ requestID: UUID,
        operation: @escaping @MainActor () async -> Void
    ) {
        guard !isApplicationTerminating, isCurrentLLMRequest(requestID) else { return }
        llmRequests.run(requestID, operation: operation)
    }

    @discardableResult
    func cancelActiveLLMRequest() -> [Task<Void, Never>] {
        llmRequests.cancel()
    }
}
