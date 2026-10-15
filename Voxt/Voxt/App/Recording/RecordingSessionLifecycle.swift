import Foundation

/// Session identity and exactly-once output/end admission move together. UI,
/// audio capture and timing data remain owned by their existing components.
nonisolated struct RecordingSessionLifecycle {
    enum EndDecision: Equatable {
        case execute
        case skipDuplicateInFlight
        case skipAlreadyCompleted
        case skipStale
    }

    private(set) var id = UUID()
    private(set) var outputGeneration = UUID()
    private(set) var isCancelled = false
    private(set) var hasCommittedOutput = false
    private(set) var endingID: UUID?
    private(set) var completedEndID: UUID?
    private var cancelledID: UUID?

    mutating func begin() {
        self = RecordingSessionLifecycle()
    }

    mutating func cancel() {
        cancelledID = id
        id = UUID()
        isCancelled = true
        hasCommittedOutput = true
        invalidateOutputDelivery()
    }

    mutating func invalidateCallbacks() {
        id = UUID()
    }

    mutating func invalidateOutputDelivery() {
        outputGeneration = UUID()
    }

    /// A posted paste may need a follow-up key after normal session teardown.
    /// Begin/cancel/dismiss invalidate it; callback invalidation alone does not.
    func acceptsOutputGeneration(_ generation: UUID) -> Bool {
        outputGeneration == generation
    }

    func accepts(_ sessionID: UUID) -> Bool {
        id == sessionID && !isCancelled
    }

    mutating func claimOutput(for sessionID: UUID) -> Bool {
        guard accepts(sessionID), !hasCommittedOutput else { return false }
        hasCommittedOutput = true
        return true
    }

    mutating func beginEnding(_ sessionID: UUID) -> EndDecision {
        if endingID == sessionID { return .skipDuplicateInFlight }
        if completedEndID == sessionID { return .skipAlreadyCompleted }
        // Cancellation invalidates callbacks immediately, but its old ID must
        // still be allowed through the synchronous resource-cleanup path.
        guard sessionID == id || sessionID == cancelledID else { return .skipStale }
        endingID = sessionID
        return .execute
    }

    mutating func completeEnding(_ sessionID: UUID) {
        guard endingID == sessionID else { return }
        endingID = nil
        completedEndID = sessionID
    }
}
