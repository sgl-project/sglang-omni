import Foundation

/// A bounded FIFO of input actions. A pending write holds every queued action
/// until that exact write is acknowledged. This type has no timer.
public struct CompositionWriteQueue<Action: Sendable>: Sendable {
    public private(set) var pendingID: UUID?
    private var actions: [Action] = []

    public init() {}

    public mutating func enqueue(_ action: Action) -> Bool {
        guard actions.count < 128 else { return false }
        actions.append(action)
        return true
    }

    /// For a step that continues the action being performed (mark the next
    /// syllable after its predecessor's commit): it runs before keys already waiting.
    public mutating func enqueueFirst(_ action: Action) -> Bool {
        guard actions.count < 128 else { return false }
        actions.insert(action, at: 0)
        return true
    }

    public mutating func next() -> Action? {
        guard pendingID == nil else { return nil }
        guard !actions.isEmpty else { return nil }
        return actions.removeFirst()
    }

    public mutating func beginWrite(id: UUID) {
        pendingID = id
    }

    public mutating func acknowledge(id: UUID) -> Bool {
        guard pendingID == id else { return false }
        pendingID = nil
        return true
    }

    public mutating func cancel() {
        pendingID = nil
        actions.removeAll()
    }
}
