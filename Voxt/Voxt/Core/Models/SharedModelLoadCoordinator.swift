import Foundation

/// Type-erases only the completion barrier, never the loaded model value.
nonisolated struct SharedModelLoadTask: Sendable {
    private let wait: @Sendable () async -> Void

    fileprivate init<Value: Sendable>(_ task: Task<Value, Error>) {
        wait = { _ = await task.result }
    }

    func waitForCompletion() async { await wait() }
}

// Value storage is independent of actor context; all owner access stays on MainActor.
nonisolated private struct SharedModelLoadEntry<Value: Sendable>: Sendable {
    let generation: UUID
    let task: Task<Value, Error>
    var waiterIDs: Set<UUID>
}

/// Coalesces current waiters while retaining invalidated loads until they exit.
/// A cancelled waiter is not evidence that native loading released its resources.
// Swift 6.3.2 crashes optimizing the synthesized deinit of this generic global-
// actor class. Isolate the members instead of the type: the compiler still
// enforces MainActor access, with no unsafe storage or Release flag changes.
nonisolated final class SharedModelLoadCoordinator<Value: Sendable> {
    private typealias Entry = SharedModelLoadEntry<Value>

    @MainActor private var entries: [String: Entry] = [:]
    @MainActor private var outstanding: [UUID: Task<Value, Error>] = [:]

    @MainActor init() {}

    @MainActor var hasPendingLoad: Bool { !entries.isEmpty }
    @MainActor var hasOutstandingLoad: Bool { !outstanding.isEmpty }

    @MainActor func value(
        for key: String,
        start: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        try Task.checkCancellation()
        let waiterID = UUID()
        let entry: Entry
        if var existing = entries[key] {
            existing.waiterIDs.insert(waiterID)
            entries[key] = existing
            entry = existing
        } else {
            let generation = UUID()
            let task = Task { try await start() }
            entry = Entry(generation: generation, task: task, waiterIDs: [waiterID])
            entries[key] = entry
            outstanding[generation] = task
        }

        return try await withTaskCancellationHandler {
            defer { finishWaiter(waiterID, key: key, generation: entry.generation) }
            let value: Value
            do {
                value = try await entry.task.value
            } catch {
                guard !Task.isCancelled, entries[key]?.generation == entry.generation else {
                    throw CancellationError()
                }
                throw error
            }
            try Task.checkCancellation()
            guard entries[key]?.generation == entry.generation else { throw CancellationError() }
            return value
        } onCancel: {
            Task { @MainActor in
                self.cancelWaiter(waiterID, key: key, generation: entry.generation)
            }
        }
    }

    @discardableResult
    @MainActor func cancelAll() -> [SharedModelLoadTask] {
        entries.removeAll()
        let tasks = Array(outstanding.values)
        tasks.forEach { $0.cancel() }
        return tasks.map { SharedModelLoadTask($0) }
    }

    @MainActor private func cancelWaiter(_ id: UUID, key: String, generation: UUID) {
        guard var entry = entries[key], entry.generation == generation,
              entry.waiterIDs.remove(id) != nil else { return }
        if entry.waiterIDs.isEmpty {
            entries[key] = nil
            entry.task.cancel()
        } else {
            entries[key] = entry
        }
    }

    @MainActor private func finishWaiter(_ id: UUID, key: String, generation: UUID) {
        // Called only after awaiting the underlying task, including cancelled
        // waiters. The generation can be retired even if its key was replaced.
        outstanding[generation] = nil
        guard var entry = entries[key], entry.generation == generation,
              entry.waiterIDs.remove(id) != nil else { return }
        entries[key] = entry.waiterIDs.isEmpty ? nil : entry
    }
}
