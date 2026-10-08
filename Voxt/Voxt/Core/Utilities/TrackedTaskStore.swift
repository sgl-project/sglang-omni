import Foundation

/// Retains cancelled work until it actually exits. Cancellation is a request,
/// not proof that inference/capture has released the resources it still owns.
@MainActor
final class TrackedTaskStore {
    private var tasks: [UUID: Task<Void, Never>] = [:]

    var isEmpty: Bool { tasks.isEmpty }
    var count: Int { tasks.count }
    var snapshot: [Task<Void, Never>] { Array(tasks.values) }

    @discardableResult
    func start(
        after dependencies: [Task<Void, Never>] = [],
        operation: @escaping @MainActor () async -> Void
    ) -> Task<Void, Never> {
        // Identity belongs to the invocation, not a possibly reused request ID.
        let token = UUID()
        let task = Task { @MainActor [weak self] in
            defer { self?.tasks[token] = nil }
            for dependency in dependencies { await dependency.value }
            guard !Task.isCancelled, self != nil else { return }
            await operation()
        }
        tasks[token] = task
        return task
    }

    @discardableResult
    func cancelAll() -> [Task<Void, Never>] {
        let pending = snapshot
        pending.forEach { $0.cancel() }
        return pending
    }

    func waitForAll() async {
        while !tasks.isEmpty {
            for task in snapshot { await task.value }
        }
    }

    deinit {
        tasks.values.forEach { $0.cancel() }
    }
}
