import Foundation

/// Intentionally ignores task cancellation until released, like an in-flight
/// native operation. Tests use it to distinguish cancel requests from completion.
@MainActor
final class ManualTaskBarrier {
    private var entered = false
    private var released = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var completionWaiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        entered = true
        let waiting = entryWaiters
        entryWaiters.removeAll()
        waiting.forEach { $0.resume() }
        guard !released else { return }
        await withCheckedContinuation { completionWaiters.append($0) }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { entryWaiters.append($0) }
    }

    func release() {
        released = true
        let waiting = completionWaiters
        completionWaiters.removeAll()
        waiting.forEach { $0.resume() }
    }
}
