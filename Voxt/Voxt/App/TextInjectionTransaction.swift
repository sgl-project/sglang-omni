import Foundation

/// One queued paste attempt. Validate immediately before injection; report its
/// outcome once, without claiming an editor acknowledgement or undoing a post.
/// The consumer separately checks its session before updating UI/history.
@MainActor
final class TextInjectionTransaction {
    typealias Completion = @MainActor (Bool) -> Void
    private let isValid: @MainActor () -> Bool
    private let inject: @MainActor (@escaping Completion) -> Void
    private let completion: Completion?
    private var hasStarted = false
    private var hasCompleted = false

    init(
        isValid: @escaping @MainActor () -> Bool,
        inject: @escaping @MainActor (@escaping Completion) -> Void,
        completion: Completion? = nil
    ) {
        self.isValid = isValid
        self.inject = inject
        self.completion = completion
    }

    func perform() {
        guard !hasStarted else { return }
        hasStarted = true
        guard isValid() else { finish(false); return }
        inject { [self] didInject in finish(didInject) }
    }

    private func finish(_ didInject: Bool) {
        guard !hasCompleted else { return }
        hasCompleted = true
        completion?(didInject)
    }
}
