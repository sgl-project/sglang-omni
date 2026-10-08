import Foundation

struct MLXCorrectionPassResult {
    let text: String?
    let error: Error?

    static func success(_ text: String?) -> MLXCorrectionPassResult {
        MLXCorrectionPassResult(text: text, error: nil)
    }

    static func failure(_ error: Error) -> MLXCorrectionPassResult {
        MLXCorrectionPassResult(text: nil, error: error)
    }
}

/// Serial ownership of correction inference. A cancelled pass keeps its slot
/// until it exits and releases its model use; a replacement cannot overlap it.
@MainActor
final class MLXCorrectionPassCoordinator {
    private struct Entry {
        let id: UUID
        let kind: MLXCorrectionPassKind
        let task: Task<MLXCorrectionPassResult, Never>
    }

    private var active: Entry?
    var hasPendingWork: Bool { active != nil }
    var currentTask: Task<MLXCorrectionPassResult, Never>? { active?.task }

    func cancel() {
        active?.task.cancel()
    }

    func run(
        kind: MLXCorrectionPassKind,
        isCurrent: @escaping @MainActor () -> Bool,
        operation: @escaping @MainActor () async -> MLXCorrectionPassResult
    ) async -> MLXCorrectionPassResult {
        guard !Task.isCancelled, isCurrent() else { return .success(nil) }
        while let previous = active {
            switch MLXTranscriptionPlanning.correctionPassSchedulingDecision(
                requestedPass: kind, inFlightPass: previous.kind
            ) {
            case .skipRequestedPass:
                return .success(nil)
            case .interruptInFlightPass:
                previous.task.cancel()
            case .startImmediately, .waitForInFlightPass:
                break
            }
            _ = await previous.task.value
            clear(ifMatching: previous.id)
            guard !Task.isCancelled, isCurrent() else { return .success(nil) }
        }

        let id = UUID()
        let task = Task { @MainActor in
            guard !Task.isCancelled, isCurrent() else { return MLXCorrectionPassResult.success(nil) }
            return await operation()
        }
        active = Entry(id: id, kind: kind, task: task)
        let result = await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
        clear(ifMatching: id)
        guard !Task.isCancelled, !task.isCancelled, isCurrent() else { return .success(nil) }
        return result
    }

    private func clear(ifMatching id: UUID) {
        guard active?.id == id else { return }
        active = nil
    }
}
