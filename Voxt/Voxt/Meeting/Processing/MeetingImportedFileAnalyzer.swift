import Foundation

@MainActor
protocol MeetingImportedFileAnalyzing: AnyObject {
    func analyze(at url: URL, progress: @escaping @MainActor @Sendable (MeetingFileAnalysisProgress) -> Void) async throws -> MeetingSessionResult
    func cancel() async
    func finish(keepingResult: Bool) async
}

/// Registers the operation before awaiting old-session cleanup. Cancellation in
/// that window must stop this import, not disappear because its task is still nil.
@MainActor
final class MeetingImportedFileAnalyzer {
    private struct Active {
        let id: UUID
        let task: Task<MeetingSessionResult, Error>
        let pipeline: any MeetingImportedFileAnalyzing
    }
    private var active: Active?
    var isRunning: Bool { active != nil }

    func analyze(
        at url: URL,
        after cleanup: Task<Void, Never>?,
        using pipeline: any MeetingImportedFileAnalyzing,
        progress: @escaping @MainActor @Sendable (MeetingFileAnalysisProgress) -> Void
    ) async throws -> MeetingSessionResult {
        let task = try start(at: url, after: cleanup, using: pipeline, progress: progress)
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    func start(
        at url: URL,
        after cleanup: Task<Void, Never>?,
        using pipeline: any MeetingImportedFileAnalyzing,
        progress: @escaping @MainActor @Sendable (MeetingFileAnalysisProgress) -> Void
    ) throws -> Task<MeetingSessionResult, Error> {
        guard active == nil else { throw MeetingFileAnalysisError.sessionAlreadyActive }
        let id = UUID()
        let task = Task(priority: .utility) { @MainActor [weak self] in
            defer { if self?.active?.id == id { self?.active = nil } }
            do {
                await cleanup?.value
                try Task.checkCancellation()
                let result = try await pipeline.analyze(at: url, progress: progress)
                try Task.checkCancellation()
                await pipeline.finish(keepingResult: true)
                try Task.checkCancellation()
                return result
            } catch {
                // Idempotent cleanup also removes an otherwise successful result
                // if cancellation arrived while its resources were being released.
                await pipeline.finish(keepingResult: false)
                throw error
            }
        }
        active = Active(id: id, task: task, pipeline: pipeline)
        return task
    }

    isolated deinit {
        guard let retired = active else { return }
        retired.task.cancel()
        Task { @MainActor in await retired.pipeline.cancel() }
    }

    /// Snapshot first, so a delayed cancellation can never target a later import.
    @discardableResult
    func cancel() -> Task<Void, Never>? {
        guard let retired = active else { return nil }
        retired.task.cancel()
        return Task { @MainActor in
            await retired.pipeline.cancel()
            _ = await retired.task.result
        }
    }
}
