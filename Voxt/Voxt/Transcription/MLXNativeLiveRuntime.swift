import Foundation
import MLXAudioSTT

nonisolated protocol MLXNativeStreamingSession: AnyObject, Sendable {
    var events: AsyncStream<TranscriptionEvent> { get }
    func feedAudio(samples: [Float])
    func stop()
    func cancel()
}

extension StreamingInferenceSession: MLXNativeStreamingSession {}
extension NemotronASRStreamingSession: MLXNativeStreamingSession {}

nonisolated struct MLXMeetingNativeStreamingConfiguration: Sendable {
    let session: any MLXNativeStreamingSession
    let liveMode: MLXLiveMode
    let qwenUsesAutomaticLanguageProtocol: Bool
    let mossVisibleOutputMode: MossASROutputMode?
}

/// Owns the installed stream, its two consumer/feed tasks and the transferred
/// model-use release. A retiring stream cannot clear its replacement's state.
/// This waits for Voxt's tasks only: the library's synchronous cancel() does not
/// expose an awaitable barrier for its internal decode/Metal work.
@MainActor
final class MLXNativeLiveRuntime {
    private struct Active {
        let id: UUID
        let session: any MLXNativeStreamingSession
        let eventTask: Task<Void, Never>
        let feedTask: Task<Void, Never>
        let releaseModel: @MainActor () -> Void
    }

    private var active: Active?
    private var retirementTasks: [UUID: Task<Void, Never>] = [:]
    var session: (any MLXNativeStreamingSession)? { active?.session }
    var hasPendingWork: Bool { active != nil || !retirementTasks.isEmpty }

    func install(
        _ session: any MLXNativeStreamingSession,
        pollInterval: Duration,
        nextSamples: @escaping @MainActor () -> [Float],
        shouldContinue: @escaping @MainActor () -> Bool,
        onEvent: @escaping @MainActor (TranscriptionEvent) -> Void,
        releaseModel: @escaping @MainActor () -> Void
    ) {
        release(cancelSession: true)
        let id = UUID()
        let events = Task { @MainActor [weak self, session] in
            for await event in session.events {
                guard !Task.isCancelled, self?.active?.id == id else { return }
                onEvent(event)
            }
        }
        let feed = Task { @MainActor [weak self, session] in
            while !Task.isCancelled, self?.active?.id == id {
                let samples = nextSamples()
                if !samples.isEmpty { session.feedAudio(samples: samples) }
                guard shouldContinue() else { return }
                do { try await Task.sleep(for: pollInterval) } catch { return }
            }
        }
        active = Active(id: id, session: session, eventTask: events, feedTask: feed, releaseModel: releaseModel)
    }

    func stopFeeding() {
        active?.feedTask.cancel()
    }

    func release(cancelSession: Bool) {
        guard let retired = active else { return }
        active = nil
        retired.feedTask.cancel()
        retired.eventTask.cancel()
        if cancelSession { retired.session.cancel() }
        // Cleanup must run even during shutdown, so it is not a cancellable work
        // submission. Retain this owner until both Voxt tasks exit and the use is released.
        retirementTasks[retired.id] = Task { @MainActor [self] in
            await retired.feedTask.value
            await retired.eventTask.value
            retired.releaseModel()
            retirementTasks[retired.id] = nil
        }
    }

    func waitForRetirement() async {
        while !retirementTasks.isEmpty {
            for task in Array(retirementTasks.values) { await task.value }
        }
    }

    isolated deinit {
        guard let retired = active else { return }
        retired.feedTask.cancel()
        retired.eventTask.cancel()
        retired.session.cancel()
        // Capture the entry, never self: an abandoned transcriber must not leak
        // its stream or model use even if explicit shutdown was not called.
        Task { @MainActor in
            await retired.feedTask.value
            await retired.eventTask.value
            retired.releaseModel()
        }
    }
}
