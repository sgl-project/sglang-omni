import Foundation

nonisolated protocol ASRStreamingSession: AnyObject, Sendable {
    var events: AsyncStream<ASRStreamingEvent> { get }
    func feedAudio(samples: [Float])
    func stop()
    func cancel()
}

nonisolated struct MLXMeetingNativeStreamingConfiguration: Sendable {
    let session: any ASRStreamingSession
    let liveMode: MLXLiveMode
    let qwenUsesAutomaticLanguageProtocol: Bool
    let mossVisibleOutputMode: MossASROutputMode?
}

/// Owns the installed stream, its two consumer/feed tasks and the transferred
/// model-use release. A retiring stream cannot clear its replacement's state.
/// HTTP sessions hold their own runtime use until the forwarding task exits;
/// the runtime separately waits for active uses and actual server process exit.
@MainActor
final class MLXNativeLiveRuntime {
    private struct Active {
        let id: UUID
        let session: any ASRStreamingSession
        let eventTask: Task<Void, Never>
        let feedTask: Task<Void, Never>
        let releaseModel: @MainActor () -> Void
    }

    private var active: Active?
    private var retirementTasks: [UUID: Task<Void, Never>] = [:]
    var session: (any ASRStreamingSession)? { active?.session }
    var hasPendingWork: Bool { active != nil || !retirementTasks.isEmpty }

    func install(
        _ session: any ASRStreamingSession,
        pollInterval: Duration,
        nextSamples: @escaping @MainActor () -> [Float],
        shouldContinue: @escaping @MainActor () -> Bool,
        onEvent: @escaping @MainActor (ASRStreamingEvent) -> Void,
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
