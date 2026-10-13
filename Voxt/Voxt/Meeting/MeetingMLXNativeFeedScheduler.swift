import Foundation

/// Serial, bounded delivery into the native stream. The permit covers delivery,
/// not the native backend's asynchronous GPU work. Session/model ownership must
/// remain pinned until the terminal event or explicit cancellation.
actor MeetingMLXNativeFeedScheduler {
    typealias Delivery = @Sendable ([Float]) async throws -> Void
    private let session: any MLXNativeStreamingSession
    private let deliver: Delivery
    private let maximumPendingSamples: Int
    private var pendingSamples: [Float] = []
    private var pendingOffset = 0
    private var drainTask: Task<Void, Never>?
    private var isStopping = false
    private var isCancelled = false
    private var failure: String?
    private var didStop = false

    init(
        session: any MLXNativeStreamingSession,
        maximumPendingSamples: Int = 160_000,
        deliver: Delivery? = nil
    ) {
        self.session = session
        self.maximumPendingSamples = maximumPendingSamples
        self.deliver = deliver ?? { samples in
            try await MeetingLocalInferenceCoordinator.shared.withPermit(.liveASRFeed) {
                try Task.checkCancellation()
                session.feedAudio(samples: samples)
            }
        }
    }

    func submit(samples: [Float], sampleRate: Double) -> Bool {
        guard !isStopping, !isCancelled, failure == nil else { return false }
        let prepared = ASRVoiceActivitySampleRateConverter.resample(samples: samples, from: sampleRate, to: 16_000)
        guard !prepared.isEmpty else { return true }
        guard pendingSampleCount + prepared.count <= maximumPendingSamples else {
            // Fail the live path rather than compressing the timeline by silently
            // omitting a frame. The coordinator already archives capture audio.
            failure = "The live audio queue is full. Recorded audio is preserved for final processing."
            return false
        }
        pendingSamples.append(contentsOf: prepared)
        startDrainIfNeeded()
        return true
    }

    var pendingSampleCount: Int { pendingSamples.count - pendingOffset }
    var failureMessage: String? { failure }

    func finish() async throws {
        isStopping = true
        if let drainTask { await drainTask.value }
        try Task.checkCancellation()
        if isCancelled { throw CancellationError() }
        if let failure { throw DeliveryError(message: failure) }
        guard pendingSampleCount == 0 else {
            throw DeliveryError(message: "Live audio delivery did not finish. Recorded audio is preserved.")
        }
        guard !didStop else { return }
        didStop = true
        session.stop()
    }

    func cancel() {
        guard !isCancelled else { return }
        isCancelled = true
        isStopping = true
        drainTask?.cancel()
        session.cancel()
        // A suspended delivery may resume after this reset. drain() rechecks
        // cancellation BEFORE advancing the offset, so it cannot corrupt it.
        pendingSamples.removeAll(keepingCapacity: false)
        pendingOffset = 0
    }

    private func startDrainIfNeeded() {
        guard drainTask == nil else { return }
        drainTask = Task { [weak self] in await self?.drain() }
    }

    private func drain() async {
        defer { drainTask = nil }
        while !isCancelled, failure == nil, pendingSampleCount > 0 {
            let end = min(pendingOffset + 3_200, pendingSamples.count)
            let chunk = Array(pendingSamples[pendingOffset..<end])
            do {
                try Task.checkCancellation()
                try await deliver(chunk)
                guard !isCancelled, !Task.isCancelled else { return }
                // Only successful delivery consumes input. Failure never advances
                // this offset and is propagated by finish / the next submit.
                pendingOffset = end
                if pendingOffset == pendingSamples.count {
                    pendingSamples.removeAll(keepingCapacity: true)
                    pendingOffset = 0
                } else if pendingOffset >= 16_000, pendingOffset * 2 >= pendingSamples.count {
                    pendingSamples.removeFirst(pendingOffset)
                    pendingOffset = 0
                }
            } catch {
                guard !isCancelled else { return }
                failure = error.localizedDescription
                return
            }
        }
    }

    struct DeliveryError: LocalizedError, Sendable {
        let message: String
        var errorDescription: String? { message }
    }
}
