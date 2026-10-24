// MeetingAudioAnalysisScheduler.swift
// Bounds and coalesces local meeting audio analysis work without dropping archived audio.

import Foundation

nonisolated struct MeetingAudioAnalysisFrame: Sendable {
    let samples: [Float]
    let sampleRate: Double
    let level: Float
    let speaker: MeetingSpeaker
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval

    var durationSeconds: TimeInterval {
        Double(samples.count) / max(sampleRate, 1)
    }

    func merging(_ next: MeetingAudioAnalysisFrame) -> MeetingAudioAnalysisFrame? {
        guard speaker == next.speaker,
              abs(sampleRate - next.sampleRate) <= 1,
              next.startSeconds >= startSeconds - 0.05
        else {
            return nil
        }
        // Capture callbacks contain continuous PCM even if MainActor delivery is briefly delayed,
        // so wall-clock timestamps can overlap. Reject only a material gap or reordering.
        guard next.startSeconds <= endSeconds + 0.25 else { return nil }

        var mergedSamples = samples
        mergedSamples.append(contentsOf: next.samples)
        let sampleClockEnd = startSeconds + Double(mergedSamples.count) / max(sampleRate, 1)
        return MeetingAudioAnalysisFrame(
            samples: mergedSamples,
            sampleRate: sampleRate,
            level: max(level, next.level),
            speaker: speaker,
            startSeconds: startSeconds,
            endSeconds: max(sampleClockEnd, next.endSeconds)
        )
    }
}

nonisolated struct MeetingAudioAnalysisStatistics: Equatable, Sendable {
    var submittedFrameCount = 0
    var mergedFrameCount = 0
    var processedBatchCount = 0
    var overloadedFrameCount = 0
    var peakPendingAudioSeconds: TimeInterval = 0
}

actor MeetingAudioAnalysisScheduler {
    enum SubmissionResult: Equatable, Sendable {
        case accepted
        case overloaded(pendingAudioSeconds: TimeInterval)
    }

    typealias Processor = @Sendable (MeetingAudioAnalysisFrame) async -> Void

    private static let maxPendingAudioSeconds: TimeInterval = 10

    private var pendingFrames: [MeetingSpeaker: MeetingAudioAnalysisFrame] = [:]
    private var drainTask: Task<Void, Never>?
    private var pendingProcessor: Processor?
    private var preferredNextSpeaker: MeetingSpeaker = .me
    private var statistics = MeetingAudioAnalysisStatistics()

    func submit(
        _ frame: MeetingAudioAnalysisFrame,
        processor: @escaping Processor
    ) -> SubmissionResult {
        statistics.submittedFrameCount += 1

        let existingDuration = pendingFrames[frame.speaker]?.durationSeconds ?? 0
        let pendingAfterSubmission = existingDuration + frame.durationSeconds
        if pendingAfterSubmission > Self.maxPendingAudioSeconds {
            statistics.overloadedFrameCount += 1
            statistics.peakPendingAudioSeconds = max(
                statistics.peakPendingAudioSeconds,
                pendingAudioSeconds
            )
            return .overloaded(pendingAudioSeconds: pendingAudioSeconds)
        }

        if let pending = pendingFrames[frame.speaker] {
            guard let merged = pending.merging(frame) else {
                statistics.overloadedFrameCount += 1
                return .overloaded(pendingAudioSeconds: pendingAudioSeconds)
            }
            pendingFrames[frame.speaker] = merged
            statistics.mergedFrameCount += 1
        } else {
            pendingFrames[frame.speaker] = frame
        }
        statistics.peakPendingAudioSeconds = max(
            statistics.peakPendingAudioSeconds,
            pendingAudioSeconds
        )

        pendingProcessor = processor
        startDrainIfNeeded()
        return .accepted
    }

    func flush() async {
        while let task = drainTask {
            await task.value
        }
    }

    func cancel() {
        drainTask?.cancel()
        pendingProcessor = nil
        pendingFrames.removeAll(keepingCapacity: false)
    }

    func currentStatistics() -> MeetingAudioAnalysisStatistics {
        statistics
    }

    func resetStatistics() {
        statistics = MeetingAudioAnalysisStatistics()
    }

    private func drain(processor: @escaping Processor) async {
        defer {
            drainTask = nil
            if !pendingFrames.isEmpty {
                startDrainIfNeeded()
            }
        }
        while !Task.isCancelled, let frame = popNextFrame() {
            await processor(frame)
            statistics.processedBatchCount += 1
        }
    }

    private func startDrainIfNeeded() {
        guard drainTask == nil, let pendingProcessor else { return }
        drainTask = Task { [weak self] in
            await self?.drain(processor: pendingProcessor)
        }
    }

    private func popNextFrame() -> MeetingAudioAnalysisFrame? {
        let first = preferredNextSpeaker
        let second: MeetingSpeaker = first == .me ? .them : .me
        if let frame = pendingFrames.removeValue(forKey: first) {
            preferredNextSpeaker = second
            return frame
        }
        if let frame = pendingFrames.removeValue(forKey: second) {
            preferredNextSpeaker = first
            return frame
        }
        return nil
    }

    private var pendingAudioSeconds: TimeInterval {
        pendingFrames.values.reduce(0) { $0 + $1.durationSeconds }
    }
}

/// Serializes live audio without coalescing frames, so VAD decisions and model
/// startup cannot reorder or drop speech at a session boundary.
actor MeetingOrderedLiveAudioScheduler {
    enum SubmissionResult: Equatable, Sendable {
        case accepted
        case overloaded(pendingAudioSeconds: TimeInterval)
    }

    typealias Processor = @Sendable (MeetingAudioAnalysisFrame) async -> Void

    private static let maxPendingAudioSeconds: TimeInterval = 10

    private var pendingFrames: [MeetingAudioAnalysisFrame] = []
    private var drainTask: Task<Void, Never>?
    private var pendingProcessor: Processor?
    private var statistics = MeetingAudioAnalysisStatistics()

    func submit(
        _ frame: MeetingAudioAnalysisFrame,
        processor: @escaping Processor
    ) -> SubmissionResult {
        statistics.submittedFrameCount += 1
        let pendingAfterSubmission = pendingAudioSeconds + frame.durationSeconds
        guard pendingAfterSubmission <= Self.maxPendingAudioSeconds else {
            statistics.overloadedFrameCount += 1
            statistics.peakPendingAudioSeconds = max(
                statistics.peakPendingAudioSeconds,
                pendingAudioSeconds
            )
            return .overloaded(pendingAudioSeconds: pendingAudioSeconds)
        }

        pendingFrames.append(frame)
        statistics.peakPendingAudioSeconds = max(
            statistics.peakPendingAudioSeconds,
            pendingAfterSubmission
        )
        pendingProcessor = processor
        startDrainIfNeeded()
        return .accepted
    }

    func flush() async {
        while let task = drainTask {
            await task.value
        }
    }

    func cancel() {
        drainTask?.cancel()
        pendingProcessor = nil
        pendingFrames.removeAll(keepingCapacity: false)
    }

    func currentStatistics() -> MeetingAudioAnalysisStatistics {
        statistics
    }

    func resetStatistics() {
        statistics = MeetingAudioAnalysisStatistics()
    }

    private func drain(processor: @escaping Processor) async {
        defer {
            drainTask = nil
            if !pendingFrames.isEmpty {
                startDrainIfNeeded()
            }
        }
        while !Task.isCancelled, !pendingFrames.isEmpty {
            let frame = pendingFrames.removeFirst()
            await processor(frame)
            statistics.processedBatchCount += 1
        }
    }

    private func startDrainIfNeeded() {
        guard drainTask == nil, let pendingProcessor else { return }
        drainTask = Task { [weak self] in
            await self?.drain(processor: pendingProcessor)
        }
    }

    private var pendingAudioSeconds: TimeInterval {
        pendingFrames.reduce(0) { $0 + $1.durationSeconds }
    }
}
