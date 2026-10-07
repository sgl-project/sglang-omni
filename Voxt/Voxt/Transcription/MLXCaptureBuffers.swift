// Capture buffers keep their existing locks; the transcriber remains their sole owner.

import Foundation

struct MLXVoiceActivitySampleContextBuffer {
    static let defaultMaximumContextSeconds: TimeInterval = 0.35

    private let maximumContextSeconds: TimeInterval
    private var pendingFrames: [ASRVoiceActivityAudioFrame] = []
    private var pendingDurationSeconds: TimeInterval = 0
    private(set) var observedFrames = false
    private(set) var observedSpeech = false

    init(maximumContextSeconds: TimeInterval = Self.defaultMaximumContextSeconds) {
        self.maximumContextSeconds = max(0, maximumContextSeconds)
    }

    mutating func append(_ frame: ASRVoiceActivityAudioFrame, isSpeech: Bool) -> [Float] {
        observedFrames = true
        guard isSpeech else {
            appendPendingFrame(frame)
            return []
        }

        observedSpeech = true
        let contextSamples = flushPendingSamples()
        return contextSamples + frame.samples
    }

    mutating func reset() {
        pendingFrames.removeAll(keepingCapacity: false)
        pendingDurationSeconds = 0
        observedFrames = false
        observedSpeech = false
    }

    mutating func finish() -> [Float] {
        // Drop trailing non-speech after the last speech burst. Pre-roll before
        // speech onsets is already flushed in `append`; trailing pad only lengthens
        // Final ASR input without helping offline recognition.
        pendingFrames.removeAll(keepingCapacity: false)
        pendingDurationSeconds = 0
        return []
    }

    private mutating func appendPendingFrame(_ frame: ASRVoiceActivityAudioFrame) {
        guard !frame.samples.isEmpty else { return }
        pendingFrames.append(frame)
        pendingDurationSeconds += Self.durationSeconds(for: frame)
        while pendingDurationSeconds > maximumContextSeconds,
              let removed = pendingFrames.first {
            pendingFrames.removeFirst()
            pendingDurationSeconds -= Self.durationSeconds(for: removed)
        }
    }

    private mutating func flushPendingSamples() -> [Float] {
        guard !pendingFrames.isEmpty else { return [] }
        let contextSamples = pendingFrames.flatMap(\.samples)
        pendingFrames.removeAll(keepingCapacity: true)
        pendingDurationSeconds = 0
        return contextSamples
    }

    private static func durationSeconds(for frame: ASRVoiceActivityAudioFrame) -> TimeInterval {
        let timestampDuration = frame.endSeconds - frame.startSeconds
        if timestampDuration.isFinite, timestampDuration > 0 {
            return timestampDuration
        }
        if frame.sampleRate.isFinite, frame.sampleRate > 0 {
            return Double(frame.samples.count) / frame.sampleRate
        }
        return 0
    }
}

/// Coalesces realtime meter samples while the main actor is busy. At most one delivery task is
/// queued, so a cold model load cannot build up seconds of stale waveform updates.
nonisolated final class MLXAudioLevelDelivery: @unchecked Sendable {
    private let lock = NSLock()
    private var latestLevel: Float?
    private var isDeliveryScheduled = false

    func submit(_ level: Float, deliver: @escaping @MainActor @Sendable (Float) -> Void) {
        lock.lock()
        latestLevel = level
        let shouldSchedule = !isDeliveryScheduled
        if shouldSchedule {
            isDeliveryScheduled = true
        }
        lock.unlock()

        guard shouldSchedule else { return }
        Task { @MainActor [weak self] in
            guard let level = self?.takeLatestLevel() else { return }
            deliver(level)
        }
    }

    func clear() {
        lock.lock()
        latestLevel = nil
        lock.unlock()
    }

    private func takeLatestLevel() -> Float? {
        lock.lock()
        defer { lock.unlock() }
        let level = latestLevel
        latestLevel = nil
        isDeliveryScheduled = false
        return level
    }
}

extension MLXTranscriber {
    final class AudioSampleStore {
        private let lock = NSLock()
        private var samples: [Float] = []
        private var callbackCount: Int = 0
        private var enabled = false
        private var voiceActivityContextBuffer = MLXVoiceActivitySampleContextBuffer()

        func noteCallback() {
            lock.lock()
            defer { lock.unlock() }
            callbackCount += 1
        }

        func append(_ newSamples: [Float]) {
            lock.lock()
            defer { lock.unlock() }
            samples.append(contentsOf: newSamples)
        }

        func configureVoiceActivityFiltering(enabled: Bool) {
            lock.lock()
            defer { lock.unlock() }
            self.enabled = enabled
            voiceActivityContextBuffer.reset()
            samples.removeAll(keepingCapacity: false)
        }

        func appendVoiceActivityFrame(_ frame: ASRVoiceActivityAudioFrame, isSpeech: Bool) {
            lock.lock()
            defer { lock.unlock() }
            guard enabled else { return }
            samples.append(contentsOf: voiceActivityContextBuffer.append(frame, isSpeech: isSpeech))
        }

        func finishVoiceActivityFiltering() {
            lock.lock()
            defer { lock.unlock() }
            guard enabled else { return }
            samples.append(contentsOf: voiceActivityContextBuffer.finish())
        }

        func voiceActivityState() -> (enabled: Bool, observedFrames: Bool, observedSpeech: Bool) {
            lock.lock()
            defer { lock.unlock() }
            return (enabled, voiceActivityContextBuffer.observedFrames, voiceActivityContextBuffer.observedSpeech)
        }

        func snapshot() -> [Float] {
            lock.lock()
            defer { lock.unlock() }
            return samples
        }

        func count() -> Int {
            lock.lock()
            defer { lock.unlock() }
            return samples.count
        }

        func samples(from startIndex: Int) -> (samples: [Float], nextIndex: Int) {
            lock.lock()
            defer { lock.unlock() }

            let clampedStart = max(0, min(startIndex, samples.count))
            guard clampedStart < samples.count else { return ([], samples.count) }
            return (Array(samples[clampedStart...]), samples.count)
        }

        func callbacksReceived() -> Int {
            lock.lock()
            defer { lock.unlock() }
            return callbackCount
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            samples.removeAll(keepingCapacity: false)
            callbackCount = 0
            enabled = false
            voiceActivityContextBuffer.reset()
        }

        func tail(sampleCount: Int) -> [Float] {
            lock.lock()
            defer { lock.unlock() }
            guard sampleCount > 0, sampleCount < samples.count else { return samples }
            return Array(samples.suffix(sampleCount))
        }
    }

    final class VoiceActivityFrameStore {
        private let lock = NSLock()
        private var frames: [ASRVoiceActivityAudioFrame] = []
        private var cursorSeconds: TimeInterval = 0

        func append(samples: [Float], sampleRate: Double, level: Float?) {
            guard !samples.isEmpty else { return }
            lock.lock()
            defer { lock.unlock() }

            let finiteRate = sampleRate.isFinite && sampleRate > 0 ? sampleRate : 0
            let duration = finiteRate > 0 ? Double(samples.count) / finiteRate : 0
            let startSeconds = cursorSeconds
            let endSeconds = startSeconds + duration
            cursorSeconds = endSeconds
            frames.append(
                ASRVoiceActivityAudioFrame(
                    samples: samples,
                    sampleRate: finiteRate,
                    startSeconds: startSeconds,
                    endSeconds: endSeconds,
                    level: level
                )
            )
        }

        func drain() -> [ASRVoiceActivityAudioFrame] {
            lock.lock()
            defer { lock.unlock() }
            guard !frames.isEmpty else { return [] }
            let drained = frames
            frames.removeAll(keepingCapacity: true)
            return drained
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            frames.removeAll(keepingCapacity: false)
            cursorSeconds = 0
        }
    }
}
