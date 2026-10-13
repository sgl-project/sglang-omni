// RecentAudioWaveformState.swift
// Provides Recent Audio Waveform State for shared utilities.

import Foundation
import Combine

struct RecentAudioWaveformModel {
    let barCount: Int

    private let sampleCount: Int
    private let minimumLevel: Float
    private let peakHoldFrames: Int
    private let peakDecayFactor: Float
    private let riseSmoothing: Float
    private let fallSmoothing: Float

    private(set) var barLevels: [Float]
    private var history: [Float]
    private var pendingInputLevel: Float = 0
    private var smoothedLevel: Float = 0
    private var heldPeakLevel: Float = 0
    private var remainingHoldFrames = 0

    init(
        barCount: Int = 16,
        historyDuration: TimeInterval = 2.0,
        framesPerSecond: Double = 18,
        silenceFloor: Float = 0,
        peakHoldFrames: Int = 1,
        peakDecayFactor: Float = 0.82,
        riseSmoothing: Float = 0.92,
        fallSmoothing: Float = 0.1
    ) {
        self.barCount = max(barCount, 1)
        self.sampleCount = max(Int((max(historyDuration, 0.5) * max(framesPerSecond, 1)).rounded()), self.barCount)
        self.minimumLevel = max(0, min(silenceFloor, 0.03))
        self.peakHoldFrames = max(peakHoldFrames, 0)
        self.peakDecayFactor = min(max(peakDecayFactor, 0.5), 0.98)
        self.riseSmoothing = min(max(riseSmoothing, 0.05), 1)
        self.fallSmoothing = min(max(fallSmoothing, 0.02), 1)

        let seed = Array(repeating: self.minimumLevel, count: self.sampleCount)
        self.history = seed
        self.barLevels = Array(repeating: self.minimumLevel, count: self.barCount)
    }

    mutating func ingest(level: Float) {
        pendingInputLevel = max(pendingInputLevel, max(0, min(level, 1)))
    }

    mutating func reset() {
        pendingInputLevel = 0
        smoothedLevel = 0
        heldPeakLevel = 0
        remainingHoldFrames = 0
        history = Array(repeating: minimumLevel, count: sampleCount)
        barLevels = Array(repeating: minimumLevel, count: barCount)
    }

    mutating func advanceFrame() {
        let targetLevel = normalizedInputLevel(pendingInputLevel)
        pendingInputLevel = 0

        let smoothing = targetLevel >= smoothedLevel ? riseSmoothing : fallSmoothing
        smoothedLevel += (targetLevel - smoothedLevel) * smoothing

        if smoothedLevel >= heldPeakLevel {
            heldPeakLevel = smoothedLevel
            remainingHoldFrames = peakHoldFrames
        } else if remainingHoldFrames > 0 {
            remainingHoldFrames -= 1
        } else {
            heldPeakLevel = max(smoothedLevel, heldPeakLevel * peakDecayFactor)
        }

        let frameLevel = max(minimumLevel, min(heldPeakLevel, 1))
        if !history.isEmpty {
            history.removeFirst()
        }
        history.append(frameLevel)
        barLevels = bucketedBars(from: history)
    }

    private func normalizedInputLevel(_ rawLevel: Float) -> Float {
        let clamped = max(0, min(rawLevel, 1))
        return Float(min(1.0, pow(Double(clamped), 0.96)))
    }

    private func bucketedBars(from samples: [Float]) -> [Float] {
        guard !samples.isEmpty else {
            return Array(repeating: minimumLevel, count: barCount)
        }

        let sampleCount = samples.count
        let barCountAsDouble = Double(barCount)
        let sampleCountAsDouble = Double(sampleCount)
        var bars: [Float] = []
        bars.reserveCapacity(barCount)

        for index in 0..<barCount {
            let start = Int((Double(index) / barCountAsDouble) * sampleCountAsDouble)
            let end = Int((Double(index + 1) / barCountAsDouble) * sampleCountAsDouble)
            let clampedStart = min(max(start, 0), sampleCount - 1)
            let clampedEnd = min(max(end, clampedStart + 1), sampleCount)
            let bucket = samples[clampedStart..<clampedEnd]
            let peak = bucket.max() ?? minimumLevel
            let average = bucket.reduce(0, +) / Float(max(bucket.count, 1))
            bars.append(max(minimumLevel, min(1, peak * 0.68 + average * 0.32)))
        }

        return bars
    }
}

@MainActor
final class RecentAudioWaveformState: ObservableObject {
    @Published private(set) var barLevels: [Float]

    let barCount: Int

    private let framesPerSecond: Double
    private var model: RecentAudioWaveformModel
    private var timer: Timer?
    private var isActive = false

    init(
        barCount: Int = 16,
        historyDuration: TimeInterval = 2.0,
        framesPerSecond: Double = 18,
        silenceFloor: Float = 0,
        peakHoldFrames: Int = 1,
        peakDecayFactor: Float = 0.8,
        riseSmoothing: Float = 0.9,
        fallSmoothing: Float = 0.11
    ) {
        let model = RecentAudioWaveformModel(
            barCount: barCount,
            historyDuration: historyDuration,
            framesPerSecond: framesPerSecond,
            silenceFloor: silenceFloor,
            peakHoldFrames: peakHoldFrames,
            peakDecayFactor: peakDecayFactor,
            riseSmoothing: riseSmoothing,
            fallSmoothing: fallSmoothing
        )
        self.model = model
        self.barCount = model.barCount
        self.framesPerSecond = max(framesPerSecond, 1)
        self.barLevels = model.barLevels
    }

    deinit {
        timer?.invalidate()
    }

    func setActive(_ active: Bool) {
        guard isActive != active else { return }
        isActive = active
        if active {
            startTimerIfNeeded()
        } else {
            stopTimer()
            reset()
        }
    }

    func ingest(level: Float) {
        model.ingest(level: level)
    }

    func reset() {
        model.reset()
        barLevels = model.barLevels
    }

#if DEBUG
    func debugAdvanceFrame() {
        advanceFrame()
    }
#endif

    private func startTimerIfNeeded() {
        guard timer == nil else { return }
        let interval = 1.0 / framesPerSecond
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.advanceFrame()
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }

    private func advanceFrame() {
        model.advanceFrame()
        barLevels = model.barLevels
    }
}
