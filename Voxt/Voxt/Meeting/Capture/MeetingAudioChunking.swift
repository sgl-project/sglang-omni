// MeetingAudioChunking.swift
// Provides Meeting Audio Chunking for meeting capture.

import Foundation
import AVFoundation

struct BufferedMeetingChunk: Sendable {
    let segmentID: UUID
    let speaker: MeetingSpeaker
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    let sampleRate: Double
    let samples: [Float]
    let isFinal: Bool
    let preventsAdjacentMerge: Bool

    nonisolated init(
        segmentID: UUID,
        speaker: MeetingSpeaker,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        sampleRate: Double,
        samples: [Float],
        isFinal: Bool,
        preventsAdjacentMerge: Bool = false
    ) {
        self.segmentID = segmentID
        self.speaker = speaker
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.sampleRate = sampleRate
        self.samples = samples
        self.isFinal = isFinal
        self.preventsAdjacentMerge = preventsAdjacentMerge
    }
}

enum MeetingChunkingProfile: Equatable, Sendable {
    case quality
    case realtime

    struct Configuration: Equatable, Sendable {
        let silenceFlushSeconds: TimeInterval
        let minSpeechSeconds: TimeInterval
        let maxChunkSeconds: TimeInterval
        let partialEmitIntervalSeconds: TimeInterval?
    }

}

enum MeetingChunkingMode: String, CaseIterable, Identifiable, Codable, Hashable, Sendable {
    case automatic
    case quality
    case realtime

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic:
            return AppLocalization.localizedString("Auto")
        case .quality:
            return AppLocalization.localizedString("Quality")
        case .realtime:
            return AppLocalization.localizedString("Realtime")
        }
    }

    var detail: String {
        switch self {
        case .automatic:
            return AppLocalization.localizedString("Use the best available chunking mode for the selected meeting model.")
        case .quality:
            return AppLocalization.localizedString("Prefer longer chunks for better context and smoother transcripts.")
        case .realtime:
            return AppLocalization.localizedString("Prefer shorter chunks for lower latency.")
        }
    }

    static func stored(in defaults: UserDefaults = .standard) -> MeetingChunkingMode {
        let rawValue = defaults.string(forKey: AppPreferenceKey.meetingChunkingMode) ?? ""
        return MeetingChunkingMode(rawValue: rawValue) ?? .automatic
    }

    func resolvedProfile(automaticProfile: MeetingChunkingProfile) -> MeetingChunkingProfile {
        switch self {
        case .automatic:
            return automaticProfile
        case .quality:
            return .quality
        case .realtime:
            return .realtime
        }
    }
}

enum MeetingFinalTranscriptOptimization {
    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: AppPreferenceKey.meetingFinalTranscriptOptimizationEnabled) as? Bool ?? true
    }
}

actor MeetingChunkAccumulator {
    private static let defaultSampleRate = 16_000.0

    private let speaker: MeetingSpeaker
    private let speechThreshold: Float
    private let config: MeetingChunkingProfile.Configuration

    private var currentSamples: [Float] = []
    private var currentStartSeconds: TimeInterval?
    private var currentSampleRate: Double = defaultSampleRate
    private var accumulatedSilenceSeconds: TimeInterval = 0
    private var currentSegmentID = UUID()
    private var lastPartialEmissionDuration: TimeInterval = 0

    init(speaker: MeetingSpeaker, speechThreshold: Float, profile: MeetingChunkingProfile) {
        self.speaker = speaker
        self.speechThreshold = speechThreshold
        switch profile {
        case .quality:
            self.config = .init(
                silenceFlushSeconds: 0.45,
                minSpeechSeconds: 0.35,
                maxChunkSeconds: 7.0,
                partialEmitIntervalSeconds: nil
            )
        case .realtime:
            self.config = .init(
                silenceFlushSeconds: 0.30,
                minSpeechSeconds: 0.18,
                maxChunkSeconds: 6.0,
                partialEmitIntervalSeconds: 2.0
            )
        }
    }

    func append(
        samples: [Float],
        sampleRate: Double,
        level: Float,
        voiceActivityIsSpeech: Bool? = nil,
        bufferEndSeconds: TimeInterval
    ) -> BufferedMeetingChunk? {
        guard !samples.isEmpty, sampleRate > 0 else { return nil }
        let isSpeech = voiceActivityIsSpeech ?? (level >= speechThreshold)
        let bufferDuration = Double(samples.count) / sampleRate
        let bufferStartSeconds = max(bufferEndSeconds - bufferDuration, 0)

        if currentStartSeconds == nil {
            guard isSpeech else { return nil }
            currentStartSeconds = bufferStartSeconds
            currentSampleRate = sampleRate
            currentSamples.removeAll(keepingCapacity: true)
            currentSegmentID = UUID()
            lastPartialEmissionDuration = 0
        }

        if abs(currentSampleRate - sampleRate) > 1 {
            if let flushed = flushCurrent(endSeconds: bufferStartSeconds, reason: .sampleRateChange) {
                currentStartSeconds = bufferStartSeconds
                currentSampleRate = sampleRate
                currentSamples = samples
                currentSegmentID = UUID()
                lastPartialEmissionDuration = 0
                accumulatedSilenceSeconds = isSpeech ? 0 : bufferDuration
                return flushed
            }
            currentStartSeconds = bufferStartSeconds
            currentSampleRate = sampleRate
            currentSamples.removeAll(keepingCapacity: true)
            currentSegmentID = UUID()
            lastPartialEmissionDuration = 0
        }

        currentSamples.append(contentsOf: samples)

        if isSpeech {
            accumulatedSilenceSeconds = 0
        } else {
            accumulatedSilenceSeconds += bufferDuration
        }

        let currentDuration = Double(currentSamples.count) / currentSampleRate
        let bufferEndSeconds = bufferStartSeconds + bufferDuration

        if accumulatedSilenceSeconds >= config.silenceFlushSeconds {
            return flushCurrent(endSeconds: bufferEndSeconds, reason: .pause)
        }

        if currentDuration >= config.maxChunkSeconds {
            return flushCurrent(endSeconds: bufferEndSeconds, reason: .duration)
        }

        if let partialEmitIntervalSeconds = config.partialEmitIntervalSeconds,
           isSpeech,
           currentDuration >= config.minSpeechSeconds,
           currentDuration - lastPartialEmissionDuration >= partialEmitIntervalSeconds {
            lastPartialEmissionDuration = currentDuration
            return makeChunk(endSeconds: bufferEndSeconds, isFinal: false)
        }

        return nil
    }

    func finish(at endSeconds: TimeInterval) -> BufferedMeetingChunk? {
        flushCurrent(endSeconds: endSeconds, reason: .finish)
    }

    private func flushCurrent(
        endSeconds: TimeInterval,
        reason: MeetingChunkFlushReason
    ) -> BufferedMeetingChunk? {
        guard let currentStartSeconds else { return nil }
        let duration = Double(currentSamples.count) / max(currentSampleRate, 1)
        defer {
            self.currentStartSeconds = nil
            self.currentSamples.removeAll(keepingCapacity: false)
            self.accumulatedSilenceSeconds = 0
            self.lastPartialEmissionDuration = 0
            self.currentSegmentID = UUID()
        }
        guard duration >= config.minSpeechSeconds else {
            return nil
        }
        return makeChunk(
            segmentID: currentSegmentID,
            startSeconds: currentStartSeconds,
            endSeconds: max(endSeconds, currentStartSeconds),
            isFinal: true,
            preventsAdjacentMerge: reason.preventsAdjacentMerge
        )
    }

    private func makeChunk(endSeconds: TimeInterval, isFinal: Bool) -> BufferedMeetingChunk? {
        guard let currentStartSeconds else { return nil }
        return makeChunk(
            segmentID: currentSegmentID,
            startSeconds: currentStartSeconds,
            endSeconds: max(endSeconds, currentStartSeconds),
            isFinal: isFinal,
            preventsAdjacentMerge: false
        )
    }

    private func makeChunk(
        segmentID: UUID,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        isFinal: Bool,
        preventsAdjacentMerge: Bool
    ) -> BufferedMeetingChunk {
        BufferedMeetingChunk(
            segmentID: segmentID,
            speaker: speaker,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            sampleRate: currentSampleRate,
            samples: currentSamples,
            isFinal: isFinal,
            preventsAdjacentMerge: preventsAdjacentMerge
        )
    }
}

private enum MeetingChunkFlushReason: Sendable {
    case pause
    case duration
    case sampleRateChange
    case finish

    nonisolated var preventsAdjacentMerge: Bool {
        switch self {
        case .pause:
            return true
        case .duration, .sampleRateChange, .finish:
            return false
        }
    }
}

enum MeetingAudioChunkWAVExporter {
    static func write(samples: [Float], sampleRate: Int, to destinationURL: URL) throws {
        let normalizedSampleRate = max(sampleRate, 1)
        let data = wavData(for: samples, sampleRate: normalizedSampleRate)
        try data.write(to: destinationURL, options: .atomic)
    }

    private static func wavData(for samples: [Float], sampleRate: Int) -> Data {
        var pcmData = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1, min(1, sample))
            let intSample = Int16((clamped * Float(Int16.max)).rounded())
            var littleEndian = intSample.littleEndian
            pcmData.append(Data(bytes: &littleEndian, count: MemoryLayout<Int16>.size))
        }

        let dataChunkSize = UInt32(pcmData.count)
        let riffChunkSize = 36 + dataChunkSize
        let byteRate = UInt32(sampleRate * 2)
        let blockAlign = UInt16(2)
        let bitsPerSample = UInt16(16)

        var data = Data()
        data.append("RIFF".data(using: .ascii)!)
        data.append(riffChunkSize.littleEndianData)
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        data.append(UInt32(16).littleEndianData)
        data.append(UInt16(1).littleEndianData)
        data.append(UInt16(1).littleEndianData)
        data.append(UInt32(sampleRate).littleEndianData)
        data.append(byteRate.littleEndianData)
        data.append(blockAlign.littleEndianData)
        data.append(bitsPerSample.littleEndianData)
        data.append("data".data(using: .ascii)!)
        data.append(dataChunkSize.littleEndianData)
        data.append(pcmData)
        return data
    }
}

extension FixedWidthInteger {
    var littleEndianData: Data {
        var value = self.littleEndian
        return Data(bytes: &value, count: MemoryLayout<Self>.size)
    }
}
