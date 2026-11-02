// ASRVoiceActivitySampleFilter.swift
// Provides reusable VAD-based audio sample filtering for ASR upload preparation.

import Foundation

nonisolated struct ASRVoiceActivitySampleFilterResult: Equatable, Sendable {
    let samples: [Float]
    let observedFrames: Bool
    let observedSpeech: Bool
    let speechSegments: [ASRVoiceActivitySegment]
    let originalDurationSeconds: TimeInterval
    let filteredDurationSeconds: TimeInterval

    var reductionRatio: Double {
        guard originalDurationSeconds > 0 else { return 0 }
        return max(0, min(1, 1 - (filteredDurationSeconds / originalDurationSeconds)))
    }
}

nonisolated enum ASRVoiceActivitySampleFilter {
    nonisolated static let defaultMaximumJoinGapSeconds: TimeInterval = 0.25

    nonisolated static func filter(
        samples: [Float],
        sampleRate: Double,
        decisions: [ASRVoiceActivityFrameDecision],
        configuration: ASRVoiceActivityConfiguration,
        maximumJoinGapSeconds: TimeInterval = defaultMaximumJoinGapSeconds
    ) -> ASRVoiceActivitySampleFilterResult {
        let safeRate = sampleRate.isFinite ? max(0, sampleRate) : 0
        let originalDuration = safeRate > 0 ? Double(samples.count) / safeRate : 0
        guard !samples.isEmpty, safeRate > 0 else {
            return ASRVoiceActivitySampleFilterResult(
                samples: [],
                observedFrames: false,
                observedSpeech: false,
                speechSegments: [],
                originalDurationSeconds: originalDuration,
                filteredDurationSeconds: 0
            )
        }

        var segmenter = ASRVoiceActivitySegmenter(configuration: configuration)
        var segments: [ASRVoiceActivitySegment] = []
        var observedFrames = false
        var observedSpeech = false

        for decision in decisions where decision.endSeconds > decision.startSeconds {
            observedFrames = true
            let result = segmenter.appendWithResolvedSpeechState(decision)
            if result.isSpeech {
                observedSpeech = true
            }
            for event in result.events {
                if let segment = acceptedSegment(from: event) {
                    segments.append(segment)
                }
            }
        }

        if let event = segmenter.finish(at: originalDuration),
           let segment = acceptedSegment(from: event) {
            segments.append(segment)
        }

        let mergedSegments = mergeSegments(
            segments,
            maximumJoinGapSeconds: maximumJoinGapSeconds,
            originalDurationSeconds: originalDuration
        )
        let filtered = samplesForSegments(
            mergedSegments,
            from: samples,
            sampleRate: safeRate
        )
        let filteredDuration = safeRate > 0 ? Double(filtered.count) / safeRate : 0

        return ASRVoiceActivitySampleFilterResult(
            samples: filtered,
            observedFrames: observedFrames,
            observedSpeech: observedSpeech,
            speechSegments: mergedSegments,
            originalDurationSeconds: originalDuration,
            filteredDurationSeconds: filteredDuration
        )
    }

    private nonisolated static func acceptedSegment(
        from event: ASRVoiceActivityEvent
    ) -> ASRVoiceActivitySegment? {
        switch event {
        case .speechEnded(let segment), .speechForced(let segment):
            return segment
        case .speechRejected, .speechStarted:
            return nil
        }
    }

    private nonisolated static func mergeSegments(
        _ segments: [ASRVoiceActivitySegment],
        maximumJoinGapSeconds: TimeInterval,
        originalDurationSeconds: TimeInterval
    ) -> [ASRVoiceActivitySegment] {
        let sorted = segments
            .filter { $0.endSeconds > $0.startSeconds }
            .sorted { lhs, rhs in
                if lhs.startSeconds == rhs.startSeconds {
                    return lhs.endSeconds < rhs.endSeconds
                }
                return lhs.startSeconds < rhs.startSeconds
            }
        guard var current = sorted.first else { return [] }

        var merged: [ASRVoiceActivitySegment] = []
        for segment in sorted.dropFirst() {
            let gap = segment.startSeconds - current.endSeconds
            if gap <= max(0, maximumJoinGapSeconds) {
                current = ASRVoiceActivitySegment(
                    startSeconds: current.startSeconds,
                    endSeconds: min(max(current.endSeconds, segment.endSeconds), originalDurationSeconds),
                    speechSeconds: current.speechSeconds + segment.speechSeconds,
                    frameCount: current.frameCount + segment.frameCount
                )
            } else {
                merged.append(clamped(current, originalDurationSeconds: originalDurationSeconds))
                current = segment
            }
        }
        merged.append(clamped(current, originalDurationSeconds: originalDurationSeconds))
        return merged
    }

    private nonisolated static func clamped(
        _ segment: ASRVoiceActivitySegment,
        originalDurationSeconds: TimeInterval
    ) -> ASRVoiceActivitySegment {
        ASRVoiceActivitySegment(
            startSeconds: min(max(0, segment.startSeconds), originalDurationSeconds),
            endSeconds: min(max(0, segment.endSeconds), originalDurationSeconds),
            speechSeconds: segment.speechSeconds,
            frameCount: segment.frameCount
        )
    }

    private nonisolated static func samplesForSegments(
        _ segments: [ASRVoiceActivitySegment],
        from samples: [Float],
        sampleRate: Double
    ) -> [Float] {
        guard sampleRate > 0, !samples.isEmpty else { return [] }
        var output: [Float] = []
        for segment in segments {
            let startIndex = max(0, min(samples.count, Int((segment.startSeconds * sampleRate).rounded(.down))))
            let endIndex = max(startIndex, min(samples.count, Int((segment.endSeconds * sampleRate).rounded(.up))))
            guard startIndex < endIndex else { continue }
            output.append(contentsOf: samples[startIndex..<endIndex])
        }
        return output
    }
}
