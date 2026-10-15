// MeetingFinalSpeechValidation.swift
// Applies offline speech evidence without treating unavailable evidence as silence.

import Foundation

struct MeetingFinalSpeechEvidence: Equatable, Sendable {
    enum Verdict: Equatable, Sendable {
        case speech
        case silence
        case unknown
    }

    private(set) var evaluatedRangesBySource: [TranscriptAudioSource: [ASROfflineSpeechRange]] = [:]
    private(set) var speechRangesBySource: [TranscriptAudioSource: [ASROfflineSpeechRange]] = [:]

    mutating func record(
        source: TranscriptAudioSource,
        assetStartSeconds: TimeInterval,
        assetDurationSeconds: TimeInterval,
        speechRanges: [ASROfflineSpeechRange]
    ) {
        let assetEndSeconds = assetStartSeconds + max(0, assetDurationSeconds)
        guard assetEndSeconds > assetStartSeconds else { return }

        evaluatedRangesBySource[source, default: []].append(
            ASROfflineSpeechRange(
                startSeconds: assetStartSeconds,
                endSeconds: assetEndSeconds
            )
        )
        speechRangesBySource[source, default: []].append(contentsOf: speechRanges.compactMap { range in
            let start = max(assetStartSeconds, assetStartSeconds + range.startSeconds)
            let end = min(assetEndSeconds, assetStartSeconds + range.endSeconds)
            guard end > start else { return nil }
            return ASROfflineSpeechRange(startSeconds: start, endSeconds: end)
        })
    }

    func verdict(for segment: MeetingTranscriptSegment) -> Verdict {
        let source = segment.audioSource ?? Self.inferredAudioSource(for: segment.speaker)
        let segmentStart = max(0, segment.startSeconds)
        let segmentEnd = max(segment.endSeconds ?? segmentStart, segmentStart)
        guard segmentEnd > segmentStart else { return .unknown }

        let speechRanges = ranges(for: source, in: speechRangesBySource)
        if speechRanges.contains(where: {
            $0.intersects(
                startSeconds: segmentStart,
                endSeconds: segmentEnd,
                tolerance: 0.12
            )
        }) {
            return .speech
        }

        let evaluatedRanges = Self.merged(ranges(for: source, in: evaluatedRangesBySource))
        if evaluatedRanges.contains(where: {
            $0.startSeconds <= segmentStart + 0.01 && $0.endSeconds >= segmentEnd - 0.01
        }) {
            return .silence
        }
        return .unknown
    }

    private func ranges(
        for source: TranscriptAudioSource,
        in storage: [TranscriptAudioSource: [ASROfflineSpeechRange]]
    ) -> [ASROfflineSpeechRange] {
        if let exact = storage[source], !exact.isEmpty {
            return exact
        }
        return storage[.mixed] ?? []
    }

    private static func inferredAudioSource(for speaker: MeetingSpeaker) -> TranscriptAudioSource {
        switch speaker {
        case .me:
            return .microphone
        case .them:
            return .systemAudio
        }
    }

    private static func merged(_ ranges: [ASROfflineSpeechRange]) -> [ASROfflineSpeechRange] {
        let sorted = ranges.sorted { lhs, rhs in
            if lhs.startSeconds == rhs.startSeconds {
                return lhs.endSeconds < rhs.endSeconds
            }
            return lhs.startSeconds < rhs.startSeconds
        }
        guard var current = sorted.first else { return [] }

        var output: [ASROfflineSpeechRange] = []
        for range in sorted.dropFirst() {
            if range.startSeconds <= current.endSeconds + 0.01 {
                current = ASROfflineSpeechRange(
                    startSeconds: current.startSeconds,
                    endSeconds: max(current.endSeconds, range.endSeconds)
                )
            } else {
                output.append(current)
                current = range
            }
        }
        output.append(current)
        return output
    }
}

enum MeetingFinalSpeechValidator {
    static func vadPolicy(
        transcriptionEngine: TranscriptionEngine,
        mlxModelRepo: String
    ) -> MLXVADPolicy {
        guard transcriptionEngine == .mlxAudio else { return .standard }
        return MLXModelCatalog.capability(for: mlxModelRepo).vadPolicy
    }

    static func validatedSegments(
        _ segments: [MeetingTranscriptSegment],
        policy: MLXVADPolicy,
        evidence: MeetingFinalSpeechEvidence?
    ) -> [MeetingTranscriptSegment] {
        switch policy {
        case .modelManaged, .preserveTimeline:
            // Preserve the model timeline; do not drop silence ranges that would compress
            // timestamped speaker segments.
            return segments
        case .standard:
            guard let evidence else { return segments }
            return segments.filter { evidence.verdict(for: $0) != .silence }
        }
    }
}
