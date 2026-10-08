// SenseVoiceMetadata.swift
// Provides Sense Voice Metadata for shared utilities.

import Foundation
import MLXAudioSTT

struct SenseVoiceSegmentMetadata: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    let text: String
    let language: String?
    let emotion: String?
    let event: String?

    nonisolated init(
        id: UUID = UUID(),
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        text: String,
        language: String?,
        emotion: String?,
        event: String?
    ) {
        self.id = id
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.text = text
        self.language = language
        self.emotion = emotion
        self.event = event
    }
}

struct SenseVoiceTranscriptMetadata: Codable, Hashable, Sendable {
    let language: String?
    let emotion: String?
    let event: String?
    let usedVADSegmentation: Bool
    let segments: [SenseVoiceSegmentMetadata]

    func formattedDebugSummary(appendingTranscript transcript: String) -> String {
        var lines: [String] = []
        lines.append("Transcript:")
        lines.append(transcript.trimmingCharacters(in: .whitespacesAndNewlines))
        lines.append("")
        lines.append("SenseVoice Metadata:")
        lines.append("Language: \(language ?? "unknown")")
        lines.append("Emotion: \(emotion ?? "unknown")")
        lines.append("Event: \(event ?? "unknown")")
        lines.append("VAD Segmentation: \(usedVADSegmentation ? "on" : "off")")
        if !segments.isEmpty {
            lines.append("")
            lines.append("Segments:")
            for segment in segments {
                let start = TranscriptFormatter.timestampString(for: segment.startSeconds)
                let end = TranscriptFormatter.timestampString(for: segment.endSeconds)
                let labels = [segment.language, segment.emotion, segment.event]
                    .compactMap { value -> String? in
                        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                        return trimmed.isEmpty ? nil : trimmed
                    }
                    .joined(separator: " / ")
                if labels.isEmpty {
                    lines.append("[\(start)-\(end)] \(segment.text)")
                } else {
                    lines.append("[\(start)-\(end)] \(labels): \(segment.text)")
                }
            }
        }
        return lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

extension SenseVoiceTranscriptMetadata {
    nonisolated static func fromOutput(
        _ output: STTOutput,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        usedVADSegmentation: Bool
    ) -> SenseVoiceTranscriptMetadata? {
        let segments = segments(
            from: output,
            startSeconds: startSeconds,
            endSeconds: endSeconds
        )
        return aggregated(
            segments: segments,
            usedVADSegmentation: usedVADSegmentation
        )
    }

    nonisolated static func aggregated(
        segments: [SenseVoiceSegmentMetadata],
        usedVADSegmentation: Bool
    ) -> SenseVoiceTranscriptMetadata? {
        guard !segments.isEmpty else { return nil }
        return SenseVoiceTranscriptMetadata(
            language: dominantMetadataValue(segments.compactMap(\.language)),
            emotion: dominantMetadataValue(segments.compactMap(\.emotion)),
            event: dominantMetadataValue(segments.compactMap(\.event)),
            usedVADSegmentation: usedVADSegmentation,
            segments: segments
        )
    }

    nonisolated static func mergeSequentialSegments(
        base: [SenseVoiceSegmentMetadata],
        next: [SenseVoiceSegmentMetadata]
    ) -> [SenseVoiceSegmentMetadata] {
        let left = base.filter { !normalizedText($0.text).isEmpty }
        let right = next.filter { !normalizedText($0.text).isEmpty }
        guard !left.isEmpty else { return right }
        guard !right.isEmpty else { return left }

        var merged = left
        var pending = right[...]

        while let last = merged.last,
              let first = pending.first,
              shouldMergeBoundarySegments(last, first) {
            let mergeResult = MLXTranscriptionPlanning.sequentialTranscriptMergeResult(
                base: last.text,
                next: first.text
            )
            let combined = SenseVoiceSegmentMetadata(
                id: last.id,
                startSeconds: min(last.startSeconds, first.startSeconds),
                endSeconds: max(last.endSeconds, first.endSeconds),
                text: mergeResult.text,
                language: mergedMetadataValue(primary: last.language, secondary: first.language),
                emotion: mergedMetadataValue(primary: last.emotion, secondary: first.emotion),
                event: mergedMetadataValue(primary: last.event, secondary: first.event)
            )
            merged[merged.count - 1] = combined
            pending = pending.dropFirst()
        }

        merged.append(contentsOf: pending)
        return merged
    }

    private nonisolated static func segments(
        from output: STTOutput,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval
    ) -> [SenseVoiceSegmentMetadata] {
        let rawSegments = output.segments ?? []
        guard !rawSegments.isEmpty else {
            return [
                SenseVoiceSegmentMetadata(
                    startSeconds: startSeconds,
                    endSeconds: max(startSeconds, endSeconds),
                    text: normalizedText(output.text),
                    language: normalizedMetadataValue(output.language),
                    emotion: nil,
                    event: nil
                )
            ]
        }

        return rawSegments.enumerated().map { index, segment in
            let fallbackRange = fallbackSegmentRange(
                index: index,
                totalCount: rawSegments.count,
                startSeconds: startSeconds,
                endSeconds: endSeconds
            )
            let segmentStart = segment.startTime ?? fallbackRange.lowerBound
            let segmentEnd = segment.endTime ?? fallbackRange.upperBound
            return SenseVoiceSegmentMetadata(
                startSeconds: min(segmentStart, segmentEnd),
                endSeconds: max(segmentStart, segmentEnd),
                text: normalizedText(segment.text),
                language: normalizedMetadataValue(segment.language ?? output.language),
                emotion: normalizedMetadataValue(segment.emotion),
                event: normalizedMetadataValue(segment.event)
            )
        }
    }

    private nonisolated static func fallbackSegmentRange(
        index: Int,
        totalCount: Int,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval
    ) -> ClosedRange<TimeInterval> {
        let lowerBound = min(startSeconds, endSeconds)
        let upperBound = max(startSeconds, endSeconds)
        guard totalCount > 1, upperBound > lowerBound else {
            return lowerBound ... upperBound
        }

        let segmentDuration = (upperBound - lowerBound) / Double(totalCount)
        let segmentStart = lowerBound + (segmentDuration * Double(index))
        let segmentEnd = index == totalCount - 1
            ? upperBound
            : min(upperBound, segmentStart + segmentDuration)
        return segmentStart ... segmentEnd
    }

    private nonisolated static func dominantMetadataValue(_ values: [String]) -> String? {
        var counts: [String: Int] = [:]
        var order: [String] = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if counts[trimmed] == nil {
                order.append(trimmed)
            }
            counts[trimmed, default: 0] += 1
        }
        guard !counts.isEmpty else { return nil }

        return order.max { lhs, rhs in
            let leftCount = counts[lhs, default: 0]
            let rightCount = counts[rhs, default: 0]
            if leftCount == rightCount {
                return (order.firstIndex(of: lhs) ?? 0) > (order.firstIndex(of: rhs) ?? 0)
            }
            return leftCount < rightCount
        }
    }

    private nonisolated static func normalizedMetadataValue(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    private nonisolated static func mergedMetadataValue(primary: String?, secondary: String?) -> String? {
        normalizedMetadataValue(primary) ?? normalizedMetadataValue(secondary)
    }

    private nonisolated static func normalizedText(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func shouldMergeBoundarySegments(
        _ left: SenseVoiceSegmentMetadata,
        _ right: SenseVoiceSegmentMetadata
    ) -> Bool {
        let mergeResult = MLXTranscriptionPlanning.sequentialTranscriptMergeResult(
            base: left.text,
            next: right.text
        )
        return mergeResult.overlapCount > 0
    }

}
