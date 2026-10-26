// Shared reliability rules for live-ended and batch transcript segments.

import Foundation

extension MLXTranscriber {
    /// Maps a live ended payload using the same reliability rules as batch.
    nonisolated static func structuredSegmentsForLiveEnded(
        output: ASRTranscriptionResult,
        modelFamily: MLXModelFamily,
        timingGranularity: MLXASRTimingGranularity
    ) -> [MLXStructuredTranscriptSegment] {
        if modelFamily == .mossTranscribeDiarize {
            return mossStructuredSegments(from: output.segments)
        }
        return reliableStructuredSegments(
            from: output.segments,
            timingGranularity: timingGranularity
        )
    }

    nonisolated static func reliableStructuredSegments(
        from rawSegments: [ASRTranscriptSegment],
        timingGranularity: MLXASRTimingGranularity
    ) -> [MLXStructuredTranscriptSegment] {
        guard timingGranularity.providesReliableSegments else { return [] }

        return rawSegments.compactMap { segment in
            guard let start = segment.startSeconds,
                  let end = segment.endSeconds,
                  start.isFinite,
                  end.isFinite,
                  end > start
            else {
                return nil
            }

            let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            return MLXStructuredTranscriptSegment(
                startSeconds: start,
                endSeconds: end,
                text: text
            )
        }
    }

    nonisolated static func mossStructuredSegments(
        from rawSegments: [ASRTranscriptSegment]
    ) -> [MLXStructuredTranscriptSegment] {
        rawSegments.compactMap { segment in
            guard let start = segment.startSeconds,
                  let end = segment.endSeconds,
                  start.isFinite,
                  end.isFinite,
                  end >= start,
                  let speakerID = segment.speakerID?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !speakerID.isEmpty
            else {
                return nil
            }

            // Structured MOSS segments carry timing and speaker metadata separately,
            // so their text should contain only user-visible speech. Reuse the MOSS
            // plain-text renderer to remove speaker protocol and acoustic annotations
            // such as `[sniff]` before the segment can reach meeting storage/export.
            let text = MossASRTranscriptRendering.renderedText(segment.text, outputMode: .plainText)
            guard !text.isEmpty else { return nil }
            return MLXStructuredTranscriptSegment(
                startSeconds: start,
                endSeconds: end,
                speakerID: speakerID,
                text: text
            )
        }
    }
}
