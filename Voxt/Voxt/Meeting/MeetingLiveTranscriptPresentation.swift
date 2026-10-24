import Foundation

/// Live presentation policy; session tokens and translation tasks stay with the coordinator.
enum MeetingLiveTranscriptPresentation {
    static func finalizedSegments(from segments: [MeetingTranscriptSegment]) -> [MeetingTranscriptSegment] {
        segments.map { segment in
            guard segment.isTranslationPending else { return segment }
            return segment.updatingTranslation(
                translatedText: segment.translatedText,
                isTranslationPending: false
            )
        }
    }

    static func events(for event: MeetingTranscriptEvent, captureMode: MeetingCaptureMode, segments: [MeetingTranscriptSegment]) -> [MeetingTranscriptEvent] {
        let event = meetingDisplayNormalizedEvent(event, captureMode: captureMode)
        guard case .final(let segment) = event else {
            return [event]
        }
        if let mergedEvent = liveOverlayMergedShortFinalEvent(for: segment, segments: segments) {
            return [mergedEvent]
        }
        let readableSegments = MeetingTranscriptPostProcessor.process(
            [segment],
            options: .liveOverlay
        )
        guard readableSegments.count > 1 else {
            return [event]
        }
        return readableSegments.map(MeetingTranscriptEvent.final)
    }

    private static func meetingDisplayNormalizedEvent(_ event: MeetingTranscriptEvent, captureMode: MeetingCaptureMode) -> MeetingTranscriptEvent {
        guard captureMode == .meeting else { return event }

        func normalizedSegment(_ segment: MeetingTranscriptSegment) -> MeetingTranscriptSegment {
            return segment.updatingSpeakerAnalysis(
                speaker: segment.speaker,
                speakerID: nil,
                speakerDisplayName: nil,
                audioSource: segment.audioSource,
                speakerConfidence: nil
            )
        }

        switch event {
        case .partial(let segment):
            return .partial(normalizedSegment(segment))
        case .final(let segment):
            return .final(normalizedSegment(segment))
        case .failed, .finished:
            return event
        }
    }

    private static func liveOverlayMergedShortFinalEvent(for segment: MeetingTranscriptSegment, segments: [MeetingTranscriptSegment]) -> MeetingTranscriptEvent? {
        let options = MeetingTranscriptPostProcessor.Options.liveOverlay
        guard let previous = segments.last,
              previous.id != segment.id,
              previous.speakerIdentityKey == segment.speakerIdentityKey
        else {
            return nil
        }

        let previousEnd = previous.endSeconds ?? previous.startSeconds
        let segmentEnd = segment.endSeconds ?? segment.startSeconds
        let gap = segment.startSeconds - previousEnd
        guard segment.startSeconds >= previous.startSeconds,
              gap >= -0.05,
              gap <= options.maxSameSpeakerMergeGapSeconds
        else {
            return nil
        }

        let previousText = MeetingTranscriptTextPostProcessor.normalizedFinalText(previous.text)
        let segmentText = MeetingTranscriptTextPostProcessor.normalizedFinalText(segment.text)
        guard !previousText.isEmpty, !segmentText.isEmpty else { return nil }

        let mergedText = MeetingTranscriptTextPostProcessor.mergedTextRemovingOverlap(previousText, segmentText)
        let shouldMergeShortFragment =
            previousText.count < options.minSegmentTextCharacters ||
            segmentText.count < options.minSegmentTextCharacters
        guard shouldMergeShortFragment,
              mergedText.count <= options.maxSegmentTextCharacters,
              max(previousEnd, segmentEnd) - previous.startSeconds <= options.maxMergedDurationSeconds
        else {
            return nil
        }

        let merged = MeetingTranscriptSegment(
            id: previous.id,
            speaker: previous.speaker,
            speakerID: previous.speakerID ?? segment.speakerID,
            speakerDisplayName: previous.speakerDisplayName ?? segment.speakerDisplayName,
            audioSource: previous.audioSource ?? segment.audioSource,
            speakerConfidence: [previous.speakerConfidence, segment.speakerConfidence]
                .compactMap { $0 }
                .max(),
            startSeconds: previous.startSeconds,
            endSeconds: max(previousEnd, segmentEnd),
            text: mergedText,
            translatedText: nil,
            isTranslationPending: false,
            preventsAdjacentMerge: true
        )
        return .final(merged)
    }
}
