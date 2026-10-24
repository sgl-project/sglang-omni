import Foundation

/// Transcript value transformations; persistence and translation execution stay in the view model.
@MainActor
enum MeetingDetailTranscriptProjection {
    static func renamingSpeaker(
        in segments: [MeetingTranscriptSegment],
        identityKey: String,
        displayName: String
    ) -> [MeetingTranscriptSegment] {
        let trimmedDisplayName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedDisplayName = trimmedDisplayName.isEmpty ? nil : trimmedDisplayName

        return segments.map { segment in
            guard speakerRenameIdentityMatches(segment, identityKey: identityKey) else { return segment }
            return segment.updatingSpeakerDisplayName(normalizedDisplayName)
        }
    }

    static func liveSegments(
        from incomingSegments: [MeetingTranscriptSegment],
        existingSegments: [MeetingTranscriptSegment],
        failedTranslationRevisions: [UUID: String],
        targetLanguage: TranslationTargetLanguage
    ) -> [MeetingTranscriptSegment] {
        mergeSegmentsPreservingTranslationState(
            incomingSegments,
            existingSegments: existingSegments
        ).map { segment in
            guard failedTranslationRevisions[segment.id] == translationRevision(
                for: segment,
                targetLanguage: targetLanguage
            ) else { return segment }
            return segment.updatingTranslation(
                translatedText: segment.translatedText,
                isTranslationPending: false
            )
        }
    }

    static func shouldTranslate(
        segment: MeetingTranscriptSegment,
        targetLanguage: TranslationTargetLanguage,
        failedTranslationRevisions: [UUID: String]
    ) -> Bool {
        guard segment.speaker == .them else { return false }
        guard failedTranslationRevisions[segment.id] != translationRevision(
            for: segment,
            targetLanguage: targetLanguage
        ) else { return false }
        let translatedText = segment.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return translatedText.isEmpty || segment.isTranslationPending
    }

    static func translationRevision(
        for segment: MeetingTranscriptSegment,
        targetLanguage: TranslationTargetLanguage
    ) -> String {
        segment.text.trimmingCharacters(in: .whitespacesAndNewlines) + "\u{0}" + targetLanguage.rawValue
    }

    static func clearingPendingTranslation(
        in segments: [MeetingTranscriptSegment]
    ) -> [MeetingTranscriptSegment] {
        segments.map { segment in
            guard segment.isTranslationPending else { return segment }
            return segment.updatingTranslation(
                translatedText: segment.translatedText,
                isTranslationPending: false
            )
        }
    }

    static func segmentsContainTranslations(_ segments: [MeetingTranscriptSegment]) -> Bool {
        segments.contains { segment in
            !(segment.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        }
    }

    private static func speakerRenameIdentityMatches(_ segment: MeetingTranscriptSegment, identityKey: String) -> Bool {
        if segment.speakerIdentityKey == identityKey {
            return true
        }

        let displayPrefix = "display:"
        guard identityKey.hasPrefix(displayPrefix) else { return false }
        let expectedDisplayName = String(identityKey.dropFirst(displayPrefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !expectedDisplayName.isEmpty else { return false }

        let currentDisplayName = segment.speakerDisplayName?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return currentDisplayName == expectedDisplayName
    }

    private static func mergeSegmentsPreservingTranslationState(
        _ incomingSegments: [MeetingTranscriptSegment],
        existingSegments: [MeetingTranscriptSegment]
    ) -> [MeetingTranscriptSegment] {
        let existingByID = Dictionary(uniqueKeysWithValues: existingSegments.map { ($0.id, $0) })
        return incomingSegments.map { incoming in
            guard let existing = existingByID[incoming.id] else { return incoming }

            let existingTranslatedText = existing.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines)
            let incomingTranslatedText = incoming.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedTranslatedText = incomingTranslatedText?.isEmpty == false
                ? incomingTranslatedText
                : (existingTranslatedText?.isEmpty == false ? existingTranslatedText : nil)
            let textChanged =
                existing.text.trimmingCharacters(in: .whitespacesAndNewlines) !=
                incoming.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let shouldRefreshTranslation = (existingTranslatedText?.isEmpty == false) && textChanged

            return MeetingTranscriptSegment(
                id: incoming.id,
                speaker: incoming.speaker,
                speakerID: incoming.speakerID,
                speakerDisplayName: incoming.speakerDisplayName,
                audioSource: incoming.audioSource,
                speakerConfidence: incoming.speakerConfidence,
                startSeconds: incoming.startSeconds,
                endSeconds: incoming.endSeconds,
                text: incoming.text,
                translatedText: resolvedTranslatedText,
                isTranslationPending: incoming.isTranslationPending || existing.isTranslationPending || shouldRefreshTranslation,
                preventsAdjacentMerge: incoming.preventsAdjacentMerge,
                isHighlighted: existing.isHighlighted || incoming.isHighlighted
            )
        }
    }
}
