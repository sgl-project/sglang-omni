import Foundation

/// Value-only history edits and selection rules; persistence remains with the store.
enum HistoryEntryPolicy {
    static func mergingCorrectedTerms(
        for existingEntry: TranscriptionHistoryEntry,
        incoming: [String]
    ) -> TranscriptionHistoryEntry? {
        let merged = mergeUniqueTerms(
            existing: existingEntry.dictionaryCorrectedTerms,
            incoming: incoming
        )
        guard merged != existingEntry.dictionaryCorrectedTerms else { return nil }
        return existingEntry.updatingDictionaryCorrectedTerms(merged)
    }

    static func correctionResult(
        for existingEntry: TranscriptionHistoryEntry,
        updatedText: String,
        correctedTerms: [String],
        correctionSnapshots: [DictionaryCorrectionSnapshot],
        replacingSnapshots: Bool
    ) -> TranscriptionHistoryEntry? {
        let trimmedUpdatedText = updatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let mergedTerms = mergeUniqueTerms(
            existing: existingEntry.dictionaryCorrectedTerms,
            incoming: correctedTerms
        )
        let resolvedSnapshots = replacingSnapshots ? correctionSnapshots : mergeCorrectionSnapshots(
            existing: existingEntry.dictionaryCorrectionSnapshots,
            incoming: correctionSnapshots
        )

        guard !trimmedUpdatedText.isEmpty else { return nil }
        let didTextChange = trimmedUpdatedText != existingEntry.text
        let didTermsChange = mergedTerms != existingEntry.dictionaryCorrectedTerms
        let didSnapshotsChange = resolvedSnapshots != existingEntry.dictionaryCorrectionSnapshots
        guard didTextChange || didTermsChange || didSnapshotsChange else { return nil }

        return existingEntry.updatingDictionaryCorrectionResult(
            text: trimmedUpdatedText,
            dictionaryCorrectedTerms: mergedTerms,
            dictionaryCorrectionSnapshots: resolvedSnapshots
        )
    }

    static func transcriptSegmentsUpdate(
        for existingEntry: TranscriptionHistoryEntry,
        segments: [TranscriptSegment],
        allowsEmptyText: Bool
    ) -> TranscriptionHistoryEntry? {
        let text = TranscriptFormatter.joinedText(for: segments)
        let canStoreEmptyMeetingTranscript = allowsEmptyText
            && existingEntry.meetingCaptureMode != nil
            && existingEntry.audioRelativePath != nil
        guard canStoreEmptyMeetingTranscript
            || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        return existingEntry.updatingTranscriptSegments(segments, text: text)
    }

    static func uniqueSortedEntries(_ entries: [TranscriptionHistoryEntry]) -> [TranscriptionHistoryEntry] {
        var seen = Set<UUID>()
        return entries
            .sorted { $0.createdAt > $1.createdAt }
            .filter { seen.insert($0.id).inserted }
    }

    struct RetentionPlan {
        let cutoff: Date?
        let kinds: Set<TranscriptionHistoryKind>
        let orderedKinds: [TranscriptionHistoryKind]

        init(
            period: HistoryRetentionPeriod,
            referenceDate: Date,
            calendar: Calendar
        ) {
            if let days = period.days {
                cutoff = calendar.date(byAdding: .day, value: -days, to: referenceDate) ?? referenceDate
            } else {
                cutoff = nil
            }
            kinds = [.normal, .translation, .rewrite]
            orderedKinds = kinds.sorted { $0.rawValue < $1.rawValue }
        }
    }

    private static func mergeCorrectionSnapshots(
        existing: [DictionaryCorrectionSnapshot],
        incoming: [DictionaryCorrectionSnapshot]
    ) -> [DictionaryCorrectionSnapshot] {
        guard !incoming.isEmpty else { return existing }
        var merged = existing
        var seen = Set(existing)
        for snapshot in incoming where seen.insert(snapshot).inserted {
            merged.append(snapshot)
        }
        return merged.sorted { lhs, rhs in
            if lhs.finalLocation == rhs.finalLocation {
                return lhs.finalLength < rhs.finalLength
            }
            return lhs.finalLocation < rhs.finalLocation
        }
    }

    private static func mergeUniqueTerms(existing: [String], incoming: [String]) -> [String] {
        var merged = existing
        let existingNormalized = Set(existing.map(DictionaryStore.normalizeTerm))
        var appended = Set<String>()

        for term in incoming {
            let normalized = DictionaryStore.normalizeTerm(term)
            guard !normalized.isEmpty else { continue }
            guard !existingNormalized.contains(normalized) else { continue }
            guard appended.insert(normalized).inserted else { continue }
            merged.append(term)
        }

        return merged
    }
}
