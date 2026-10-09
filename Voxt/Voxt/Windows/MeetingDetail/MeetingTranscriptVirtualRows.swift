// MeetingTranscriptVirtualRows.swift
// Flattened virtual-row models and pure list builders for meeting detail transcript.

import Foundation

struct MeetingTranscriptSegmentPresentation: Identifiable, Equatable {
    let segment: MeetingTranscriptSegment
    let speakerTitle: String
    let isActive: Bool
    let showsTranslation: Bool
    let isSearchMatch: Bool

    var id: UUID { segment.id }
}

enum MeetingTranscriptVirtualRow: Identifiable, Equatable {
    case speakerHeader(id: String, title: String, count: Int)
    case segment(MeetingTranscriptSegmentPresentation)

    var id: String {
        switch self {
        case .speakerHeader(let id, _, _):
            return "header:\(id)"
        case .segment(let presentation):
            return presentation.id.uuidString
        }
    }

    var segmentID: UUID? {
        switch self {
        case .speakerHeader:
            return nil
        case .segment(let presentation):
            return presentation.id
        }
    }
}

enum MeetingTranscriptScrollAnchor: Equatable {
    case top
    case center
    case bottom
}

struct MeetingTranscriptScrollRequest: Equatable {
    let rowID: String
    let anchor: MeetingTranscriptScrollAnchor
    let generation: UInt64
}

struct MeetingDetailSpeakerGroup: Identifiable, Equatable {
    let id: String
    let title: String
    let speaker: MeetingSpeaker
    let segments: [MeetingTranscriptSegment]
    let wordCount: Int
}

enum MeetingTranscriptListSupport {
    static func normalizedSearchQuery(_ query: String) -> String {
        query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func segmentMatchesSearch(
        _ segment: MeetingTranscriptSegment,
        query: String,
        speakerTitle: String
    ) -> Bool {
        let normalized = normalizedSearchQuery(query)
        guard !normalized.isEmpty else { return true }
        return segment.text.localizedCaseInsensitiveContains(normalized)
            || (segment.translatedText?.localizedCaseInsensitiveContains(normalized) ?? false)
            || speakerTitle.localizedCaseInsensitiveContains(normalized)
            || MeetingTranscriptFormatter.timestampString(for: segment.startSeconds)
                .localizedCaseInsensitiveContains(normalized)
    }

    static func displayedSegments(
        from segments: [MeetingTranscriptSegment],
        searchQuery: String,
        speakerTitle: (MeetingTranscriptSegment) -> String
    ) -> [MeetingTranscriptSegment] {
        let normalized = normalizedSearchQuery(searchQuery)
        guard !normalized.isEmpty else { return segments }
        return segments.filter { segment in
            segmentMatchesSearch(segment, query: normalized, speakerTitle: speakerTitle(segment))
        }
    }

    static func speakerOrdinals(for segments: [MeetingTranscriptSegment]) -> [String: Int] {
        var ordinals: [String: Int] = [:]
        let sortedSegments = segments.sorted { lhs, rhs in
            if lhs.startSeconds == rhs.startSeconds {
                return lhs.id.uuidString < rhs.id.uuidString
            }
            return lhs.startSeconds < rhs.startSeconds
        }
        for segment in sortedSegments {
            let key = segment.speakerIdentityKey
            guard ordinals[key] == nil else { continue }
            ordinals[key] = ordinals.count + 1
        }
        return ordinals
    }

    static func speakerGroups(
        from segments: [MeetingTranscriptSegment],
        titleForSegment: (MeetingTranscriptSegment) -> String
    ) -> [MeetingDetailSpeakerGroup] {
        let grouped = Dictionary(grouping: segments, by: \.speakerIdentityKey)
        return grouped.map { key, groupSegments in
            let sortedSegments = groupSegments.sorted { $0.startSeconds < $1.startSeconds }
            let representative = sortedSegments.first
            let wordCount = sortedSegments.reduce(0) { partial, segment in
                partial + segment.text.split(whereSeparator: \.isWhitespace).count
            }
            return MeetingDetailSpeakerGroup(
                id: key,
                title: representative.map(titleForSegment) ?? "",
                speaker: representative?.speaker ?? .them,
                segments: sortedSegments,
                wordCount: wordCount
            )
        }
        .sorted { lhs, rhs in
            guard let lhsStart = lhs.segments.first?.startSeconds,
                  let rhsStart = rhs.segments.first?.startSeconds
            else {
                return lhs.title < rhs.title
            }
            return lhsStart < rhsStart
        }
    }

    static func timelineRows(
        from segments: [MeetingTranscriptSegment],
        activeSegmentID: UUID?,
        showsTranslation: Bool,
        searchQuery: String,
        speakerTitle: (MeetingTranscriptSegment) -> String
    ) -> [MeetingTranscriptVirtualRow] {
        let normalized = normalizedSearchQuery(searchQuery)
        return segments.map { segment in
            .segment(
                MeetingTranscriptSegmentPresentation(
                    segment: segment,
                    speakerTitle: speakerTitle(segment),
                    isActive: activeSegmentID == segment.id,
                    showsTranslation: showsTranslation,
                    isSearchMatch: !normalized.isEmpty
                        && segmentMatchesSearch(segment, query: normalized, speakerTitle: speakerTitle(segment))
                )
            )
        }
    }

    static func speakerMarkRows(
        from groups: [MeetingDetailSpeakerGroup],
        activeSegmentID: UUID?,
        showsTranslation: Bool,
        searchQuery: String
    ) -> [MeetingTranscriptVirtualRow] {
        let normalized = normalizedSearchQuery(searchQuery)
        var rows: [MeetingTranscriptVirtualRow] = []
        rows.reserveCapacity(groups.reduce(0) { $0 + $1.segments.count + 1 })
        for group in groups where !group.segments.isEmpty {
            rows.append(.speakerHeader(id: group.id, title: group.title, count: group.segments.count))
            for segment in group.segments {
                rows.append(
                    .segment(
                        MeetingTranscriptSegmentPresentation(
                            segment: segment,
                            speakerTitle: group.title,
                            isActive: activeSegmentID == segment.id,
                            showsTranslation: showsTranslation,
                            isSearchMatch: !normalized.isEmpty
                                && segmentMatchesSearch(segment, query: normalized, speakerTitle: group.title)
                        )
                    )
                )
            }
        }
        return rows
    }

    /// Returns indexes that need content reload when identity sequence is unchanged.
    /// Returns `nil` when the identity structure changed and a structural update is required.
    static func contentChangedIndexes(
        from oldRows: [MeetingTranscriptVirtualRow],
        to newRows: [MeetingTranscriptVirtualRow]
    ) -> IndexSet? {
        guard oldRows.count == newRows.count else { return nil }
        var changed = IndexSet()
        for index in oldRows.indices {
            let oldRow = oldRows[index]
            let newRow = newRows[index]
            guard oldRow.id == newRow.id else { return nil }
            if oldRow != newRow {
                changed.insert(index)
            }
        }
        return changed
    }

    static func isSuffixAppend(
        from oldRows: [MeetingTranscriptVirtualRow],
        to newRows: [MeetingTranscriptVirtualRow]
    ) -> Bool {
        guard newRows.count > oldRows.count else { return false }
        for index in oldRows.indices {
            guard oldRows[index].id == newRows[index].id else { return false }
        }
        return true
    }
}
