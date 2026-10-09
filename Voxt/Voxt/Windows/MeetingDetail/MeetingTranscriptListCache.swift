import Foundation

/// Main-actor value cache for full live snapshots. Snapshots can revise ANY earlier
/// segment; this is not an append-only cache. No tasks can publish stale revisions.
@MainActor
struct MeetingTranscriptListCache {
    private var previous: [MeetingTranscriptSegment] = []
    private(set) var ordinals: [String: Int] = [:]
    private var groupsByID: [String: MeetingDetailSpeakerGroup] = [:]
    private var inputsByID: [String: [MeetingTranscriptSegment]] = [:]
    private(set) var rebuiltGroupCount = 0
    private(set) var ordinalRebuildCount = 0

    mutating func updateOrdinals(for segments: [MeetingTranscriptSegment]) {
        // Text, translation and display-name changes do not affect ordinal ordering.
        let sameTopology = previous.count == segments.count && zip(previous, segments).allSatisfy { pair in
            pair.0.id == pair.1.id && pair.0.startSeconds == pair.1.startSeconds
                && pair.0.speakerIdentityKey == pair.1.speakerIdentityKey
        }
        if !sameTopology {
            let prefixUnchanged = !previous.isEmpty && segments.count > previous.count
                && zip(previous, segments.prefix(previous.count)).allSatisfy { pair in
                    pair.0.id == pair.1.id && pair.0.startSeconds == pair.1.startSeconds
                        && pair.0.speakerIdentityKey == pair.1.speakerIdentityKey
                }
            let additions = prefixUnchanged ? Array(segments.dropFirst(previous.count)) : []
            let previousEnd = previous.map(\.startSeconds).max() ?? 0
            if prefixUnchanged, additions.allSatisfy({ $0.startSeconds > previousEnd }) {
                let newOrdinals = MeetingTranscriptListSupport.speakerOrdinals(for: additions)
                for (key, _) in newOrdinals.sorted(by: { $0.value < $1.value }) where ordinals[key] == nil {
                    ordinals[key] = ordinals.count + 1
                }
            } else {
                // Reorder, removal, timing correction or equal-time UUID ordering
                // can change earlier ordinals; rebuild rather than assuming append.
                ordinals = MeetingTranscriptListSupport.speakerOrdinals(for: segments)
                ordinalRebuildCount += 1
            }
        }
        previous = segments
    }

    mutating func groups(
        for displayedSegments: [MeetingTranscriptSegment],
        title: (MeetingTranscriptSegment) -> String
    ) -> [MeetingDetailSpeakerGroup] {
        let incoming = Dictionary(grouping: displayedSegments, by: \.speakerIdentityKey)
        var next: [String: MeetingDetailSpeakerGroup] = [:]
        rebuiltGroupCount = 0
        for (key, segments) in incoming {
            // Compare in incoming order first: unchanged speakers avoid sorting,
            // recounting words and reallocating their presentation data.
            if let old = groupsByID[key],
               inputsByID[key] == segments,
               let representative = old.segments.first,
               old.title == title(representative) {
                next[key] = old
            } else {
                let built = MeetingTranscriptListSupport.speakerGroups(from: segments, titleForSegment: title)
                next[key] = built.first
                rebuiltGroupCount += 1
            }
        }
        groupsByID = next
        inputsByID = incoming
        return next.values.sorted {
            let left = $0.segments.first?.startSeconds ?? 0
            let right = $1.segments.first?.startSeconds ?? 0
            return left == right ? $0.id < $1.id : left < right
        }
    }
}
