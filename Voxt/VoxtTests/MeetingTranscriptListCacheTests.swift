import XCTest
@testable import Voxt

@MainActor
final class MeetingTranscriptListCacheTests: XCTestCase {
    private func segment(_ speaker: MeetingSpeaker, _ start: Double, _ text: String) -> MeetingTranscriptSegment {
        .init(speaker: speaker, startSeconds: start, endSeconds: start + 1, text: text)
    }

    func testEarlierTextRevisionRebuildsOnlyAffectedSpeaker() {
        var cache = MeetingTranscriptListCache()
        let first = segment(.me, 0, "original")
        let second = segment(.them, 2, "other speaker")
        cache.updateOrdinals(for: [first, second])
        _ = cache.groups(for: [first, second], title: { $0.speaker.displayTitle })
        let revision = MeetingTranscriptSegment(id: first.id, speaker: .me, startSeconds: 0, endSeconds: 1, text: "revised earlier words")
        cache.updateOrdinals(for: [revision, second])
        let result = cache.groups(for: [revision, second], title: { $0.speaker.displayTitle })
        XCTAssertEqual(cache.ordinalRebuildCount, 1)
        XCTAssertEqual(cache.rebuiltGroupCount, 1)
        XCTAssertEqual(result.flatMap(\.segments).map(\.text), ["revised earlier words", "other speaker"])
        XCTAssertEqual(result.first?.wordCount, 3)
    }

    func testChronologicalAppendDoesNotResortPreviousOrdinals() {
        var cache = MeetingTranscriptListCache()
        let a = segment(.me, 0, "first")
        let b = segment(.them, 2, "new speaker")
        cache.updateOrdinals(for: [a])
        cache.updateOrdinals(for: [a, b])
        XCTAssertEqual(cache.ordinalRebuildCount, 1)
        XCTAssertEqual(cache.ordinals, MeetingTranscriptListSupport.speakerOrdinals(for: [a, b]))
        let earlierB = MeetingTranscriptSegment(id: b.id, speaker: .them, startSeconds: -1, endSeconds: 1, text: "corrected time")
        cache.updateOrdinals(for: [a, earlierB])
        XCTAssertEqual(cache.ordinalRebuildCount, 2)
        XCTAssertEqual(cache.ordinals, MeetingTranscriptListSupport.speakerOrdinals(for: [a, earlierB]))
    }

    func testDeletionUndoAndSearchSubsetCannotLeaveStaleGroups() {
        var cache = MeetingTranscriptListCache()
        let a = segment(.me, 0, "first")
        let b = segment(.them, 1, "second")
        for snapshot in [[a, b], [b], [a, b], [], [a]] {
            cache.updateOrdinals(for: snapshot)
            XCTAssertEqual(cache.ordinals, MeetingTranscriptListSupport.speakerOrdinals(for: snapshot))
            let result = cache.groups(for: snapshot, title: { $0.speaker.displayTitle })
            let reference = MeetingTranscriptListSupport.speakerGroups(from: snapshot, titleForSegment: { $0.speaker.displayTitle })
            XCTAssertEqual(result, reference)
        }
    }

    func testUnorderedUnchangedSnapshotReusesGroupsAndTitleChangeInvalidates() {
        var cache = MeetingTranscriptListCache()
        let snapshot = [segment(.me, 10, "later"), segment(.me, 0, "earlier")]
        cache.updateOrdinals(for: snapshot)
        _ = cache.groups(for: snapshot, title: { _ in "Speaker" })
        let unchanged = cache.groups(for: snapshot, title: { _ in "Speaker" })
        XCTAssertEqual(cache.rebuiltGroupCount, 0)
        XCTAssertEqual(unchanged.first?.segments.first?.text, "earlier")
        let renamed = cache.groups(for: snapshot, title: { _ in "Renamed" })
        XCTAssertEqual(cache.rebuiltGroupCount, 1)
        XCTAssertEqual(renamed.first?.title, "Renamed")
    }
}
