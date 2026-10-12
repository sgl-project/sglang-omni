// MeetingTranscriptVirtualListTests.swift
// Tests for meeting detail virtual transcript rows and height cache.

import XCTest
@testable import Voxt

final class MeetingTranscriptVirtualListTests: XCTestCase {
    func testDisplayedSegmentsFiltersByTextAndTranslation() {
        let segments = [
            MeetingTranscriptSegment(
                speaker: .me,
                startSeconds: 0,
                endSeconds: 2,
                text: "hello world"
            ),
            MeetingTranscriptSegment(
                id: UUID(),
                speaker: .them,
                startSeconds: 2,
                endSeconds: 4,
                text: "unrelated",
                translatedText: "bonjour"
            ),
        ]

        let filtered = MeetingTranscriptListSupport.displayedSegments(
            from: segments,
            searchQuery: "bonjour",
            speakerTitle: { $0.speaker.displayTitle }
        )

        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered.first?.translatedText, "bonjour")
    }

    func testSpeakerMarkRowsFlattenHeadersAndSegments() {
        let first = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 1,
            text: "one"
        )
        let second = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 2,
            endSeconds: 3,
            text: "two"
        )
        let groups = MeetingTranscriptListSupport.speakerGroups(
            from: [first, second],
            titleForSegment: { $0.speaker.displayTitle }
        )
        let rows = MeetingTranscriptListSupport.speakerMarkRows(
            from: groups,
            activeSegmentID: second.id,
            showsTranslation: false,
            searchQuery: ""
        )

        XCTAssertEqual(rows.count, 4)
        guard case .speakerHeader = rows[0] else {
            return XCTFail("Expected speaker header at 0")
        }
        guard case .segment(let firstPresentation) = rows[1] else {
            return XCTFail("Expected segment at 1")
        }
        XCTAssertEqual(firstPresentation.segment.id, groups[0].segments[0].id)
        guard case .speakerHeader = rows[2] else {
            return XCTFail("Expected speaker header at 2")
        }
        guard case .segment(let secondPresentation) = rows[3] else {
            return XCTFail("Expected segment at 3")
        }
        XCTAssertTrue(secondPresentation.isActive)
        XCTAssertEqual(secondPresentation.segment.id, second.id)
    }

    func testTimelineRowsMarkActiveAndSearchMatch() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 5,
            endSeconds: 8,
            text: "release checklist"
        )
        let rows = MeetingTranscriptListSupport.timelineRows(
            from: [segment],
            activeSegmentID: segment.id,
            showsTranslation: true,
            searchQuery: "checklist",
            speakerTitle: { _ in "Them" }
        )

        XCTAssertEqual(rows.count, 1)
        guard case .segment(let presentation) = rows[0] else {
            return XCTFail("Expected segment row")
        }
        XCTAssertTrue(presentation.isActive)
        XCTAssertTrue(presentation.isSearchMatch)
        XCTAssertTrue(presentation.showsTranslation)
    }

    func testContentChangedIndexesDetectsActiveToggleOnly() {
        let segment = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 1,
            text: "alpha"
        )
        let inactive = MeetingTranscriptListSupport.timelineRows(
            from: [segment],
            activeSegmentID: nil,
            showsTranslation: false,
            searchQuery: "",
            speakerTitle: { _ in "Me" }
        )
        let active = MeetingTranscriptListSupport.timelineRows(
            from: [segment],
            activeSegmentID: segment.id,
            showsTranslation: false,
            searchQuery: "",
            speakerTitle: { _ in "Me" }
        )

        let changed = MeetingTranscriptListSupport.contentChangedIndexes(from: inactive, to: active)
        XCTAssertEqual(changed, IndexSet(integer: 0))
    }

    func testContentChangedIndexesReturnsNilOnIdentityChange() {
        let first = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 1,
            text: "a"
        )
        let second = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 1,
            endSeconds: 2,
            text: "b"
        )
        let oldRows = MeetingTranscriptListSupport.timelineRows(
            from: [first],
            activeSegmentID: nil,
            showsTranslation: false,
            searchQuery: "",
            speakerTitle: { $0.speaker.displayTitle }
        )
        let newRows = MeetingTranscriptListSupport.timelineRows(
            from: [second],
            activeSegmentID: nil,
            showsTranslation: false,
            searchQuery: "",
            speakerTitle: { $0.speaker.displayTitle }
        )

        XCTAssertNil(MeetingTranscriptListSupport.contentChangedIndexes(from: oldRows, to: newRows))
        XCTAssertFalse(MeetingTranscriptListSupport.isSuffixAppend(from: oldRows, to: newRows))
    }

    func testSuffixAppendDetection() {
        let first = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 1,
            text: "a"
        )
        let second = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 1,
            endSeconds: 2,
            text: "b"
        )
        let oldRows = MeetingTranscriptListSupport.timelineRows(
            from: [first],
            activeSegmentID: nil,
            showsTranslation: false,
            searchQuery: "",
            speakerTitle: { $0.speaker.displayTitle }
        )
        let newRows = MeetingTranscriptListSupport.timelineRows(
            from: [first, second],
            activeSegmentID: nil,
            showsTranslation: false,
            searchQuery: "",
            speakerTitle: { $0.speaker.displayTitle }
        )

        XCTAssertTrue(MeetingTranscriptListSupport.isSuffixAppend(from: oldRows, to: newRows))
        XCTAssertTrue(MeetingTranscriptListSupport.isSuffixAppend(from: [], to: newRows))
    }

    func testHeightCacheStableForSameContentAndInvalidateOnTranslation() {
        var cache = MeetingTranscriptRowHeightCache()
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 3,
            text: String(repeating: "hello ", count: 40),
            translatedText: String(repeating: "bonjour ", count: 40)
        )
        let row = MeetingTranscriptVirtualRow.segment(
            MeetingTranscriptSegmentPresentation(
                segment: segment,
                speakerTitle: "Them",
                isActive: false,
                showsTranslation: false,
                isSearchMatch: false
            )
        )

        let withoutTranslation = cache.cachedHeight(for: row, width: 420, showsTranslation: false)
        let again = cache.cachedHeight(for: row, width: 420, showsTranslation: false)
        XCTAssertEqual(withoutTranslation, again)

        let withTranslation = MeetingTranscriptRowHeightCache.estimate(
            for: row,
            width: 420,
            showsTranslation: true
        )
        XCTAssertGreaterThan(withTranslation, withoutTranslation)

        let taller = ceil(withoutTranslation + 12)
        XCTAssertTrue(
            cache.storeMeasuredHeight(taller, for: row, width: 420, showsTranslation: false)
        )
        XCTAssertEqual(cache.height(for: row, width: 420, showsTranslation: false), taller)

        // On-screen GeometryReader reports may shrink overestimated rows.
        let refined = ceil(withoutTranslation - 8)
        XCTAssertTrue(
            cache.storeReportedHeight(refined, for: row, width: 420, showsTranslation: false)
        )
        XCTAssertEqual(cache.height(for: row, width: 420, showsTranslation: false), refined)
        XCTAssertTrue(cache.hasReportedHeight(for: row, width: 420, showsTranslation: false))

        // Unreliable widths must not poison the cache.
        XCTAssertFalse(
            cache.storeMeasuredHeight(999, for: row, width: 40, showsTranslation: false)
        )

        cache.invalidateWidth(420)
        XCTAssertFalse(cache.hasMeasuredHeight(for: row, width: 420, showsTranslation: false))
        XCTAssertEqual(
            cache.height(for: row, width: 420, showsTranslation: false),
            MeetingTranscriptRowHeightCache.estimate(for: row, width: 420, showsTranslation: false)
        )
    }

    func testMultiLineSegmentEstimateExceedsSingleLine() {
        let short = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 1,
            text: "Short"
        )
        let long = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 1,
            endSeconds: 2,
            text: String(repeating: "This is a longer transcript line that should wrap. ", count: 6)
        )
        let shortHeight = MeetingTranscriptRowHeightCache.estimateSegmentHeight(
            text: short.text,
            translatedText: nil,
            isTranslationPending: false,
            width: 360,
            showsTranslation: false
        )
        let longHeight = MeetingTranscriptRowHeightCache.estimateSegmentHeight(
            text: long.text,
            translatedText: nil,
            isTranslationPending: false,
            width: 360,
            showsTranslation: false
        )
        XCTAssertGreaterThan(longHeight, shortHeight + 20)
    }

    func testNarrowWidthEstimateIsTallerThanWideWidth() {
        let segment = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 0,
            endSeconds: 3,
            text: String(repeating: "hello world ", count: 24)
        )
        let row = MeetingTranscriptVirtualRow.segment(
            MeetingTranscriptSegmentPresentation(
                segment: segment,
                speakerTitle: "Them",
                isActive: false,
                showsTranslation: false,
                isSearchMatch: false
            )
        )

        let narrow = MeetingTranscriptRowHeightCache.estimate(
            for: row,
            width: 180,
            showsTranslation: false
        )
        let wide = MeetingTranscriptRowHeightCache.estimate(
            for: row,
            width: 520,
            showsTranslation: false
        )
        XCTAssertGreaterThan(narrow, wide)
    }

    func testSpeakerHeaderHeightIsCompact() {
        let row = MeetingTranscriptVirtualRow.speakerHeader(id: "me", title: "Me", count: 3)
        let height = MeetingTranscriptRowHeightCache.estimate(
            for: row,
            width: 400,
            showsTranslation: false
        )
        XCTAssertEqual(height, 28)
    }
}

@MainActor
final class MeetingDetailTranscriptListCacheTests: XCTestCase {
    func testViewModelCachesDisplayedSegmentsAndSpeakerGroups() {
        let them = MeetingTranscriptSegment(
            speaker: .them,
            startSeconds: 1,
            endSeconds: 2,
            text: "agenda"
        )
        let me = MeetingTranscriptSegment(
            speaker: .me,
            startSeconds: 0,
            endSeconds: 1,
            text: "intro"
        )
        let viewModel = MeetingDetailViewModel(
            title: "Meeting Details",
            subtitle: "Today",
            historyEntryID: UUID(),
            initialSummary: nil,
            initialSummaryChatMessages: [],
            initialSummarySettings: MeetingSummarySettingsSnapshot(
                autoGenerate: false,
                promptTemplate: "prompt",
                modelSelectionID: nil
            ),
            summaryModelOptions: [],
            summarySettingsProvider: {
                MeetingSummarySettingsSnapshot(
                    autoGenerate: false,
                    promptTemplate: "prompt",
                    modelSelectionID: nil
                )
            },
            summaryModelOptionsProvider: { [] },
            segments: [me, them],
            audioURL: nil,
            translationHandler: { text, _ in
                MeetingTranslationOperation(executionScope: .externalRequest) { text }
            },
            summaryStatusProvider: { _ in
                MeetingSummaryProviderStatus(isAvailable: false, message: "Unavailable")
            },
            summaryGenerator: { _, _ in
                throw NSError(domain: "test", code: 1)
            },
            summaryPersistence: { _, _ in nil },
            summaryChatAnswerer: { _, _, _, _, _ in "" },
            summaryChatPersistence: { _, _ in nil },
            transcriptSegmentsPersistence: { _, _ in nil }
        )

        XCTAssertEqual(viewModel.displayedSegments.count, 2)
        XCTAssertEqual(viewModel.speakerGroups.count, 2)

        viewModel.setSearchQuery("agenda")
        XCTAssertEqual(viewModel.displayedSegments.map(\.text), ["agenda"])
        XCTAssertEqual(viewModel.speakerGroups.count, 1)
        XCTAssertEqual(viewModel.displayedNewestSegmentID(), them.id)
    }
}
