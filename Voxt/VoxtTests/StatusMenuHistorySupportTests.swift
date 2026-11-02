import AppKit
import XCTest
@testable import Voxt

@MainActor
final class StatusMenuHistorySupportTests: XCTestCase {
    func testHistoryRoutesIncludeAllEnabledCategories() {
        XCTAssertEqual(
            StatusMenuHistorySupport.filters(availability: .allEnabled),
            [.transcription, .translation, .rewrite, .note, .transcript]
        )
    }

    func testDisabledFeaturesHideTheirHistoryRoutes() {
        let availability = FeatureAvailabilitySettings(
            translationEnabled: false,
            rewriteEnabled: false,
            notesEnabled: false,
            appEnhancementEnabled: false,
            meetingEnabled: false,
            filesEnabled: false
        )
        XCTAssertEqual(StatusMenuHistorySupport.filters(availability: availability), [.transcription])
    }

    func testRecentEntriesMergeKindsBeforeApplyingFiveItemLimit() {
        let entries = (0..<8).map { makeEntry(index: $0, kind: $0.isMultiple(of: 2) ? .normal : .translation) }
        let unrelated = [makeEntry(index: -2, kind: .rewrite), makeEntry(index: -1, kind: .transcript)]
        let result = StatusMenuHistorySupport.recentEntries(
            from: Array(entries.reversed()) + unrelated,
            availability: .allEnabled
        )
        XCTAssertEqual(result.map(\.id), Array(entries.prefix(5)).map(\.id))
    }

    func testDisablingTranslationBackfillsWithTranscriptionRecords() {
        let transcriptions = (10..<15).map { makeEntry(index: $0) }
        let translations = (0..<5).map { makeEntry(index: $0, kind: .translation) }
        let result = StatusMenuHistorySupport.recentEntries(
            from: translations + transcriptions,
            availability: FeatureAvailabilitySettings(translationEnabled: false)
        )
        XCTAssertEqual(result.map(\.id), transcriptions.map(\.id))
    }

    func testEmptyAndShortHistoryDoNotCreateExtraRecords() {
        XCTAssertTrue(StatusMenuHistorySupport.recentEntries(from: [], availability: .allEnabled).isEmpty)
        let entry = makeEntry(index: 0)
        XCTAssertEqual(StatusMenuHistorySupport.recentEntries(from: [entry], availability: .allEnabled), [entry])
    }

    func testPreviewCollapsesWhitespaceWithoutChangingSourceText() {
        let entry = makeEntry(index: 0, text: "  first\n\tsecond  ")
        XCTAssertEqual(StatusMenuHistorySupport.previewTitle(for: entry), "first second")
        XCTAssertEqual(entry.previewText, "  first\n\tsecond  ")
    }

    func testPreviewTruncatesAtRenderedWidthAndMarksRepositoryTruncation() {
        let emoji = "👨‍👩‍👧‍👦"
        XCTAssertEqual(StatusMenuHistorySupport.previewTitle(for: makeEntry(index: 0, text: emoji)), emoji)
        let entry = makeEntry(index: 0, text: String(repeating: emoji, count: 49))
        let preview = StatusMenuHistorySupport.previewTitle(for: entry)
        XCTAssertTrue(preview.hasSuffix("…"))
        XCTAssertLessThan(preview.dropLast().count, 49)
        let emojiCharacter = emoji.first!
        XCTAssertTrue(preview.dropLast().allSatisfy { $0 == emojiCharacter })

        let partial = makeEntry(index: 1, text: "short preview", textLength: 1_000)
        XCTAssertEqual(StatusMenuHistorySupport.previewTitle(for: partial), "short preview…")
    }

    private func makeEntry(
        index: Int,
        kind: TranscriptionHistoryKind = .normal,
        text: String = "text",
        textLength: Int? = nil
    ) -> TranscriptionHistoryListEntry {
        TranscriptionHistoryListEntry(
            id: UUID(),
            previewText: text,
            textLength: textLength ?? text.unicodeScalars.count,
            createdAt: Date(timeIntervalSince1970: 1_000 - Double(index)),
            kind: kind,
            displayTitle: nil
        )
    }
}
