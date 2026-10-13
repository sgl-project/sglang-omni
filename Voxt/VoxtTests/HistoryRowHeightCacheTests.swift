import XCTest
@testable import Voxt

final class HistoryRowHeightCacheTests: XCTestCase {
    func testShortTextUsesMinimumHeight() {
        let height = HistoryRowHeightCache.estimate(
            text: "Short transcription",
            width: 520,
            verticalInset: 4
        )

        XCTAssertEqual(height, HistoryRowHeightCache.minimumHeight)
    }

    func testWrappedTextGrowsButStopsAtThreeLines() {
        let shortHeight = HistoryRowHeightCache.estimate(
            text: "Short transcription",
            width: 360,
            verticalInset: 4
        )
        let longHeight = HistoryRowHeightCache.estimate(
            text: String(repeating: "This is a long history result that should wrap. ", count: 8),
            width: 360,
            verticalInset: 4
        )
        let maximumHeight = ceil(
            HistoryRowHeightCache.textLineHeight * CGFloat(HistoryRowHeightCache.maximumTextLines)
                + HistoryRowHeightCache.textLineSpacing * CGFloat(HistoryRowHeightCache.maximumTextLines - 1)
                + HistoryRowHeightCache.rowVerticalPadding
                + 8
        )

        XCTAssertGreaterThan(longHeight, shortHeight)
        XCTAssertLessThanOrEqual(longHeight, maximumHeight)
    }

    func testNarrowWidthProducesTallerRows() {
        let text = String(repeating: "A narrow history column needs more lines. ", count: 3)
        let narrowHeight = HistoryRowHeightCache.estimate(
            text: text,
            width: 260,
            verticalInset: 4
        )
        let wideHeight = HistoryRowHeightCache.estimate(
            text: text,
            width: 620,
            verticalInset: 4
        )

        XCTAssertGreaterThan(narrowHeight, wideHeight)
    }

    func testWidthBucketsAreStableAndPositive() {
        XCTAssertEqual(HistoryRowHeightCache.widthBucket(420), 424)
        XCTAssertEqual(HistoryRowHeightCache.widthBucket(423), 424)
        XCTAssertEqual(HistoryRowHeightCache.widthBucket(0), 1)
    }
}
