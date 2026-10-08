// LiveTranscriptSegmentationStateTests.swift
// Provides Live Transcript Segmentation State Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LiveTranscriptSegmentationStateTests: XCTestCase {
    func testVisibleTextReturnsFullTranscriptBeforeAnyFreeze() {
        var state = LiveTranscriptSegmentationState()

        XCTAssertEqual(state.visibleText(for: "hello world"), "hello world")
        XCTAssertEqual(state.currentVisibleText, "hello world")
    }

    func testFreezeCurrentSegmentKeepsOnlyNewSuffixVisible() {
        var state = LiveTranscriptSegmentationState()
        _ = state.visibleText(for: "hello world")

        XCTAssertEqual(state.freezeCurrentSegment(), "hello world")
        XCTAssertEqual(state.visibleText(for: "hello world again"), "again")
    }

    func testVisibleTextHandlesPunctuationDifferencesAcrossASRCorrections() {
        var state = LiveTranscriptSegmentationState()
        _ = state.visibleText(for: "今天下午三点开会")
        _ = state.freezeCurrentSegment()

        XCTAssertEqual(
            state.visibleText(for: "今天下午三点开会，我们再看一下预算"),
            "我们再看一下预算"
        )
    }

    func testVisibleTextFallsBackToCanonicalCharacterBoundaryWhenASRRewritesFrozenPrefix() {
        var state = LiveTranscriptSegmentationState()
        _ = state.visibleText(for: "数字123")
        _ = state.freezeCurrentSegment()

        XCTAssertEqual(
            state.visibleText(for: "数字一二三，新增APP"),
            "新增APP"
        )
    }

    func testDisplayTextShowsBoundaryMarkerInlineWithoutHidingHistory() {
        var state = LiveTranscriptSegmentationState()
        _ = state.visibleText(for: "数字123")
        _ = state.freezeCurrentSegment(markerLabel: "00:05")

        XCTAssertEqual(
            state.displayText(for: "数字一二三，新增APP"),
            "数字一二三， • 00:05 新增APP"
        )
    }

    func testDisplayTextSupportsMultipleBoundaryMarkers() {
        var state = LiveTranscriptSegmentationState()
        _ = state.visibleText(for: "第一条")
        _ = state.freezeCurrentSegment(markerLabel: "00:03")
        _ = state.visibleText(for: "第一条第二条")
        _ = state.freezeCurrentSegment(markerLabel: "00:07")

        XCTAssertEqual(
            state.displayText(for: "第一条第二条第三条"),
            "第一条 • 00:03 第二条 • 00:07 第三条"
        )
    }
}
