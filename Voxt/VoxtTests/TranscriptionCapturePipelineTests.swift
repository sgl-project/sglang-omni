// TranscriptionCapturePipelineTests.swift
// Provides Transcription Capture Pipeline Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class TranscriptionCapturePipelineTests: XCTestCase {
    func testResolveUsesFinalOnlyWhenRealtimeDisplayIsDisabled() {
        XCTAssertEqual(
            TranscriptionCapturePipeline.resolve(
                realtimeTextDisplayEnabled: false,
                captureSessionMode: .standard
            ),
            .finalOnly
        )
    }

    func testResolveUsesLiveDisplayWhenRealtimeDisplayIsEnabled() {
        XCTAssertEqual(
            TranscriptionCapturePipeline.resolve(
                realtimeTextDisplayEnabled: true,
                captureSessionMode: .standard
            ),
            .liveDisplay
        )
    }

    func testResolvePrefersNoteSessionPipeline() {
        XCTAssertEqual(
            TranscriptionCapturePipeline.resolve(
                realtimeTextDisplayEnabled: false,
                captureSessionMode: .noteSession
            ),
            .noteSession
        )
        XCTAssertEqual(
            TranscriptionCapturePipeline.resolve(
                realtimeTextDisplayEnabled: true,
                captureSessionMode: .noteSession
            ),
            .noteSession
        )
    }

    func testFinalOnlyStageLabelsAvoidLivePartialStage() {
        XCTAssertEqual(
            TranscriptionCapturePipeline.finalOnly.stageLabels,
            ["record", "stop", "finalASR", "enhance", "deliver"]
        )
    }

    func testLiveDisplayStageLabelsIncludeLivePartialStage() {
        XCTAssertEqual(
            TranscriptionCapturePipeline.liveDisplay.stageLabels,
            ["record", "livePartial", "stop", "previewASR", "finalASR", "enhance", "deliver"]
        )
    }

    func testNoteSessionStageLabelsIncludeNoteSegmentStage() {
        XCTAssertEqual(
            TranscriptionCapturePipeline.noteSession.stageLabels,
            ["record", "livePartial", "noteSegment", "stop", "finalASR", "enhance", "deliver"]
        )
    }

}
