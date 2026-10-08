// RecordingSessionSupportTests.swift
// Provides Recording Session Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RecordingSessionSupportTests: XCTestCase {
    func testFallbackInjectBundleIDRejectsOwnAppBundleID() {
        XCTAssertNil(
            RecordingSessionSupport.fallbackInjectBundleID(
                from: "com.voxt.Voxt.dev",
                ownBundleID: "com.voxt.Voxt.dev"
            )
        )
        XCTAssertEqual(
            RecordingSessionSupport.fallbackInjectBundleID(
                from: "com.apple.TextEdit",
                ownBundleID: "com.voxt.Voxt.dev"
            ),
            "com.apple.TextEdit"
        )
    }

    func testExtractTranscriptionTextValuePrefersKnownKeysAndNestedContent() {
        let payload: [String: Any] = [
            "metadata": ["ignored": true],
            "data": [
                "content": [
                    ["text": "  hello world  "]
                ]
            ]
        ]

        XCTAssertEqual(
            RecordingSessionSupport.extractTranscriptionTextValue(from: payload),
            "hello world"
        )
    }

    func testPromptEchoSuppressionDropsGeneratedASRPrompt() {
        let prompt = """
        The speaker's primary language is Simplified Chinese, and they may also speak English. Mixed-language speech is expected. Preserve names, product terms, URLs, and code-like text exactly as spoken.
        """

        XCTAssertTrue(RecordingSessionSupport.isLikelyPromptEcho(prompt))
        XCTAssertEqual(RecordingSessionSupport.textAfterSuppressingPromptEcho(prompt), "")
    }

    func testPromptEchoSuppressionDropsCustomPromptEcho() {
        let prompt = "Always preserve the phrase Voxt Server and product names exactly when transcribing user speech."
        let echoed = "Always preserve the phrase Voxt Server and product names exactly when transcribing user speech."

        XCTAssertTrue(RecordingSessionSupport.isLikelyPromptEcho(echoed, prompt: prompt))
        XCTAssertEqual(RecordingSessionSupport.textAfterSuppressingPromptEcho(echoed, prompt: prompt), "")
    }

    func testPromptEchoSuppressionPreservesNormalTranscription() {
        let text = "历史记录文本点击后会复制，复制成功后给出一个 toast 提示。"

        XCTAssertFalse(RecordingSessionSupport.isLikelyPromptEcho(text))
        XCTAssertEqual(RecordingSessionSupport.textAfterSuppressingPromptEcho(text), text)
    }

    func testOutputLabelAndOverlayIconModeStayAligned() {
        XCTAssertEqual(RecordingSessionSupport.outputLabel(for: .transcription), "transcription")
        XCTAssertEqual(RecordingSessionSupport.overlayIconMode(for: .transcription), .transcription)
        XCTAssertEqual(
            RecordingSessionSupport.overlayIconMode(for: .transcription, isNoteSession: true),
            .note
        )
        XCTAssertEqual(RecordingSessionSupport.outputLabel(for: .translation), "translation")
        XCTAssertEqual(RecordingSessionSupport.overlayIconMode(for: .translation), .translation)
        XCTAssertEqual(RecordingSessionSupport.outputLabel(for: .rewrite), "rewrite")
        XCTAssertEqual(RecordingSessionSupport.overlayIconMode(for: .rewrite), .rewrite)
    }
}
