// TranscriptionDoubleTapRewriteResolverTests.swift
// Provides Transcription Double Tap Rewrite Resolver Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class TranscriptionDoubleTapRewriteResolverTests: XCTestCase {
    func testRewriteDoubleTapWakeForcesTapTriggerMode() {
        let suiteName = UUID().uuidString
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)

        HotkeyPreference.saveRewriteActivationMode(.doubleTapTranscriptionHotkey, defaults: defaults)
        HotkeyPreference.saveTriggerMode(.longPress, defaults: defaults)

        XCTAssertEqual(HotkeyPreference.loadTriggerMode(defaults: defaults), .tap)
    }

    func testDoubleTapRewriteSchedulesTranscriptionOnFirstTap() {
        let action = TranscriptionDoubleTapRewriteResolver.resolve(
            state: .init(
                triggerMode: .tap,
                rewriteActivationMode: .doubleTapTranscriptionHotkey,
                isSessionActive: false,
                hasPendingTranscriptionStart: false
            )
        )

        XCTAssertEqual(action, .scheduleDelayedTranscriptionStart)
    }

    func testDoubleTapRewriteStartsRewriteOnSecondTap() {
        let action = TranscriptionDoubleTapRewriteResolver.resolve(
            state: .init(
                triggerMode: .tap,
                rewriteActivationMode: .doubleTapTranscriptionHotkey,
                isSessionActive: false,
                hasPendingTranscriptionStart: true
            )
        )

        XCTAssertEqual(action, .startRewrite)
    }

    func testDoubleTapRewriteFallsBackToStandardHandlingWhenLongPressIsActive() {
        let action = TranscriptionDoubleTapRewriteResolver.resolve(
            state: .init(
                triggerMode: .longPress,
                rewriteActivationMode: .doubleTapTranscriptionHotkey,
                isSessionActive: false,
                hasPendingTranscriptionStart: false
            )
        )

        XCTAssertEqual(action, .useStandardHandling)
    }

    func testDoubleTapRewriteFallsBackToStandardHandlingWhenSessionIsAlreadyActive() {
        let action = TranscriptionDoubleTapRewriteResolver.resolve(
            state: .init(
                triggerMode: .tap,
                rewriteActivationMode: .doubleTapTranscriptionHotkey,
                isSessionActive: true,
                hasPendingTranscriptionStart: true
            )
        )

        XCTAssertEqual(action, .useStandardHandling)
    }
}
