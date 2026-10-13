// FeaturePromptDraftCoordinatorTests.swift
// Provides Feature Prompt Draft Coordinator Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class FeaturePromptDraftCoordinatorTests: XCTestCase {
    func testInitSeedsDraftAndSyncedText() {
        let coordinator = FeaturePromptDraftCoordinator(text: "Base prompt")

        XCTAssertEqual(coordinator.draft, "Base prompt")
        XCTAssertEqual(coordinator.lastSyncedText, "Base prompt")
    }

    func testTakePendingPersistReturnsNilWhenDraftMatchesSyncedText() {
        var coordinator = FeaturePromptDraftCoordinator(text: "Base prompt")

        XCTAssertNil(coordinator.takePendingPersist())
    }

    func testTakePendingPersistReturnsDraftAndAdvancesSyncedText() {
        var coordinator = FeaturePromptDraftCoordinator(text: "Base prompt")
        coordinator.updateDraft("Edited prompt")

        XCTAssertEqual(coordinator.takePendingPersist(), "Edited prompt")
        XCTAssertEqual(coordinator.lastSyncedText, "Edited prompt")
        XCTAssertNil(coordinator.takePendingPersist())
    }

    func testTakePendingPersistSkipsStaleDebouncePayload() {
        var coordinator = FeaturePromptDraftCoordinator(text: "Base prompt")
        coordinator.updateDraft("Edited once")
        coordinator.updateDraft("Edited twice")

        XCTAssertNil(coordinator.takePendingPersist(expectedText: "Edited once"))
        XCTAssertEqual(coordinator.lastSyncedText, "Base prompt")
        XCTAssertEqual(coordinator.takePendingPersist(expectedText: "Edited twice"), "Edited twice")
    }

    func testSyncExternalTextIgnoresRoundTripEchoOfOwnWrite() {
        var coordinator = FeaturePromptDraftCoordinator(text: "Base prompt")
        coordinator.updateDraft("Edited prompt")
        XCTAssertEqual(coordinator.takePendingPersist(), "Edited prompt")

        coordinator.syncExternalText("Edited prompt")

        XCTAssertEqual(coordinator.draft, "Edited prompt")
        XCTAssertEqual(coordinator.lastSyncedText, "Edited prompt")
    }

    func testSyncExternalTextAppliesRealExternalMutation() {
        var coordinator = FeaturePromptDraftCoordinator(text: "Base prompt")

        coordinator.syncExternalText("External prompt")

        XCTAssertEqual(coordinator.draft, "External prompt")
        XCTAssertEqual(coordinator.lastSyncedText, "External prompt")
        XCTAssertNil(coordinator.takePendingPersist())
    }

    func testPendingPersistPreservesUnicodeSpacesAndNewlinesExactly() {
        var coordinator = FeaturePromptDraftCoordinator(text: "Base prompt")
        let editedPrompt = "请保留 空格\n并保留换行。日本語も保持する"

        coordinator.updateDraft(editedPrompt)

        XCTAssertEqual(coordinator.takePendingPersist(), editedPrompt)
        XCTAssertEqual(coordinator.draft, editedPrompt)
        XCTAssertEqual(coordinator.lastSyncedText, editedPrompt)
    }

    func testRoundTripEchoDoesNotMoveDraftAfterMultilineEdit() {
        var coordinator = FeaturePromptDraftCoordinator(text: "第一行")
        let editedPrompt = "第一行\n第二行 最后一个字"
        coordinator.updateDraft(editedPrompt)
        XCTAssertEqual(coordinator.takePendingPersist(), editedPrompt)

        coordinator.syncExternalText(editedPrompt)

        XCTAssertEqual(coordinator.draft, editedPrompt)
        XCTAssertNil(coordinator.takePendingPersist())
    }
}

final class PromptTextEditorUpdatePolicyTests: XCTestCase {
    func testMarkedTextIsNotPublishedToAutosave() {
        XCTAssertFalse(
            PromptTextEditorUpdatePolicy.shouldPublishChange(hasMarkedText: true)
        )
    }

    func testCommittedTextIsPublishedToAutosave() {
        XCTAssertTrue(
            PromptTextEditorUpdatePolicy.shouldPublishChange(hasMarkedText: false)
        )
    }

    func testAutosaveRoundTripDoesNotReassignIdenticalText() {
        XCTAssertFalse(
            PromptTextEditorUpdatePolicy.shouldApplyExternalText(
                currentText: "用户正在输入",
                externalText: "用户正在输入",
                hasMarkedText: false
            )
        )
    }

    func testExternalStateCannotReplaceActiveInputMethodComposition() {
        XCTAssertFalse(
            PromptTextEditorUpdatePolicy.shouldApplyExternalText(
                currentText: "用户正在shuru",
                externalText: "用户正在",
                hasMarkedText: true
            )
        )
    }

    func testRealExternalChangeIsAppliedAfterCompositionCommits() {
        XCTAssertTrue(
            PromptTextEditorUpdatePolicy.shouldApplyExternalText(
                currentText: "旧提示词",
                externalText: "新提示词\n第二行",
                hasMarkedText: false
            )
        )
    }
}
