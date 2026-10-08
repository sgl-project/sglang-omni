// MLXModelPerRepoStateSupportTests.swift
// Provides MLXModel Per Repo State Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MLXModelPerRepoStateSupportTests: XCTestCase {
    func testResolvedStatePrefersStoredState() {
        let storedStates: [String: MLXModelManager.ModelState] = [
            MLXModelManager.canonicalModelRepo("mlx-community/Qwen3-ASR-0.6B-4bit"): .downloaded
        ]
        let state = MLXModelPerRepoStateSupport.resolvedState(
            for: "mlx-community/Qwen3-ASR-0.6B-4bit",
            currentRepo: MLXModelManager.defaultModelRepo,
            currentState: MLXModelManager.ModelState.notDownloaded,
            storedStates: storedStates,
            isDownloaded: { _ in false },
            hasResumableDownload: { _ in false }
        )

        XCTAssertEqual(state, MLXModelManager.ModelState.downloaded)
    }

    func testResolvedStateDerivesPausedForResumableNonCurrentRepo() {
        let repo = "mlx-community/parakeet-tdt-0.6b-v3"
        let state = MLXModelPerRepoStateSupport.resolvedState(
            for: repo,
            currentRepo: MLXModelManager.defaultModelRepo,
            currentState: MLXModelManager.ModelState.notDownloaded,
            storedStates: [String: MLXModelManager.ModelState](),
            isDownloaded: { _ in false },
            hasResumableDownload: { candidate in
                MLXModelManager.canonicalModelRepo(candidate) == MLXModelManager.canonicalModelRepo(repo)
            }
        )

        guard case .paused = state else {
            return XCTFail("Expected paused state for resumable repo, got \(state)")
        }
    }

    func testApplyPausedStatusMessageUpdatesStoredAndCurrentValues() {
        var currentMessage: String?
        var storedMessages = [String: String]()
        let repo = MLXModelManager.defaultModelRepo

        MLXModelPerRepoStateSupport.applyPausedStatusMessage(
            "Resume available",
            for: repo,
            currentRepo: repo,
            currentMessage: &currentMessage,
            storedMessages: &storedMessages
        )

        XCTAssertEqual(currentMessage, "Resume available")
        XCTAssertEqual(
            storedMessages[MLXModelManager.canonicalModelRepo(repo)],
            "Resume available"
        )
    }

    func testClearStateRemovesStoredValuesAndCurrentPausedMessage() {
        let repo = MLXModelManager.defaultModelRepo
        var currentMessage: String? = "Resume available"
        var storedStates: [String: MLXModelManager.ModelState] = [
            MLXModelManager.canonicalModelRepo(repo): .downloaded
        ]
        var storedMessages = [
            MLXModelManager.canonicalModelRepo(repo): "Resume available"
        ]

        MLXModelPerRepoStateSupport.clearState(
            for: repo,
            currentRepo: repo,
            currentPausedStatusMessage: &currentMessage,
            storedStates: &storedStates,
            storedMessages: &storedMessages
        )

        XCTAssertTrue(storedStates.isEmpty)
        XCTAssertTrue(storedMessages.isEmpty)
        XCTAssertNil(currentMessage)
    }
}
